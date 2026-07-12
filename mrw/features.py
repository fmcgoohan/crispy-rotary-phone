"""Stage 3 — audio features per PLAN §4 stage 3; contract:
schemas/audio_features.schema.json v1.0.0 (frozen).

Reviewed conventions implemented here:
- 10 ms hop (100 Hz grid), analysis window 2048 (PLAN §4.3).
- Tempo/beat grid computed once on the mix; downbeats via
  `assumed_4_4_phase_fit` (OQ-4) — the beat phase (0..3) maximizing mean
  onset strength, ties to the lowest phase.
- rms_db: dBFS, silence floor clamped at -80.0.
- spectral_centroid_hz: silent frames hold the last valid centroid, frames
  before the first valid one are backfilled from it (review 001 R-2); a
  channel silent throughout writes all zeros (review 002 R-9).
- onsets: strengths normalized by the p98 of the channel's onset-strength
  envelope, clamped to [0, 1] (R-1); `strength_reference: 1.0` sentinel when
  no onsets were detected (R-9).
- loudness: EBU R128 integrated LUFS; a fully-silent program returns -inf
  from the meter and is clamped to -70.0 (the R128 absolute gate) so the
  document stays valid JSON.
- vocal_activity: hysteresis over the vocal stem's rms_db with the params
  embedded in the document (R-5 / OQ-5).

All values rounded per the precision contract before serialization.

Determinism policy (D5): unlike stems (raw float artifacts, thread-pinned on
CPU), this stage emits documents only — the precision-contract rounding IS
the jitter policy (PLAN §7 layer 1). Last-ulp float differences from BLAS
thread ordering in librosa/numpy/pyloudnorm sit far below the coarsest
rounding step (0.001 s / 0.01 dB / 0.1 Hz), so no thread pinning is needed;
the double-run byte-identity smoke test is the standing evidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import TOOL_VERSION, canonical, config
from .library import Library, PrerequisiteError
from .models import (
    AudioFeaturesDocument,
    Beats,
    ChannelFeatures,
    DocumentEntry,
    Loudness,
    Onsets,
    RunMetadata,
    StemFeatures,
    Tempo,
    Timeseries,
    VocalActivity,
    VocalActivityParams,
    VocalRegion,
)
from .stems import STEM_NAMES

SAMPLE_RATE = 44100
HOP = 441  # 10 ms at 44.1 kHz — the 100 Hz uniform grid (PLAN §4.3)
HOP_SECONDS = 0.01
N_FFT = 2048
SILENCE_FLOOR_DB = -80.0


class FeaturesError(RuntimeError):
    """Stage failure — recorded in the manifest as status=failed; exit 1."""


@dataclass
class FeaturesResult:
    track_id: str
    bpm_global: float
    n_vocal_regions: int
    already_done: bool


def _load(path: Path):
    import numpy as np
    import soundfile as sf

    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise FeaturesError(f"{path.name} is {sample_rate} Hz; expected {SAMPLE_RATE}")
    return data.mean(axis=1).astype(np.float32), data


def _rms_db(y):
    import librosa
    import numpy as np

    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP, center=True)[0]
    db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    return np.maximum(db, SILENCE_FLOOR_DB)


def _centroid_hz(y, rms_db):
    """Centroid with hold-last + backfill over silent frames (R-2/R-9)."""
    import librosa
    import numpy as np

    cent = librosa.feature.spectral_centroid(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP, center=True
    )[0]
    silent = rms_db <= SILENCE_FLOOR_DB + 1e-9
    if silent.all():
        return np.zeros_like(cent)  # R-9: silent throughout → all zeros
    n = len(cent)
    last_valid = np.where(~silent, np.arange(n), -1)
    np.maximum.accumulate(last_valid, out=last_valid)
    first_valid = int(np.flatnonzero(~silent)[0])
    last_valid[last_valid < 0] = first_valid  # backfill the leading silence
    return cent[last_valid]


def _onsets(y) -> Onsets:
    import librosa
    import numpy as np

    env = librosa.onset.onset_strength(y=y, sr=SAMPLE_RATE, hop_length=HOP)
    frames = librosa.onset.onset_detect(
        onset_envelope=env, sr=SAMPLE_RATE, hop_length=HOP, backtrack=False
    )
    if len(frames) == 0:
        return Onsets(times=[], strengths=[], strength_reference=1.0)  # R-9 sentinel
    reference = float(np.percentile(env, 98))
    # Keep the schema's exclusiveMinimum satisfied even for whisper-quiet
    # channels whose p98 would round to 0.000.
    reference = max(canonical.round_ratio(reference), 0.001)
    times = librosa.frames_to_time(frames, sr=SAMPLE_RATE, hop_length=HOP)
    strengths = np.clip(env[frames] / reference, 0.0, 1.0)
    return Onsets(
        times=[canonical.round_seconds(t) for t in times],
        strengths=[canonical.round_ratio(s) for s in strengths],
        strength_reference=reference,
    )


def _timeseries(values, unit: str, rounder) -> Timeseries:
    return Timeseries(
        unit=unit,
        start_seconds=0.0,
        hop_seconds=HOP_SECONDS,
        values=[rounder(v) for v in values],
    )


def _channel_features(y) -> ChannelFeatures:
    rms = _rms_db(y)
    cent = _centroid_hz(y, rms)
    return ChannelFeatures(
        rms_db=_timeseries(rms, "dbfs", canonical.round_db),
        spectral_centroid_hz=_timeseries(cent, "hz", canonical.round_hz),
        onsets=_onsets(y),
    )


def _beats_and_tempo(y_mix) -> tuple[Tempo, Beats]:
    import librosa
    import numpy as np

    env = librosa.onset.onset_strength(y=y_mix, sr=SAMPLE_RATE, hop_length=HOP)
    bpm, beat_frames = librosa.beat.beat_track(
        onset_envelope=env, sr=SAMPLE_RATE, hop_length=HOP, trim=False
    )
    bpm = float(np.atleast_1d(bpm)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=SAMPLE_RATE, hop_length=HOP)

    # Confidence heuristic: regularity of inter-beat intervals (coefficient
    # of variation). A metronomic grid scores ~1.0; an erratic one → 0.
    if len(beat_times) >= 3:
        intervals = np.diff(beat_times)
        cv = float(intervals.std() / max(intervals.mean(), 1e-9))
        confidence = max(0.0, min(1.0, 1.0 - 4.0 * cv))
    else:
        confidence = 0.0

    # OQ-4: assumed-4/4 phase fit — pick the beat phase whose every-4th
    # beats have the highest mean onset strength; ties take the lowest phase
    # (np.argmax's first-max rule) for determinism. Strength is the local
    # max in a ±30 ms window around each beat: tracked beats sit a frame or
    # two off the onset-energy peak, and sampling the envelope exactly at
    # the beat frame reads pre-onset noise instead of the hit.
    if len(beat_frames) >= 4:
        at_beats = np.array(
            [float(env[max(0, f - 3) : f + 4].max()) for f in beat_frames]
        )
        phase_means = [float(at_beats[p::4].mean()) for p in range(4)]
        downbeat_offset = int(np.argmax(phase_means))
    else:
        downbeat_offset = 0

    return (
        Tempo(
            bpm_global=canonical.round_bpm(bpm),
            confidence=canonical.round_ratio(confidence),
        ),
        Beats(
            times=[canonical.round_seconds(t) for t in beat_times],
            beats_per_bar=4,
            downbeat_offset=downbeat_offset,
            downbeat_method="assumed_4_4_phase_fit",
        ),
    )


def _integrated_lufs(stereo) -> float:
    import numpy as np
    import pyloudnorm

    lufs = pyloudnorm.Meter(SAMPLE_RATE).integrated_loudness(stereo.astype(np.float64))
    if not np.isfinite(lufs):
        lufs = -70.0  # R128 absolute gate: silent program, not -inf
    return canonical.round_db(max(lufs, -70.0))


def _vocal_activity(vocal_rms_db, fcfg: config.FeaturesConfig) -> VocalActivity:
    import numpy as np

    # D3 (PR #6 review): config values pass through the precision contract
    # like every other document field, and the ROUNDED values drive the
    # computation below (round 3) — the recorded params are the exact
    # provenance of the regions, not an approximation of it.
    enter_db = canonical.round_db(fcfg.vocal_enter_db)
    exit_db = canonical.round_db(fcfg.vocal_exit_db)
    min_region_seconds = canonical.round_seconds(fcfg.vocal_min_region_seconds)
    min_gap_seconds = canonical.round_seconds(fcfg.vocal_min_gap_seconds)
    params = VocalActivityParams(
        enter_db=enter_db,
        exit_db=exit_db,
        min_region_seconds=min_region_seconds,
        min_gap_seconds=min_gap_seconds,
    )
    spans: list[list[int]] = []
    active = False
    start = 0
    for i, v in enumerate(vocal_rms_db):
        if not active and v >= enter_db:
            active, start = True, i
        elif active and v < exit_db:
            active = False
            spans.append([start, i])
    if active:
        spans.append([start, len(vocal_rms_db)])

    merged: list[list[int]] = []
    max_gap = int(round(min_gap_seconds / HOP_SECONDS))
    for span in spans:
        if merged and span[0] - merged[-1][1] < max_gap:
            merged[-1][1] = span[1]
        else:
            merged.append(span)

    min_len = int(round(min_region_seconds / HOP_SECONDS))
    regions = [
        VocalRegion(
            start_seconds=canonical.round_seconds(s * HOP_SECONDS),
            end_seconds=canonical.round_seconds(e * HOP_SECONDS),
            mean_rms_db=canonical.round_db(float(np.mean(vocal_rms_db[s:e]))),
        )
        for s, e in merged
        if e - s >= min_len
    ]
    return VocalActivity(params=params, regions=regions)


def run_features(track: str, library: Library, cfg: config.Config) -> FeaturesResult:
    track_id = library.resolve_track_id(track)
    manifest = library.read_manifest(track_id)
    if manifest is None or manifest.documents.source.status != "ok":
        raise PrerequisiteError(f"track {track_id} has no successful ingest")

    track_dir = library.track_dir(track_id)
    stems_dir = track_dir / "stems"
    stems_state = manifest.stems
    if stems_state is None or stems_state.status != "ok":
        raise PrerequisiteError(
            f"track {track_id}: stems are not separated — run `mrw stems {track_id}` first"
        )
    if not stems_state.retained or not all(
        (stems_dir / f"{n}.flac").is_file() for n in STEM_NAMES
    ):
        raise PrerequisiteError(
            f"track {track_id}: stem files are not on disk (retained: "
            f"{stems_state.retained}) — re-run `mrw stems {track_id}` with "
            "stems.retain = true"
        )

    features_hash = config.stage_hash(cfg.features)
    prior = manifest.documents.audio_features
    if (
        prior.status == "ok"
        and prior.config_hash == features_hash
        and (track_dir / "audio_features.json").is_file()
    ):
        return FeaturesResult(track_id, 0.0, -1, already_done=True)

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()

    def _record(entry: DocumentEntry) -> None:
        manifest.documents.audio_features = entry
        library.write_manifest(track_id, manifest)

    try:
        mix_mono, mix_stereo = _load(track_dir / "source_audio.flac")
        # The mix onset envelope is computed here (beat grid) and again in
        # _channel_features (mix onsets) — deliberate: sharing it would
        # couple the two code paths for a sub-second saving per track.
        tempo, beats = _beats_and_tempo(mix_mono)
        stem_channels: dict[str, ChannelFeatures] = {}
        vocal_rms = None
        for name in STEM_NAMES:
            y, _ = _load(stems_dir / f"{name}.flac")
            channel = _channel_features(y)
            stem_channels[name] = channel
            if name == "vocals":
                # Hysteresis runs over the ROUNDED series — the exact values
                # published in the document — so regions are derivable from
                # the document's own envelope + params (PR #6 round 3), and
                # the RMS isn't computed twice.
                import numpy as np

                vocal_rms = np.array(channel.rms_db.values)

        document = AudioFeaturesDocument(
            track_id=track_id,
            duration_seconds=canonical.round_seconds(len(mix_mono) / SAMPLE_RATE),
            tempo=tempo,
            beats=beats,
            loudness=Loudness(integrated_lufs=_integrated_lufs(mix_stereo)),
            mix=_channel_features(mix_mono),
            stems=StemFeatures(**stem_channels),
            vocal_activity=_vocal_activity(vocal_rms, cfg.features),
        )
        content_sha = library.write_document(track_id, "audio_features.json", document)
        _record(
            DocumentEntry(
                status="ok",
                path="audio_features.json",
                schema_version=document.schema_version,
                content_sha256=content_sha,
                config_hash=features_hash,
                run=RunMetadata(
                    started_at=started_at,
                    duration_seconds=round(time.monotonic() - t0, 2),
                    tool_version=TOOL_VERSION,
                ),
            )
        )
        return FeaturesResult(
            track_id=track_id,
            bpm_global=document.tempo.bpm_global,
            n_vocal_regions=len(document.vocal_activity.regions),
            already_done=False,
        )
    except PrerequisiteError:
        raise
    except Exception as exc:
        _record(
            DocumentEntry(
                status="failed",
                config_hash=features_hash,
                error=str(exc)[:500],
            )
        )
        raise FeaturesError(str(exc)[:500]) from exc
