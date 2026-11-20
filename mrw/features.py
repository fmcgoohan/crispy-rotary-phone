"""Stage 3 — audio features per PLAN §4 stage 3; contract:
schemas/audio_features.schema.json v1.1.0.

Reviewed conventions implemented here:
- 10 ms hop (100 Hz grid), analysis window 2048 (PLAN §4.3).
- Tempo/beat grid computed once on the mix (librosa beat_track, trim=True:
  no beats through beatless intros/outros); tempo.confidence is the beats'
  onset support against off-beat probes at k/7 of a beat (FEAT-3,
  _beat_confidence).
  A mix with no beat evidence (mix rms at the floor throughout, no beats
  tracked) fails the stage rather than writing bpm_global 0.0
  (DET-11/SCH-9, H3; review D-2).
- Downbeats via
  `assumed_4_4_phase_fit` (OQ-4) — the beat phase (0..3) maximizing mean
  onset strength, ties to the lowest phase.
- rms_db: dBFS, silence floor clamped at -80.0, rounded BEFORE any silence
  or gate decision reads it (FEAT-13).
- spectral_centroid_hz: silent frames hold the last valid centroid, frames
  before the first valid one are backfilled from it (review 001 R-2); a
  channel silent throughout writes all zeros (review 002 R-9).
- onsets: strengths normalized by the p98 of the channel's onset-strength
  envelope, clamped to [0, 1] (R-1); level-gated (FEAT-1): onsets in frames
  at the -80 floor are dropped, and a stem's onsets are dropped where the
  stem sits more than |onset_gate_relative_db| below the mix (the gate is
  embedded as onsets.gate, S4); `strength_reference: 1.0` sentinel when no
  onsets remain (R-9) — always so for a channel at the floor throughout.
- loudness: EBU R128 integrated LUFS; a fully-silent program returns -inf
  from the meter and is clamped to -70.0 (the R128 absolute gate) so the
  document stays valid JSON; input shorter than one 400 ms block is
  measured as a single block spanning the signal (FEAT-8).
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
from typing import Optional

from . import TOOL_VERSION, canonical, config, versions
from .library import Library, PrerequisiteError, mark_dependents_stale
from .models import (
    AudioFeaturesDocument,
    Beats,
    ChannelFeatures,
    DocumentEntry,
    Loudness,
    OnsetGate,
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

# Structural constants (review 006 F-1 audit): these are the schema contract
# itself (hop, window, floor are stated in the schema/PLAN), riding
# tool_version rather than config; the p98 onset reference is likewise
# schema-stated (R-1). Output-affecting *tuning* knobs live in FeaturesConfig
# so they participate in the stage config_hash.
SAMPLE_RATE = 44100
HOP = 441  # 10 ms at 44.1 kHz — the 100 Hz uniform grid (PLAN §4.3)
HOP_SECONDS = 0.01
N_FFT = 2048
SILENCE_FLOOR_DB = -80.0
# tempo.confidence probe half-window (FEAT-3): estimator-structural and
# stated in the schema's confidence formula, so it rides the algorithm
# version rather than config (confidence is advisory — the same review 006
# F-1 audit call as the old CV multiplier).
BEAT_SALIENCE_WINDOW_SECONDS = 0.03


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
    """The published rms_db series: dBFS, floor-clamped, ROUNDED to 2 dp.

    Every silence/gate decision downstream (centroid hold, onset gating,
    vocal activity) reads this rounded series, so each one is reproducible
    from the document's own values (FEAT-13; PR #6 round 3 precedent)."""
    import librosa
    import numpy as np

    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP, center=True)[0]
    db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    db = np.maximum(db, SILENCE_FLOOR_DB)
    return np.array([canonical.round_db(v) for v in db], dtype=np.float64)


def _at_floor(rms_db):
    return rms_db <= SILENCE_FLOOR_DB


def _centroid_hz(y, rms_db):
    """Centroid with hold-last + backfill over silent frames (R-2/R-9)."""
    import librosa
    import numpy as np

    cent = librosa.feature.spectral_centroid(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP, center=True
    )[0]
    silent = _at_floor(rms_db)
    if silent.all():
        return np.zeros_like(cent)  # R-9: silent throughout → all zeros
    n = len(cent)
    last_valid = np.where(~silent, np.arange(n), -1)
    np.maximum.accumulate(last_valid, out=last_valid)
    first_valid = int(np.flatnonzero(~silent)[0])
    last_valid[last_valid < 0] = first_valid  # backfill the leading silence
    return cent[last_valid]


def _onsets(y, rms_db, mix_rms_db=None, relative_db: Optional[float] = None) -> Onsets:
    """Onsets gated by level (FEAT-1).

    librosa's onset envelope and peak picking are gain-invariant, so a stem
    holding only dither or separation bleed yields as many full-strength
    onsets as a real part. Gate on the published (rounded) rms_db series:
    an onset is dropped when its frame sits at the -80 dB floor, and — for a
    stem (relative_db given) — when the stem's rms_db at that frame is more
    than |relative_db| below the mix's rms_db at the same frame. A channel at
    the floor throughout takes the R-9 sentinel directly, so rms, centroid
    and onsets agree. The strength reference stays the p98 of the channel's
    whole (ungated) envelope: the gate selects onsets, it does not rescale
    them.
    """
    import librosa
    import numpy as np

    gate = OnsetGate(floor_db=SILENCE_FLOOR_DB, relative_db=relative_db)
    sentinel = Onsets(times=[], strengths=[], strength_reference=1.0, gate=gate)  # R-9
    floor = _at_floor(rms_db)
    if floor.all():
        return sentinel

    env = librosa.onset.onset_strength(y=y, sr=SAMPLE_RATE, hop_length=HOP)
    frames = librosa.onset.onset_detect(
        onset_envelope=env, sr=SAMPLE_RATE, hop_length=HOP, backtrack=False
    )
    # The envelope and the rms series share the centred 10 ms frame grid;
    # clamp in case their lengths differ by a frame at the tail.
    idx = np.minimum(frames, len(rms_db) - 1)
    keep = ~floor[idx]
    if relative_db is not None:
        midx = np.minimum(frames, len(mix_rms_db) - 1)
        keep &= (rms_db[idx] - mix_rms_db[midx]) >= relative_db
    frames = frames[keep]
    if len(frames) == 0:
        return sentinel
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
        gate=gate,
    )


def _timeseries(values, unit: str, rounder) -> Timeseries:
    return Timeseries(
        unit=unit,
        start_seconds=0.0,
        hop_seconds=HOP_SECONDS,
        values=[rounder(v) for v in values],
    )


def _channel_features(y, mix_rms_db=None, relative_db: Optional[float] = None):
    """(features, rounded rms series). The mix passes no mix_rms_db and gets
    the floor gate only; a stem passes the mix's series and the relative gate."""
    rms = _rms_db(y)
    cent = _centroid_hz(y, rms)
    features = ChannelFeatures(
        rms_db=_timeseries(rms, "dbfs", canonical.round_db),
        spectral_centroid_hz=_timeseries(cent, "hz", canonical.round_hz),
        onsets=_onsets(y, rms, mix_rms_db, relative_db),
    )
    return features, rms


def _pool(env, frame: int, w: int) -> float:
    frame = min(max(int(frame), 0), len(env) - 1)
    return float(env[max(0, frame - w) : frame + w + 1].max())


# Off-beat probe positions, as fractions of the local beat period (FEAT-3,
# review D-1): the six 1/7-beat points. None of them coincides with a
# common subdivision — 8ths (1/2), triplets (1/3, 2/3), 16ths (1/4, 3/4),
# quintuplets (k/5), sextuplets (k/6) — so at least one probe falls between
# subdivision hits and off_i reads the gap, not the hat. Septuplet
# subdivisions are the (rare) blind spot. Structural, like the window.
BEAT_OFF_PROBES = tuple(k / 7 for k in range(1, 7))


def _beat_confidence(env, beat_frames) -> float:
    """Onset support for the beat grid (FEAT-3) — see the schema description.

    For beat i at frame b_i with local period p_i (the following inter-beat
    interval; the last beat reuses the previous one): on_i is the envelope
    max within ±BEAT_SALIENCE_WINDOW of b_i; off_i is the SMALLEST of the
    same pooled value at the BEAT_OFF_PROBES points b_i + k·p_i/7 (k = 1..6),
    skipping a probe whose window would contain either beat frame (it would
    read the beat itself; only possible above ~350 BPM). Taking the
    minimum over probes that avoid every common subdivision means 8ths,
    swung 8ths, triplets, 16ths, quintuplets and sextuplets under a correct
    grid all keep off_i at the gap level (review D-1: the earlier 1/3 + 2/3
    probes sat exactly on triplets). confidence = mean_i clip((on_i - off_i)
    / ref, 0, 1) with ref = p90 of on_i, so beats placed over nothing count
    0 and beats stronger than their off-beat gaps count up to 1. Beatless
    audio (drone, noise) has on ≈ off → ~0.1-0.2; a click track → ~0.9.
    Half/double-tempo errors are not detected: a grid at half the true tempo
    still sits on onsets.
    """
    import numpy as np

    if len(beat_frames) < 2:
        return 0.0
    w = max(1, int(round(BEAT_SALIENCE_WINDOW_SECONDS / HOP_SECONDS)))
    frames = np.asarray(beat_frames, dtype=np.int64)
    periods = np.diff(frames)
    periods = np.append(periods, periods[-1])
    on = np.array([_pool(env, f, w) for f in frames])
    off = []
    for f, p in zip(frames, periods):
        offsets = [round(x * p) for x in BEAT_OFF_PROBES]
        clear = [d for d in offsets if w < d < p - w]
        # Beat periods under 2w+2 frames (>~750 BPM) leave no clear probe;
        # librosa's tempo range never gets there, but stay total: midpoint.
        off.append(min(_pool(env, f + d, w) for d in (clear or [p // 2])))
    off = np.array(off)
    ref = float(np.percentile(on, 90))
    if ref <= 0.0:
        return 0.0
    return float(np.clip((on - off) / ref, 0.0, 1.0).mean())


def _beats_and_tempo(y_mix, phase_fit_window_seconds: float) -> tuple[Tempo, Beats]:
    import librosa
    import numpy as np

    env = librosa.onset.onset_strength(y=y_mix, sr=SAMPLE_RATE, hop_length=HOP)
    if not env.any():
        raise FeaturesError(
            "no beat evidence in mix (no onset energy: silent or constant signal); "
            "no beat grid can be written"
        )
    # trim=True (librosa's default): beats over weak leading/trailing onset
    # strength are dropped, so beatless intros and outros carry no invented
    # grid (FEAT-3). Interior breakdowns are not trimmed — confidence (and a
    # consumer reading onsets) is what reveals those.
    bpm, beat_frames = librosa.beat.beat_track(
        onset_envelope=env, sr=SAMPLE_RATE, hop_length=HOP, trim=True
    )
    bpm = float(np.atleast_1d(bpm)[0])
    if len(beat_frames) == 0 or canonical.round_bpm(bpm) <= 0.0:
        # Contract-preserving H3 choice (DET-11/SCH-9): bpm_global keeps its
        # exclusiveMinimum 0 promise; the stage fails instead of writing 0.0.
        raise FeaturesError(
            "no beat evidence in mix (the beat tracker found no beats); "
            "no beat grid can be written"
        )
    beat_times = librosa.frames_to_time(beat_frames, sr=SAMPLE_RATE, hop_length=HOP)
    confidence = _beat_confidence(env, beat_frames)

    # OQ-4: assumed-4/4 phase fit — pick the beat phase whose every-4th
    # beats have the highest mean onset strength; ties take the lowest phase
    # (np.argmax's first-max rule) for determinism. Strength is the local
    # max in a ±phase_fit_window_seconds window around each beat (config,
    # review 006 F-1): tracked beats sit a frame or two off the onset-energy
    # peak, and sampling the envelope exactly at the beat frame reads
    # pre-onset noise instead of the hit.
    if len(beat_frames) >= 4:
        w = max(1, int(round(phase_fit_window_seconds / HOP_SECONDS)))
        at_beats = np.array(
            [float(env[max(0, f - w) : f + w + 1].max()) for f in beat_frames]
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
    import math

    import numpy as np
    import pyloudnorm

    n = stereo.shape[0]
    block = 0.400  # BS.1770 gating block
    if n < block * SAMPLE_RATE:
        # Shorter than one 400 ms block: pyloudnorm refuses the input
        # (FEAT-8). Measure it as a single block spanning the whole signal
        # (same K-weighting and -70 absolute gate, no relative gating — one
        # block has nothing to gate against). Nudged down an ulp so the
        # library's length check (n >= block * rate) holds exactly.
        block = n / SAMPLE_RATE
        while block * SAMPLE_RATE > n:
            block = math.nextafter(block, 0.0)
    meter = pyloudnorm.Meter(SAMPLE_RATE, block_size=block)
    lufs = meter.integrated_loudness(stereo.astype(np.float64))
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
            f"track {track_id}: stems are not separated (status "
            f"{stems_state.status if stems_state else 'missing'}) — run "
            f"`mrw stems {track_id}` first"
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
        and versions.is_current("audio_features", prior.algorithm_version)
        and (track_dir / "audio_features.json").is_file()
    ):
        return FeaturesResult(track_id, 0.0, -1, already_done=True)

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()

    def _record(entry: DocumentEntry) -> None:
        def apply(m) -> None:
            m.documents.audio_features = entry
            if entry.status == "ok" and entry.content_sha256 != prior.content_sha256:
                mark_dependents_stale(m, "audio_features")

        apply(manifest)
        library.update_manifest(track_id, apply)

    try:
        mix_mono, mix_stereo = _load(track_dir / "source_audio.flac")
        # The mix goes first: its rounded rms series is the reference for
        # the stems' relative onset gate (FEAT-1) and for the silence check
        # below. D3: the configured gate passes through the precision
        # contract and the ROUNDED value both drives the gate and is
        # embedded in each stem's onsets.gate.
        mix_channel, mix_rms = _channel_features(mix_mono)
        if _at_floor(mix_rms).all():
            # Review D-2: the document's own silence rule (published rms at
            # the floor throughout → R-9 sentinel onsets, zero centroid) must
            # also deny a beat grid. Dither of ±1 LSB has a non-zero onset
            # envelope and beat_track would invent ~125 BPM over it.
            raise FeaturesError(
                "no beat evidence in mix (mix rms_db at the -80 dB floor "
                "throughout: silent); no beat grid can be written"
            )
        # The mix onset envelope is computed here (beat grid) and again in
        # _channel_features (mix onsets) — deliberate: sharing it would
        # couple the two code paths for a sub-second saving per track.
        tempo, beats = _beats_and_tempo(mix_mono, cfg.features.phase_fit_window_seconds)
        relative_db = canonical.round_db(cfg.features.onset_gate_relative_db)
        stem_channels: dict[str, ChannelFeatures] = {}
        vocal_rms = None
        for name in STEM_NAMES:
            y, _ = _load(stems_dir / f"{name}.flac")
            channel, rms = _channel_features(y, mix_rms, relative_db)
            stem_channels[name] = channel
            if name == "vocals":
                # Hysteresis runs over the ROUNDED series — the exact values
                # published in the document — so regions are derivable from
                # the document's own envelope + params (PR #6 round 3).
                vocal_rms = rms

        document = AudioFeaturesDocument(
            track_id=track_id,
            duration_seconds=canonical.round_seconds(len(mix_mono) / SAMPLE_RATE),
            tempo=tempo,
            beats=beats,
            loudness=Loudness(integrated_lufs=_integrated_lufs(mix_stereo)),
            mix=mix_channel,
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
                algorithm_version=versions.current("audio_features"),
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
