"""M3 smoke test (PLAN §11, extended per the M3 milestone task):

1. Ground truth: generated click track at 120 BPM, first click at 0.25 s,
   accent every 4th click → BPM within ±1, beats within 30 ms of true click
   times, downbeat phase lands on the accented clicks (tolerances stated
   inline).
2. Conventions: silent-throughout stem → all-zeros centroid, empty onsets,
   strength_reference 1.0 (R-9); leading silence → backfilled centroid with
   no sentinel sweep (R-2).
3. Determinism: double-run byte identity of audio_features.json.
4. Failure path: features without stems → prerequisite error (exit 2),
   library intact.

Fast tests fabricate per-stem FLACs directly (no Demucs); the end-to-end
chain onto a real separation is marked `slow`.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from mrw.cli import app
from mrw.stems import STEM_NAMES

REPO = Path(__file__).resolve().parents[1]
SCHEMAS = REPO / "schemas"
SAMPLE_RATE = 44100
BPM = 120.0
FIRST_CLICK = 0.25
CLICK_PERIOD = 60.0 / BPM  # 0.5 s

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobeg not available",
)

runner = CliRunner()


def _click_track(seconds: float) -> np.ndarray:
    """120 BPM clicks starting at 0.25 s; every 4th click accented (downbeat)."""
    n = int(SAMPLE_RATE * seconds)
    y = np.zeros(n, dtype=np.float32)
    click_len = SAMPLE_RATE // 100  # 10 ms
    t = np.arange(click_len) / SAMPLE_RATE
    burst = np.sin(2 * np.pi * 1500 * t).astype(np.float32) * np.linspace(
        1.0, 0.0, click_len, dtype=np.float32
    )
    k = 0
    while True:
        start = int(round((FIRST_CLICK + k * CLICK_PERIOD) * SAMPLE_RATE))
        if start + click_len > n:
            break
        amp = 0.9 if k % 4 == 0 else 0.45
        y[start : start + click_len] += amp * burst
        k += 1
    return y


def _tone(seconds: float, hz: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _write_stereo_wav(path: Path, mono: np.ndarray) -> None:
    pcm = (np.clip(mono, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(np.repeat(pcm[:, None], 2, axis=1).tobytes())


def _fabricate_track(
    library: Path, mix: np.ndarray, stems: dict[str, np.ndarray], tmp: Path
) -> str:
    """Ingest a mix, then plant synthetic stems + an ok stems manifest entry —
    the fast-suite substitute for a real Demucs separation."""
    wav = tmp / f"mix_{abs(hash(mix.tobytes())) % 10**8}.wav"
    _write_stereo_wav(wav, mix)
    result = runner.invoke(app, ["ingest", str(wav), "--library", str(library)])
    assert result.exit_code == 0, result.output
    track_id = next(p.name for p in library.iterdir() if p.is_dir())

    stems_dir = library / track_id / "stems"
    stems_dir.mkdir()
    for name in STEM_NAMES:
        y = stems.get(name, np.zeros_like(mix))
        pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
        sf.write(
            stems_dir / f"{name}.flac",
            np.repeat(pcm[:, None], 2, axis=1),
            SAMPLE_RATE,
            subtype="PCM_16",
        )
    manifest_path = library / track_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["stems"] = {"status": "ok", "retained": True, "config_hash": "0" * 16}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return track_id


def _run_features(track_id: str, library: Path) -> dict:
    result = runner.invoke(app, ["features", track_id, "--library", str(library)])
    assert result.exit_code == 0, result.output
    return json.loads((library / track_id / "audio_features.json").read_text())


def _validate(document: dict) -> None:
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    registry = Registry()
    for f in SCHEMAS.glob("*.schema.json"):
        registry = registry.with_resource(
            f.name, Resource.from_contents(json.loads(f.read_text()))
        )
    schema = json.loads((SCHEMAS / "audio_features.schema.json").read_text())
    errors = list(Draft202012Validator(schema, registry=registry).iter_errors(document))
    assert not errors, "\n".join(e.message for e in errors)


@pytest.fixture(scope="module")
def click_doc(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("m3_click")
    mix = _click_track(20.0)
    library = tmp / "lib"
    track_id = _fabricate_track(
        library, mix, {"drums": mix}, tmp  # vocals/bass/other silent (R-9 case)
    )
    return _run_features(track_id, library)


def test_click_track_beat_grid(click_doc: dict) -> None:
    _validate(click_doc)
    # BPM tolerance ±1 (PLAN M3 smoke: "detected BPM in [119, 121]").
    assert 119.0 <= click_doc["tempo"]["bpm_global"] <= 121.0
    assert click_doc["tempo"]["confidence"] >= 0.5

    # Beat-time tolerance: 30 ms (PLAN M3 smoke) against the true click grid.
    times = np.array(click_doc["beats"]["times"])
    assert len(times) >= 30
    offsets = (times - FIRST_CLICK) % CLICK_PERIOD
    dist = np.minimum(offsets, CLICK_PERIOD - offsets)
    assert float(np.median(dist)) <= 0.030
    assert (dist <= 0.030).mean() >= 0.9  # ≥90% of beats on the click grid

    # Downbeat phase: the accented clicks are 2.0 s apart starting at 0.25 s.
    beats = click_doc["beats"]
    assert beats["downbeat_method"] == "assumed_4_4_phase_fit"
    assert beats["beats_per_bar"] == 4
    downbeat_time = times[beats["downbeat_offset"]]
    phase = (downbeat_time - FIRST_CLICK) % (4 * CLICK_PERIOD)
    assert min(phase, 4 * CLICK_PERIOD - phase) <= 0.030


def test_silent_stem_conventions(click_doc: dict) -> None:
    # bass was silent throughout (R-9): all-zeros centroid, empty onsets,
    # sentinel strength_reference, rms pinned at the floor.
    bass = click_doc["stems"]["bass"]
    assert all(v == 0.0 for v in bass["spectral_centroid_hz"]["values"])
    assert bass["onsets"]["times"] == []
    assert bass["onsets"]["strengths"] == []
    assert bass["onsets"]["strength_reference"] == 1.0
    assert all(v == -80.0 for v in bass["rms_db"]["values"])

    # drums carried the clicks: onsets present, strengths in [0, 1], p98 ref.
    drums = click_doc["stems"]["drums"]
    assert len(drums["onsets"]["times"]) >= 30
    assert drums["onsets"]["strength_reference"] > 0
    assert all(0.0 <= s <= 1.0 for s in drums["onsets"]["strengths"])


def test_leading_silence_centroid_backfill(tmp_path: Path) -> None:
    # vocals: 2 s silence then 3 s of 880 Hz tone (R-2: backfill, no sweep).
    vocals = np.concatenate([np.zeros(2 * SAMPLE_RATE, np.float32), _tone(3.0, 880.0)])
    mix = _tone(5.0, 220.0) + vocals
    library = tmp_path / "lib"
    track_id = _fabricate_track(library, mix, {"vocals": vocals}, tmp_path)
    doc = _run_features(track_id, library)
    _validate(doc)

    cent = doc["stems"]["vocals"]["spectral_centroid_hz"]["values"]
    rms = doc["stems"]["vocals"]["rms_db"]["values"]
    head = cent[:150]  # first 1.5 s — all inside the leading silence
    assert all(v > 0 for v in head), "leading silence must be backfilled, not zero"
    assert max(head) - min(head) < 1.0, "backfill must be flat (no sentinel sweep)"
    # R-2's exact convention: the backfill value IS the first valid frame's
    # centroid (note: that frame straddles the silence→tone boundary, so its
    # value is onset-transient-skewed — the convention backfills it anyway).
    first_valid = next(i for i, v in enumerate(rms) if v > -80.0)
    assert head[0] == cent[first_valid]

    # Vocal activity: one region covering roughly the tone span (OQ-5 params
    # embedded per R-5).
    va = doc["vocal_activity"]
    assert va["params"]["enter_db"] == -35.0
    assert len(va["regions"]) == 1
    region = va["regions"][0]
    assert abs(region["start_seconds"] - 2.0) < 0.15
    assert abs(region["end_seconds"] - 5.0) < 0.15


def test_features_double_run_byte_identity(tmp_path: Path) -> None:
    mix = _click_track(8.0)
    docs = []
    for name in ("lib_a", "lib_b"):
        library = tmp_path / name
        track_id = _fabricate_track(library, mix, {"drums": mix}, tmp_path)
        _run_features(track_id, library)
        docs.append((library / track_id / "audio_features.json").read_bytes())
    assert docs[0] == docs[1]


def test_features_without_stems_is_prerequisite_error(tmp_path: Path) -> None:
    wav = tmp_path / "plain.wav"
    _write_stereo_wav(wav, _tone(2.0, 330.0))
    library = tmp_path / "lib"
    result = runner.invoke(app, ["ingest", str(wav), "--library", str(library)])
    assert result.exit_code == 0
    track_id = next(p.name for p in library.iterdir() if p.is_dir())

    result = runner.invoke(app, ["features", track_id, "--library", str(library)])
    assert result.exit_code == 2
    assert "mrw stems" in result.output  # tells the user what to run
    manifest = json.loads((library / track_id / "manifest.json").read_text())
    assert manifest["documents"]["audio_features"]["status"] == "pending"
    assert not (library / track_id / "audio_features.json").exists()


@pytest.mark.slow
def test_features_on_real_separation(tmp_path: Path) -> None:
    """End-to-end chain: ingest → Demucs stems → features, schema-valid."""
    mix = _click_track(30.0) * 0.6 + np.concatenate(
        [_tone(30.0, 220.0, 0.2)]
    )
    wav = tmp_path / "mix30.wav"
    _write_stereo_wav(wav, mix)
    library = tmp_path / "lib"
    cfg = tmp_path / "mrw.toml"
    cfg.write_text('[stems]\ndevice = "cpu"\n')

    result = runner.invoke(app, ["ingest", str(wav), "--library", str(library)])
    assert result.exit_code == 0, result.output
    track_id = next(p.name for p in library.iterdir() if p.is_dir())
    result = runner.invoke(
        app, ["stems", track_id, "--library", str(library), "--config", str(cfg)]
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app, ["features", track_id, "--library", str(library), "--config", str(cfg)]
    )
    assert result.exit_code == 0, result.output

    doc = json.loads((library / track_id / "audio_features.json").read_text())
    _validate(doc)
    assert 119.0 <= doc["tempo"]["bpm_global"] <= 121.0
    manifest = json.loads((library / track_id / "manifest.json").read_text())
    assert manifest["documents"]["audio_features"]["status"] == "ok"
