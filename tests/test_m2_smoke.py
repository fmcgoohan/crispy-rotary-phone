"""M2 smoke test (PLAN §11, extended per the M2 milestone review):

1. 30 s generated fixture (tone + click track mixed) → 4 stem files exist,
   each duration matches the source ±1 frame.
2. CPU double-run: byte-identical stem files across two runs (T2).
3. retain: false → no stems/ afterward; manifest records retained: false so
   a dependent-stage re-run knows to regenerate.
4. Failure path: missing source audio → failed status recorded, library
   intact, exit code 1 (T4).

Demucs-invoking tests are marked `slow` (minutes on CPU: determinism pins
torch to one thread). The 30 s separations are shared via a module fixture
so the suite pays for exactly two of them plus one short retain:false run.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mrw.cli import app
from mrw.stems import STEM_NAMES

REPO = Path(__file__).resolve().parents[1]
SCHEMAS = REPO / "schemas"
SAMPLE_RATE = 44100

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)

runner = CliRunner()


def _write_mix(path: Path, seconds: float) -> None:
    """Tone + click track: 220 Hz sine bed with a decaying click every 0.5 s."""
    n = int(SAMPLE_RATE * seconds)
    frames = bytearray()
    click_period = SAMPLE_RATE // 2
    click_len = SAMPLE_RATE // 200  # 5 ms
    for i in range(n):
        v = 0.25 * math.sin(2 * math.pi * 220 * i / SAMPLE_RATE)
        pos = i % click_period
        if pos < click_len:
            v += 0.65 * (1.0 - pos / click_len) * math.sin(2 * math.pi * 1500 * i / SAMPLE_RATE)
        s = int(32767 * max(-1.0, min(1.0, v)))
        frames += struct.pack("<hh", s, s)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(frames))


def _cpu_config(tmp: Path, retain: bool = True) -> Path:
    cfg = tmp / "mrw.toml"
    cfg.write_text(f'[stems]\ndevice = "cpu"\nretain = {str(retain).lower()}\n')
    return cfg


def _ingest(media: Path, library: Path) -> str:
    result = runner.invoke(app, ["ingest", str(media), "--library", str(library)])
    assert result.exit_code == 0, result.output
    return next(p.name for p in library.iterdir() if p.is_dir())


def _run_stems(track: str, library: Path, cfg: Path) -> None:
    result = runner.invoke(
        app, ["stems", track, "--library", str(library), "--config", str(cfg)]
    )
    assert result.exit_code == 0, result.output


def _sample_count(flac: Path) -> int:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(flac), "-f", "s16le", "-"],
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    return len(proc.stdout) // (2 * 2)


@pytest.fixture(scope="module")
def double_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, str]:
    """Ingest + separate the 30 s mix twice (two library roots, CPU)."""
    base = tmp_path_factory.mktemp("m2")
    mix = base / "mix30.wav"
    _write_mix(mix, seconds=30.0)
    cfg = _cpu_config(base)
    libs = []
    track_id = ""
    for name in ("lib_a", "lib_b"):
        library = base / name
        track_id = _ingest(mix, library)
        _run_stems(track_id, library, cfg)
        libs.append(library)
    return libs[0], libs[1], track_id


@pytest.mark.slow
def test_stems_exist_with_matching_duration(double_run) -> None:
    lib_a, _, track_id = double_run
    track_dir = lib_a / track_id
    source_samples = _sample_count(track_dir / "source_audio.flac")
    for name in STEM_NAMES:
        stem = track_dir / "stems" / f"{name}.flac"
        assert stem.is_file(), f"missing stem {name}"
        assert abs(_sample_count(stem) - source_samples) <= 1  # ±1 frame

    manifest = json.loads((track_dir / "manifest.json").read_text())
    stems = manifest["stems"]
    assert stems["status"] == "ok"
    assert stems["retained"] is True
    assert stems["run"]["device"] == "cpu"

    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    registry = Registry()
    for f in SCHEMAS.glob("*.schema.json"):
        registry = registry.with_resource(
            f.name, Resource.from_contents(json.loads(f.read_text()))
        )
    schema = json.loads((SCHEMAS / "manifest.schema.json").read_text())
    errors = list(Draft202012Validator(schema, registry=registry).iter_errors(manifest))
    assert not errors, "\n".join(e.message for e in errors)


@pytest.mark.slow
def test_cpu_double_run_stem_byte_identity(double_run) -> None:
    lib_a, lib_b, track_id = double_run
    for name in STEM_NAMES:
        a = (lib_a / track_id / "stems" / f"{name}.flac").read_bytes()
        b = (lib_b / track_id / "stems" / f"{name}.flac").read_bytes()
        assert a == b, f"stem {name} differs across CPU runs"


@pytest.mark.slow
def test_retain_false_leaves_no_stems(tmp_path: Path) -> None:
    mix = tmp_path / "mix6.wav"
    _write_mix(mix, seconds=6.0)
    library = tmp_path / "lib"
    track_id = _ingest(mix, library)
    _run_stems(track_id, library, _cpu_config(tmp_path, retain=False))

    track_dir = library / track_id
    assert not (track_dir / "stems").exists()
    manifest = json.loads((track_dir / "manifest.json").read_text())
    # A dependent-stage re-run reads exactly this to know it must regenerate.
    assert manifest["stems"]["status"] == "ok"
    assert manifest["stems"]["retained"] is False


@pytest.mark.slow
def test_retain_flip_removes_previously_retained_stems(tmp_path: Path) -> None:
    # H2 (M2 review round 2): retain true→false must not leave stale stems/
    # on disk contradicting `retained: false` in the manifest.
    mix = tmp_path / "mix6.wav"
    _write_mix(mix, seconds=6.0)
    library = tmp_path / "lib"
    track_id = _ingest(mix, library)
    track_dir = library / track_id

    _run_stems(track_id, library, _cpu_config(tmp_path, retain=True))
    assert (track_dir / "stems").is_dir()

    _run_stems(track_id, library, _cpu_config(tmp_path, retain=False))
    assert not (track_dir / "stems").exists()
    manifest = json.loads((track_dir / "manifest.json").read_text())
    assert manifest["stems"]["status"] == "ok"
    assert manifest["stems"]["retained"] is False


@pytest.mark.slow
def test_silent_input_produces_finite_near_silent_stems(tmp_path: Path) -> None:
    # H2 (M2 review): silent input must not NaN — it separates unnormalized.
    import numpy as np
    import soundfile as sf

    silent = tmp_path / "silence3.wav"
    with wave.open(str(silent), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"\x00" * (SAMPLE_RATE * 3 * 4))

    library = tmp_path / "lib"
    track_id = _ingest(silent, library)
    _run_stems(track_id, library, _cpu_config(tmp_path))

    for name in STEM_NAMES:
        data, _ = sf.read(library / track_id / "stems" / f"{name}.flac")
        assert np.isfinite(data).all()
        assert np.abs(data).max() < 0.1, f"stem {name} is not near-silent"


def test_missing_source_audio_fails_cleanly(tmp_path: Path) -> None:
    mix = tmp_path / "mix2.wav"
    _write_mix(mix, seconds=2.0)
    library = tmp_path / "lib"
    track_id = _ingest(mix, library)
    track_dir = library / track_id
    (track_dir / "source_audio.flac").unlink()

    result = runner.invoke(
        app,
        ["stems", track_id, "--library", str(library),
         "--config", str(_cpu_config(tmp_path))],
    )
    assert result.exit_code == 1
    manifest = json.loads((track_dir / "manifest.json").read_text())
    assert manifest["stems"]["status"] == "failed"
    assert "source_audio.flac" in manifest["stems"]["error"]
    # Library intact: prior documents untouched, no partial stems/.
    assert (track_dir / "source.json").is_file()
    assert not (track_dir / "stems").exists()
    assert not (track_dir / ".stems.tmp").exists()


def test_unknown_track_is_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["stems", "deadbeef00000000", "--library", str(tmp_path / "lib")]
    )
    assert result.exit_code == 2
