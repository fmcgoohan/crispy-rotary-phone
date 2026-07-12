"""Review 005 field-finding fixes (F-1 MPS auto-retry, F-2 cpu_threads).

Both are tested fast by stubbing `_separate` — the fixes are control-flow
and config plumbing, not separation quality (covered by the slow M2 suite).
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

import mrw.stems
from mrw.cli import app
from mrw.stems import STEM_NAMES

SAMPLE_RATE = 44100

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)

runner = CliRunner()


def _make_wav(path: Path, seconds: float = 2.0) -> None:
    frames = bytearray()
    for i in range(int(SAMPLE_RATE * seconds)):
        v = int(32767 * 0.3 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE))
        frames += struct.pack("<hh", v, v)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(frames))


def _ingest(media: Path, library: Path) -> str:
    result = runner.invoke(app, ["ingest", str(media), "--library", str(library)])
    assert result.exit_code == 0, result.output
    return next(p.name for p in library.iterdir() if p.is_dir())


def _fake_stems(out_dir: Path) -> None:
    import numpy as np
    import soundfile as sf

    for name in STEM_NAMES:
        sf.write(
            out_dir / f"{name}.flac",
            np.zeros((SAMPLE_RATE, 2), dtype="int16"),
            SAMPLE_RATE,
            subtype="PCM_16",
        )


def test_mps_failure_retries_on_cpu(tmp_path: Path, monkeypatch) -> None:
    wav = tmp_path / "t.wav"
    _make_wav(wav)
    library = tmp_path / "lib"
    track_id = _ingest(wav, library)

    calls: list[str] = []

    def fake_separate(flac, out_dir, model, device, cpu_threads=1):
        calls.append(device)
        if device == "mps":
            raise RuntimeError("MPS op not supported (simulated)")
        _fake_stems(out_dir)

    monkeypatch.setattr(mrw.stems, "_separate", fake_separate)
    cfg = tmp_path / "mrw.toml"
    cfg.write_text('[stems]\ndevice = "mps"\n')
    result = runner.invoke(
        app, ["stems", track_id, "--library", str(library), "--config", str(cfg)]
    )
    assert result.exit_code == 0, result.output
    assert calls == ["mps", "cpu"]

    manifest = json.loads((library / track_id / "manifest.json").read_text())
    assert manifest["stems"]["status"] == "ok"
    assert manifest["stems"]["run"]["device"] == "cpu"  # what actually ran
    for name in STEM_NAMES:
        assert (library / track_id / "stems" / f"{name}.flac").is_file()


def test_cpu_failure_does_not_retry(tmp_path: Path, monkeypatch) -> None:
    wav = tmp_path / "t.wav"
    _make_wav(wav)
    library = tmp_path / "lib"
    track_id = _ingest(wav, library)

    calls: list[str] = []

    def fake_separate(flac, out_dir, model, device, cpu_threads=1):
        calls.append(device)
        raise RuntimeError("boom")

    monkeypatch.setattr(mrw.stems, "_separate", fake_separate)
    cfg = tmp_path / "mrw.toml"
    cfg.write_text('[stems]\ndevice = "cpu"\n')
    result = runner.invoke(
        app, ["stems", track_id, "--library", str(library), "--config", str(cfg)]
    )
    assert result.exit_code == 1
    assert calls == ["cpu"]  # exactly one attempt
    manifest = json.loads((library / track_id / "manifest.json").read_text())
    assert manifest["stems"]["status"] == "failed"


def test_cpu_threads_plumbed_and_hashes_config(tmp_path: Path, monkeypatch) -> None:
    wav = tmp_path / "t.wav"
    _make_wav(wav)
    library = tmp_path / "lib"
    track_id = _ingest(wav, library)

    seen: list[int] = []

    def fake_separate(flac, out_dir, model, device, cpu_threads=1):
        seen.append(cpu_threads)
        _fake_stems(out_dir)

    monkeypatch.setattr(mrw.stems, "_separate", fake_separate)
    cfg = tmp_path / "mrw.toml"
    cfg.write_text('[stems]\ndevice = "cpu"\ncpu_threads = 4\n')
    result = runner.invoke(
        app, ["stems", track_id, "--library", str(library), "--config", str(cfg)]
    )
    assert result.exit_code == 0, result.output
    assert seen == [4]

    # cpu_threads is part of the stems config subset: changing it must
    # change the stage hash, so a re-run is not a no-op.
    hash_4 = json.loads((library / track_id / "manifest.json").read_text())["stems"][
        "config_hash"
    ]
    cfg.write_text('[stems]\ndevice = "cpu"\ncpu_threads = 1\n')
    result = runner.invoke(
        app, ["stems", track_id, "--library", str(library), "--config", str(cfg)]
    )
    assert result.exit_code == 0, result.output
    assert "no-op" not in result.output
    hash_1 = json.loads((library / track_id / "manifest.json").read_text())["stems"][
        "config_hash"
    ]
    assert hash_1 != hash_4
