"""P1 group C: video timing on the audio clock.

ORCH-13 (ingest decodes an explicitly chosen, recorded audio stream),
VID-2 (video.json times on the audio timeline, VFR-safe frame indices),
VID-3 (motion pairs straddling a cut are excluded) and VID-11 (the last
shot never extends past the frames that exist).

Fixtures are generated in-test with ffmpeg (tones, solid colours, seeded
noise textures) — no media is checked in.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from mrw import ingest
from mrw.cli import app

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)

runner = CliRunner()


def _ffmpeg(*args: str) -> None:
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", *args], capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()[:600]


def _ingest(media: Path, library: Path) -> Path:
    result = runner.invoke(app, ["ingest", str(media), "--library", str(library)])
    assert result.exit_code == 0, result.output
    return next(p for p in library.iterdir() if p.is_dir())


def _dominant_hz(flac: Path) -> float:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(flac), "-ac", "1", "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(x))
    return float(np.argmax(spectrum) * 44100 / len(x))


# --- ORCH-13: audio stream selection --------------------------------------------


def test_select_audio_stream_rule() -> None:
    v = {"index": 0, "codec_type": "video"}
    a1 = {"index": 1, "codec_type": "audio", "disposition": {"default": 0}}
    a2 = {"index": 2, "codec_type": "audio", "disposition": {"default": 1}}
    a3 = {"index": 3, "codec_type": "audio"}
    assert ingest.select_audio_stream([v, a1, a2, a3]) == (a2, 3)  # default wins
    assert ingest.select_audio_stream([v, a1, a3]) == (a1, 2)  # else first audio
    assert ingest.select_audio_stream([v]) == (None, 0)


def _two_audio_streams(path: Path, dispositions: tuple[str, str]) -> None:
    """Stream 0: 440 Hz mono; stream 1: 880 Hz stereo (Matroska audio)."""
    _ffmpeg(
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=44100",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=2:sample_rate=44100",
        "-map", "0:a", "-map", "1:a", "-ac:a:1", "2",
        "-c:a", "flac",
        "-disposition:a:0", dispositions[0], "-disposition:a:1", dispositions[1],
        "-default_mode", "passthrough",
        str(path),
    )


@needs_ffmpeg
def test_ingest_decodes_the_default_audio_stream(tmp_path: Path) -> None:
    # The default disposition is on the SECOND stream: that one is decoded,
    # described and recorded.
    media = tmp_path / "two_audio.mka"
    _two_audio_streams(media, ("0", "default"))
    track = _ingest(media, tmp_path / "lib")
    source = json.loads((track / "source.json").read_text())
    assert source["schema_version"] == "1.1.0"
    audio = source["media"]["audio"]
    assert audio["stream_index"] == 1 and audio["stream_count"] == 2
    assert audio["channels"] == 2  # describes stream 1, not stream 0 (mono)
    assert abs(_dominant_hz(track / "source_audio.flac") - 880) < 5


@needs_ffmpeg
def test_ingest_without_default_flag_takes_first_audio_stream(tmp_path: Path) -> None:
    # No default flag anywhere: ffmpeg's own selection would prefer the
    # stereo stream 1 (more channels); the recorded rule takes stream 0.
    media = tmp_path / "no_default.mka"
    _two_audio_streams(media, ("0", "0"))
    track = _ingest(media, tmp_path / "lib")
    audio = json.loads((track / "source.json").read_text())["media"]["audio"]
    assert audio["stream_index"] == 0 and audio["stream_count"] == 2
    assert audio["channels"] == 1
    assert abs(_dominant_hz(track / "source_audio.flac") - 440) < 5
