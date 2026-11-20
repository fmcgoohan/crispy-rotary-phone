"""Video stage robustness: the stream the shot detector reads, and frames
that fail to decode partway through the stream.

Fixtures are generated in-test with ffmpeg (solid colours, a sine tone);
damage is introduced with ffmpeg's noise bitstream filter on one packet,
written with +bitexact so the fixture bytes are reproducible.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from mrw import video
from mrw.cli import app

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)

runner = CliRunner()


def _ffmpeg(*args: str) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *args], capture_output=True, check=False
    )
    assert proc.returncode == 0, proc.stderr.decode()[:600]


def _run_video(media: Path, library: Path) -> dict:
    result = runner.invoke(app, ["ingest", str(media), "--library", str(library)])
    assert result.exit_code == 0, result.output
    track = next(p for p in library.iterdir() if p.is_dir())
    result = runner.invoke(app, ["video", track.name, "--library", str(library)])
    assert result.exit_code == 0, result.output
    doc = json.loads((track / "video.json").read_text())
    doc["_track"] = track
    return doc


def _colour(track: Path, rep: dict) -> str:
    from PIL import Image

    with Image.open(track / rep["path"]) as im:
        r, g, b = np.asarray(im.convert("RGB")).reshape(-1, 3).mean(axis=0)
    for name, ref in {"red": (255, 0, 0), "blue": (0, 0, 255),
                      "green": (0, 128, 0), "yellow": (255, 255, 0)}.items():
        if max(abs(r - ref[0]), abs(g - ref[1]), abs(b - ref[2])) < 40:
            return name
    return "?"


# --- the detector reads the analysed stream ------------------------------------


def test_single_video_stream_is_passed_to_the_detector_untouched(monkeypatch) -> None:
    # No stream copy (no subprocess at all) when the file has one video stream.
    def boom(*a, **k):
        raise AssertionError("no subprocess expected")

    monkeypatch.setattr(video.subprocess, "run", boom)
    info = video.VideoStreamInfo(index=0, width=4, height=4, fps=24.0,
                                 start_time=0.0, n_video_streams=1)
    with video._detector_input(Path("x.mp4"), info) as p:
        assert p == Path("x.mp4")


def _two_video_streams(path: Path) -> None:
    """Stream 0: red 1 s → blue 2 s (cut at 1 s). Stream 1: green 2 s →
    yellow 1 s (cut at 2 s). Stream 2: audio. Stream 1 carries the default
    disposition — OpenCV still opens stream 0, the first video stream."""
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=red:s=320x240:r=24:d=1",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=24:d=2",
        "-f", "lavfi", "-i", "color=c=green:s=320x240:r=24:d=2",
        "-f", "lavfi", "-i", "color=c=yellow:s=320x240:r=24:d=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3:sample_rate=44100",
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0,setsar=1[x];"
        "[2:v][3:v]concat=n=2:v=1:a=0,setsar=1[y]",
        "-map", "[x]", "-map", "[y]", "-map", "4:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le",
        "-disposition:v:0", "0", "-disposition:v:1", "default", str(path),
    )


@needs_ffmpeg
@pytest.mark.slow
def test_shots_follow_the_analysed_stream_not_the_first(tmp_path: Path,
                                                        monkeypatch) -> None:
    # The stage analyses the first video stream that is not an attached
    # picture. ffmpeg's muxers write cover art after the real streams, so a
    # file whose analysed stream is not OpenCV's pick (the first video
    # stream) cannot be generated with ffmpeg; the selection is redirected
    # to stream 1 instead, standing in for any rule that picks a stream
    # other than the first. Frame table, extraction and motion follow
    # stream.index; the shots must too.
    media = tmp_path / "two.mkv"
    _two_video_streams(media)

    first = _run_video(media, tmp_path / "lib0")
    assert [s["start_seconds"] for s in first["shots"]] == [0.0, 1.0]

    real = video._video_stream_info

    def second_stream(path: Path) -> video.VideoStreamInfo:
        info = real(path)
        assert info.index == 0 and info.n_video_streams == 2
        return video.VideoStreamInfo(index=1, width=info.width, height=info.height,
                                     fps=info.fps, start_time=info.start_time,
                                     n_video_streams=info.n_video_streams)

    monkeypatch.setattr(video, "_video_stream_info", second_stream)
    doc = _run_video(media, tmp_path / "lib1")
    shots = doc["shots"]
    assert [(s["start_frame"], s["end_frame"]) for s in shots] == [(0, 48), (48, 72)]
    assert [s["start_seconds"] for s in shots] == [0.0, 2.0]
    assert {_colour(doc["_track"], r) for r in shots[0]["representative_frames"]} == {"green"}
    assert {_colour(doc["_track"], r) for r in shots[1]["representative_frames"]} == {"yellow"}
