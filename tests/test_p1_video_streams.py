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


# --- frames that fail to decode mid-stream --------------------------------------


def _damaged_clip(tmp_path: Path) -> Path:
    """Red 2 s → blue 2 s at 24 fps (no B-frames, one keyframe per 48
    frames), PCM audio; then packet 24 (pts 1.0 s, a P-frame) has its bytes
    overwritten by the noise bitstream filter so the decoder rejects it
    ("Invalid NAL unit size"): 96 packets, 95 decoded frames. +bitexact
    keeps the remuxed fixture byte-reproducible."""
    src = tmp_path / "src.mkv"
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=red:s=320x240:r=24:d=2",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=24:d=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4:sample_rate=44100",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,setsar=1[v]",
        "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0",
        "-x264-params", "keyint=48:min-keyint=48:scenecut=0",
        "-c:a", "pcm_s16le", str(src),
    )
    media = tmp_path / "damaged.mkv"
    _ffmpeg("-i", str(src), "-map", "0", "-c", "copy", "-fflags", "+bitexact",
            "-bsf:v", "noise=amount=if(eq(n\\,24)\\,1\\,0)", str(media))
    counts = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-count_packets",
         "-count_frames", "-show_entries", "stream=nb_read_packets,nb_read_frames",
         "-of", "json", str(media)],
        capture_output=True, text=True, check=True,
    ).stdout
    stream = json.loads(counts)["streams"][0]
    # The fixture really has one packet the decoder outputs nothing for.
    assert (int(stream["nb_read_packets"]), int(stream["nb_read_frames"])) == (96, 95)
    return media


@needs_ffmpeg
@pytest.mark.slow
def test_frame_lost_mid_stream_shifts_nothing_and_is_reported(tmp_path: Path) -> None:
    media = _damaged_clip(tmp_path)
    doc = _run_video(media, tmp_path / "lib")
    shots = doc["shots"]
    # Indices count decoded frames: the blue shot starts at decoded frame
    # 47 (packet 48), and its time is still the cut's own pts.
    assert [(s["start_frame"], s["end_frame"]) for s in shots] == [(0, 47), (47, 95)]
    assert [s["start_seconds"] for s in shots] == [0.0, 2.0]
    # Last frame 3.958 + its 41 ms Matroska duration (1 ms timebase).
    assert shots[1]["end_seconds"] == doc["video_span"]["end_seconds"] == 3.999
    assert doc["timing"]["undecodable_frames"] == 1
    assert doc["timing"]["vfr"] is True  # the 2-frame gap at 1.0 s
    # Shot 0's midpoint (1.0 s) is the lost frame: the nearest frame that
    # exists is used (tie → earlier), never a mislabelled neighbour.
    rep0 = shots[0]["representative_frames"][0]
    assert rep0["time_seconds"] == 0.958
    assert _colour(doc["_track"], rep0) == "red"
    assert shots[1]["representative_frames"][0]["time_seconds"] == 3.0
    assert _colour(doc["_track"], shots[1]["representative_frames"][0]) == "blue"
    assert shots[0]["motion"]["p95"] < 0.002 and shots[1]["motion"]["p95"] < 0.002


def _bframe_avi(tmp_path: Path, codec: list[str]) -> Path:
    """Red 2 s → blue 2 s at 25 fps with B-frames, PCM audio, in AVI. AVI
    packets carry dts only, so decoded frames' times are reconstructed by
    the decoder: the first comes out at 0.08 s and the delayed frames
    flushed at end of stream carry no timestamp at all (review V-R1)."""
    media = tmp_path / "bframes.avi"
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=red:s=320x240:r=25:d=2",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=25:d=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4:sample_rate=44100",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,setsar=1[v]",
        "-map", "[v]", "-map", "2:a", *codec, "-pix_fmt", "yuv420p",
        "-c:a", "pcm_s16le", "-fflags", "+bitexact", str(media),
    )
    counts = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-count_packets",
         "-count_frames", "-show_entries", "stream=nb_read_packets,nb_read_frames",
         "-of", "json", str(media)],
        capture_output=True, text=True, check=True,
    ).stdout
    stream = json.loads(counts)["streams"][0]
    assert (int(stream["nb_read_packets"]), int(stream["nb_read_frames"])) == (100, 100)
    return media


@needs_ffmpeg
@pytest.mark.slow
@pytest.mark.parametrize(("codec", "first"), [
    # first decoded frame time = the decoder's reorder delay (2 / 1 frames);
    # h264 flushes 2 untimed frames at the end, mpeg4 1.
    (["-c:v", "libx264", "-bf", "3"], 0.08),
    (["-c:v", "mpeg4", "-bf", "2"], 0.04),
], ids=["h264", "mpeg4"])
def test_bframe_avi_keeps_real_clock_and_reports_no_loss(
    tmp_path: Path, codec: list[str], first: float
) -> None:
    doc = _run_video(_bframe_avi(tmp_path, codec), tmp_path / "lib")
    # The real container clock survives the untimed trailing frames: the
    # stream starts at its first decoded frame, not at a synthesised 0.
    assert doc["timing"]["offset_seconds"] == first
    assert doc["video_span"] == {"start_seconds": first,
                                 "end_seconds": round(first + 4.0, 3)}
    shots = doc["shots"]
    assert [(s["start_frame"], s["end_frame"]) for s in shots] == [(0, 50), (50, 100)]
    assert [s["start_seconds"] for s in shots] == [first, round(first + 2.0, 3)]
    # 100 packets, 100 decoded frames: dts-only packets are not matched
    # against presentation times, so no false loss is reported.
    assert doc["timing"]["undecodable_frames"] == 0
    assert doc["timing"]["vfr"] is False
    reps = [s["representative_frames"][0] for s in shots]
    assert [_colour(doc["_track"], r) for r in reps] == ["red", "blue"]
