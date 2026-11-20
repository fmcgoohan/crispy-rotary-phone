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


# --- VID-2 / VID-11: pure timeline helpers ---------------------------------------

from mrw.video import (  # noqa: E402
    Timeline,
    VideoError,
    frame_table_from_packets,
    is_vfr,
    representative_index,
    shot_bounds,
)


def test_frame_table_sorts_skips_discard_and_uses_last_duration() -> None:
    packets = [  # decode order (B-frames): pts not monotonic
        {"pts_time": "0.500000", "duration_time": "0.033333", "flags": "K__"},
        {"pts_time": "0.566667", "duration_time": "0.033333", "flags": "___"},
        {"pts_time": "0.533333", "duration_time": "0.033333", "flags": "___"},
        {"pts_time": "0.600000", "duration_time": "0.066667", "flags": "___"},
        {"pts_time": "0.466667", "duration_time": "0.033333", "flags": "_D_"},
    ]
    pts, end = frame_table_from_packets(packets, 30.0)
    assert pts == [0.5, 0.533333, 0.566667, 0.6]
    assert end == pytest.approx(0.666667)  # last frame's own duration (VFR-safe)


def test_frame_table_without_timestamps_is_synthesised() -> None:
    packets = [{"pts_time": "N/A", "flags": "K__"}] * 3
    assert frame_table_from_packets(packets, 10.0) == ([0.0, 0.1, 0.2], 0.3)
    with pytest.raises(VideoError):
        frame_table_from_packets([], 10.0)


def test_vfr_indicator() -> None:
    cfr = [i / 24 for i in range(100)]
    mkv_ms = [round(i / 24, 3) for i in range(100)]  # 41/42 ms timebase jitter
    vfr = [i / 30 for i in range(60)] + [2 + i / 15 for i in range(30)]
    assert not is_vfr(cfr) and not is_vfr(mkv_ms)
    assert is_vfr(vfr)


def test_timeline_offsets_and_clamping() -> None:
    pts = [0.5 + i / 24 for i in range(96)]
    late = Timeline(pts=pts, end=4.5, origin=0.0)  # video starts after audio
    assert late.offset_seconds == 0.5 and late.first_frame == 0
    times = late.frame_times()
    assert times[0] == 0.5 and times[48] == 2.5 and times[-1] == 4.5

    early = Timeline(pts=[i / 24 for i in range(96)], end=4.0, origin=0.5)
    assert early.offset_seconds == -0.5
    assert early.first_frame == 12  # frames before audio t = 0 are excluded
    times = early.frame_times()
    assert times[0] == 0.0 and times[12] == 0.0 and times[-1] == 3.5
    assert early.index_near(2.0 + 0.01) == 48  # nearest frame


def test_shot_bounds_end_at_the_frames_that_exist() -> None:
    assert shot_bounds([48, 0, 200], 0, 96) == [(0, 48), (48, 96)]
    assert shot_bounds([], 12, 96) == [(12, 96)]
    assert shot_bounds([5, 30], 12, 96) == [(12, 30), (30, 96)]
    with pytest.raises(VideoError):
        shot_bounds([], 96, 96)


def test_representative_index_is_nearest_and_clamped() -> None:
    times = [i / 10 for i in range(11)]  # 10 frames + end
    assert representative_index(times, 2, 6, 0.42) == 4
    assert representative_index(times, 2, 6, 0.45) == 4  # tie → earlier
    assert representative_index(times, 2, 6, 9.0) == 5  # clamped into shot
    assert representative_index(times, 2, 6, 0.0) == 2


# --- VID-2 / VID-11: the stage on generated clips ---------------------------------


def _run_video(media: Path, library: Path) -> dict:
    track = _ingest(media, library)
    result = runner.invoke(app, ["video", track.name, "--library", str(library)])
    assert result.exit_code == 0, result.output
    doc = json.loads((track / "video.json").read_text())
    assert doc["schema_version"] == "1.1.0"
    shots = doc["shots"]
    assert shots[0]["start_seconds"] == doc["video_span"]["start_seconds"]
    assert shots[-1]["end_seconds"] == doc["video_span"]["end_seconds"]
    for a, b in zip(shots, shots[1:]):
        assert a["end_frame"] == b["start_frame"]
        assert a["end_seconds"] == b["start_seconds"]
    doc["_track"] = track
    return doc


def _red_blue_video(path: Path, seg: tuple[float, float], rates=(24, 24)) -> None:
    """Red then blue, hard cut; with two rates the result is VFR."""
    _ffmpeg(
        "-f", "lavfi", "-i", f"color=c=red:s=320x240:r={rates[0]}:d={seg[0]}",
        "-f", "lavfi", "-i", f"color=c=blue:s=320x240:r={rates[1]}:d={seg[1]}",
        "-filter_complex",
        "[0:v]settb=1/90000,setsar=1[a];[1:v]settb=1/90000,setsar=1[b];"
        "[a][b]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-fps_mode", "passthrough",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    )


def _frame_colour(track: Path, rep: dict) -> str:
    from PIL import Image

    with Image.open(track / rep["path"]) as im:
        r, g, b = np.asarray(im.convert("RGB")).reshape(-1, 3).mean(axis=0)
    return "red" if r > 150 and b < 100 else "blue" if b > 150 and r < 100 else "?"


@needs_ffmpeg
@pytest.mark.slow
def test_video_starting_after_audio_is_on_the_audio_clock(tmp_path: Path) -> None:
    # Video stream starts 0.5 s after the audio (-itsoffset): the cut at
    # 2.0 s of video is at 2.5 s on the audio timeline.
    video = tmp_path / "v.mp4"
    _red_blue_video(video, (2, 2))
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=5:sample_rate=44100",
            "-c:a", "aac", str(tmp_path / "a.m4a"))
    media = tmp_path / "offset.mp4"
    _ffmpeg("-itsoffset", "0.5", "-i", str(video), "-i", str(tmp_path / "a.m4a"),
            "-map", "0:v", "-map", "1:a", "-c", "copy", str(media))
    doc = _run_video(media, tmp_path / "lib")
    assert doc["timing"] == {"offset_seconds": 0.5, "vfr": False}
    assert doc["video_span"] == {"start_seconds": 0.5, "end_seconds": 4.5}
    shots = doc["shots"]
    assert [(s["start_frame"], s["end_frame"]) for s in shots] == [(0, 48), (48, 96)]
    assert shots[0]["end_seconds"] == 2.5
    assert shots[0]["representative_frames"][0]["time_seconds"] == 1.5
    # Motion shares the shots' origin: no padded head frames.
    from mrw.video import _motion_pass

    values, times = _motion_pass(media, 320, 240, 0, 0.5)
    assert len(times) == 48  # 4.0 s at 12 fps, not 54 (no padded head)
    assert len(values) == 47
    assert times[0] == pytest.approx(0.5) and times[24] == pytest.approx(2.5)


@needs_ffmpeg
@pytest.mark.slow
def test_video_starting_before_audio_excludes_pre_origin_frames(tmp_path: Path) -> None:
    # Audio starts 0.5 s after the video: frames before audio t = 0 are
    # outside the timeline; times never go negative.
    video = tmp_path / "v.mp4"
    _red_blue_video(video, (2, 2))
    media = tmp_path / "early.mkv"
    _ffmpeg("-i", str(video), "-itsoffset", "0.5",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4:sample_rate=44100",
            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "pcm_s16le",
            str(media))
    doc = _run_video(media, tmp_path / "lib")
    assert doc["timing"]["offset_seconds"] == -0.5
    shots = doc["shots"]
    assert shots[0]["start_frame"] == 12 and shots[0]["start_seconds"] == 0.0
    assert shots[0]["end_frame"] == 48 and shots[0]["end_seconds"] == 1.5
    # Matroska's 1 ms timebase: frame times are exact to ±1 ms only.
    assert doc["video_span"]["end_seconds"] == pytest.approx(3.5, abs=0.001)


@needs_ffmpeg
@pytest.mark.slow
def test_vfr_clip_gets_exact_frame_indices_and_times(tmp_path: Path) -> None:
    # 2 s red at 30 fps, then 2 s blue at 15 fps: 60 + 30 frames. An
    # average-rate index (22.5 fps) would put the cut at frame 45 and pick
    # representative frames from the wrong shot.
    video = tmp_path / "v.mp4"
    _red_blue_video(video, (2, 2), rates=(30, 15))
    media = tmp_path / "vfr.mp4"
    _ffmpeg("-i", str(video),
            "-f", "lavfi", "-i", "sine=frequency=440:duration=4:sample_rate=44100",
            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", str(media))
    doc = _run_video(media, tmp_path / "lib")
    assert doc["timing"] == {"offset_seconds": 0.0, "vfr": True}
    shots = doc["shots"]
    assert [(s["start_frame"], s["end_frame"]) for s in shots] == [(0, 60), (60, 90)]
    assert shots[0]["end_seconds"] == 2.0
    assert doc["video_span"]["end_seconds"] == 4.0
    red_rep = shots[0]["representative_frames"][0]
    blue_rep = shots[1]["representative_frames"][0]
    assert red_rep["time_seconds"] == 1.0 and blue_rep["time_seconds"] == 3.0
    assert _frame_colour(doc["_track"], red_rep) == "red"
    assert _frame_colour(doc["_track"], blue_rep) == "blue"


@needs_ffmpeg
@pytest.mark.slow
def test_audio_tail_never_extends_the_last_shot(tmp_path: Path) -> None:
    # MKV (no nb_frames): 4.0 s / 96 frames of video, 7 s of audio. The old
    # duration × fps estimate (168 frames) stretched the 1 s blue shot to
    # 4 s and asked the decoder for frames that do not exist.
    video = tmp_path / "v.mp4"
    _red_blue_video(video, (3, 1))
    media = tmp_path / "tail.mkv"
    _ffmpeg("-i", str(video),
            "-f", "lavfi", "-i", "sine=frequency=440:duration=7:sample_rate=44100",
            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "flac", str(media))
    doc = _run_video(media, tmp_path / "lib")
    shots = doc["shots"]
    assert [(s["start_frame"], s["end_frame"]) for s in shots] == [(0, 72), (72, 96)]
    # Matroska's 1 ms timebase: frame times are exact to ±1 ms only.
    assert doc["video_span"]["end_seconds"] == pytest.approx(4.0, abs=0.001)
    rep = shots[1]["representative_frames"][0]
    assert rep["time_seconds"] == pytest.approx(3.5, abs=0.001)
    assert _frame_colour(doc["_track"], rep) == "blue"


# --- VID-3: motion pairs straddling a cut ----------------------------------------


def test_cut_straddling_pair_belongs_to_no_shot() -> None:
    from mrw.video import MOTION_FPS, shot_motion

    # 25 samples (2 s at 12 fps), cut at 1.0 s = sample 12: pair 11 joins
    # samples 11 (shot A) and 12 (shot B) and carries a big spike.
    times = [k / MOTION_FPS for k in range(25)]
    values = [0.0] * 24
    values[11] = 0.05
    a = shot_motion(values, times, 0.0, 1.0)
    b = shot_motion(values, times, 1.0, 2.0)
    assert (a.mean, a.p95) == (0.0, 0.0) and (b.mean, b.p95) == (0.0, 0.0)
    # A pair fully inside a shot still counts.
    values[5] = 0.05
    assert shot_motion(values, times, 0.0, 1.0).p95 > 0.002
    # Too short for two samples → the H2 0.0/0.0 convention.
    assert shot_motion(values, times, 0.95, 1.05).p95 == 0.0


def _texture(path: Path, seed: int, size: tuple[int, int], tint: tuple[int, int, int]) -> None:
    from PIL import Image, ImageFilter

    rng = np.random.default_rng(seed)
    w, h = size
    noise = rng.integers(0, 256, (h // 4, w // 4), dtype=np.uint8)
    img = Image.fromarray(noise).resize((w, h), Image.BICUBIC).filter(
        ImageFilter.GaussianBlur(1)
    )
    grey = np.asarray(img, dtype=np.float64) / 255.0
    rgb = np.stack([grey * c for c in tint], axis=-1).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


@needs_ffmpeg
@pytest.mark.slow
def test_static_textured_shots_stay_static_across_a_hard_cut(tmp_path: Path) -> None:
    # Two 1 s static textured scenes joined by a hard cut, then a 1.5 s
    # scene scrolling 60 px/s. The cross-cut flow between the unrelated
    # textures used to land in one of the 12-pair static shots and push
    # its p95 into the "gentle movement" band.
    _texture(tmp_path / "a.png", 1, (320, 240), (255, 120, 60))
    _texture(tmp_path / "b.png", 2, (320, 240), (60, 120, 255))
    _texture(tmp_path / "c.png", 3, (640, 240), (80, 255, 80))
    media = tmp_path / "textures.mp4"
    _ffmpeg(
        "-loop", "1", "-framerate", "24", "-t", "1", "-i", str(tmp_path / "a.png"),
        "-loop", "1", "-framerate", "24", "-t", "1", "-i", str(tmp_path / "b.png"),
        "-loop", "1", "-framerate", "24", "-t", "1.5", "-i", str(tmp_path / "c.png"),
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3.5:sample_rate=44100",
        "-filter_complex",
        "[0:v]setsar=1[a];[1:v]setsar=1[b];"
        "[2:v]crop=320:240:x='t*60':y=0,setsar=1[c];"
        "[a][b][c]concat=n=3:v=1:a=0,format=yuv420p[v]",
        "-map", "[v]", "-map", "3:a",
        "-c:v", "libx264", "-qp", "0", "-c:a", "aac", str(media),
    )
    doc = _run_video(media, tmp_path / "lib")
    shots = doc["shots"]
    assert [(s["start_frame"], s["end_frame"]) for s in shots] == [
        (0, 24), (24, 48), (48, 84)
    ]
    assert shots[0]["motion"]["p95"] < 0.002
    assert shots[1]["motion"]["p95"] < 0.002
    assert shots[2]["motion"]["mean"] > 0.002  # real movement still measured
