"""P0-1 (VID-1): shot palettes are clustered from pixels, not row averages."""

from __future__ import annotations

import numpy as np

from mrw.video import palette_from_samples, palette_sample


def _rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))


def _near(hex_color, target, tol=30) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(_rgb(hex_color), target))


def test_full_resolution_two_colour_frame_keeps_both_colours():
    # Left 70 % red, right 30 % blue at 1080p: the stage path (per-frame
    # downscale, then pool). The pre-fix N×1 pooling turned this into purples.
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:, :1344] = (255, 0, 0)
    frame[:, 1344:] = (0, 0, 255)
    swatches = palette_from_samples(palette_sample(frame))
    assert _near(swatches[0].hex, (255, 0, 0))
    assert abs(swatches[0].proportion - 0.7) < 0.02
    assert any(_near(s.hex, (0, 0, 255)) and abs(s.proportion - 0.3) < 0.02
               for s in swatches)
    assert not any(_near(s.hex, (128, 0, 128), tol=60) for s in swatches)


def test_pooling_across_frames_weights_pixels():
    red = np.full((720, 1280, 3), (255, 0, 0), dtype=np.uint8)
    blue = np.full((720, 1280, 3), (0, 0, 255), dtype=np.uint8)
    samples = np.concatenate([palette_sample(red), palette_sample(blue)])
    assert samples.shape == (2 * 256 * 144, 3)
    swatches = palette_from_samples(samples)
    top = sorted(swatches, key=lambda s: -s.proportion)[:2]
    assert {(_near(s.hex, (255, 0, 0)), _near(s.hex, (0, 0, 255))) for s in top} \
        == {(True, False), (False, True)}
    assert all(abs(s.proportion - 0.5) < 0.01 for s in top)
    assert palette_from_samples(samples) == swatches  # deterministic


# --- P0-2 (VID-4): rotated (portrait phone) video ------------------------------

import shutil
import subprocess

import pytest

from mrw.video import _extract_frames_rgb, _motion_pass, _video_stream_info

def _ffmpeg_has_display_rotation() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-h", "full"],
                         capture_output=True, text=True).stdout
    return "display_rotation" in out  # input option added in FFmpeg 6


needs_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_has_display_rotation(),
    reason="needs ffmpeg >= 6 (-display_rotation) to build the fixture",
)


def _rotated_clip(tmp_path):
    # 320x180 stored, tagged with a 90° display matrix (how phones store
    # portrait video). Left half red, right half blue in stored orientation.
    plain = tmp_path / "plain.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "color=c=red:s=160x180:r=10:d=1",
         "-f", "lavfi", "-i", "color=c=blue:s=160x180:r=10:d=1",
         "-filter_complex", "[0][1]hstack", "-pix_fmt", "yuv444p",
         "-c:v", "libx264", "-qp", "0", str(plain)],
        check=True,
    )
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-display_rotation", "90",
         "-i", str(plain), "-c", "copy", str(rotated)],
        check=True,
    )
    return rotated


@needs_ffmpeg
def test_rotated_video_frames_have_display_geometry(tmp_path):
    clip = _rotated_clip(tmp_path)
    fps, w, h, n = _video_stream_info(clip)
    assert (w, h) == (180, 320)  # portrait, not the stored 320x180
    frame = _extract_frames_rgb(clip, [0], w, h)[0]
    assert frame.shape == (320, 180, 3)
    # Upright frame: the stored left/right halves become top/bottom (or
    # bottom/top) halves, each a solid colour — never interleaved stripes.
    top, bottom = frame[:150], frame[170:]
    for half in (top, bottom):
        assert half.std(axis=(0, 1)).max() < 8
    assert {tuple(top.mean(axis=(0, 1)) > 128), tuple(bottom.mean(axis=(0, 1)) > 128)} \
        == {(True, False, False), (False, False, True)}
    values, times = _motion_pass(clip, w, h)
    assert len(values) == len(times) and max(values) < 0.002  # static scene


@needs_ffmpeg
def test_frame_geometry_mismatch_fails_loudly(tmp_path):
    from mrw.video import VideoError

    clip = _rotated_clip(tmp_path)
    with pytest.raises(VideoError, match="bytes"):
        _extract_frames_rgb(clip, [0], 320, 179)
