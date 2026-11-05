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
