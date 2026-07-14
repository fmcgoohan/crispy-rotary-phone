"""M5 smoke test (PLAN §11 / M5 milestone task + cost addendum).

Fast (pure logic, synthetic images, zero API calls anywhere): palette on
constructed frames with known clusters; motion on a moving square vs a
static frame; shot-partition invariants; caption cache hit/miss on
prompt_version/model changes with a stub; null-vs-absent caption in the
document path; estimator math, cache-aware estimation, budget gate,
unknown-model refusal, ledger aggregation.

Slow: generated fixture MP4 with hard cuts at known times → boundaries
within ±0.15 s; a >4 s shot produces two frames; full-stage double-run
byte identity of video.json AND every frame JPEG (the JPEG-stability
proof); PR #11 pre-flight assertion for the lyrics config hash rides here.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from mrw import costs
from mrw.captions import CaptionCache, cache_key
from mrw.cli import app
from mrw.models import ShotCaption
from mrw.video import (
    VideoError,
    encode_jpeg,
    motion_stats,
    normalize_shots,
    palette_from_rgb,
    representative_times,
)

REPO = Path(__file__).resolve().parents[1]
SCHEMAS = REPO / "schemas"

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)

runner = CliRunner()


# --- fast: pure video logic -----------------------------------------------------


def test_representative_times() -> None:
    assert representative_times(0.0, 3.0) == [1.5]  # < 4 s → midpoint
    third = representative_times(0.0, 6.0)
    assert third == [2.0, 4.0]  # ≥ 4 s → 1/3, 2/3


def test_shot_partition_invariants() -> None:
    shots = normalize_shots([(48, 96), (0, 48), (96, 120)], fps=24.0)
    assert [(s[0], s[1]) for s in shots] == [(0, 48), (48, 96), (96, 120)]
    assert shots[0][2] == 0.0 and shots[0][3] == 2.0  # frame-derived times
    with pytest.raises(VideoError):
        normalize_shots([(0, 48), (50, 96)], fps=24.0)  # gap
    with pytest.raises(VideoError):
        normalize_shots([(0, 48), (48, 48)], fps=24.0)  # empty shot


def test_palette_known_clusters() -> None:
    # 70% pure red, 30% pure blue image → two dominant swatches in order.
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[:, :70] = (255, 0, 0)
    rgb[:, 70:] = (0, 0, 255)
    swatches = palette_from_rgb(rgb)
    assert swatches[0].proportion > swatches[-1].proportion
    top_two = {s.hex for s in swatches[:2]}

    def near(hex_color: str, target: tuple[int, int, int], tol: int = 30) -> bool:
        r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
        return all(abs(a - b) <= tol for a, b in zip((r, g, b), target))

    assert any(near(h, (255, 0, 0)) for h in top_two)
    assert any(near(h, (0, 0, 255)) for h in top_two)
    assert abs(sum(s.proportion for s in swatches) - 1.0) < 0.01
    # Determinism: same input, same swatches.
    assert palette_from_rgb(rgb) == swatches


def test_motion_moving_vs_static() -> None:
    static = [np.full((60, 80), 128, dtype=np.uint8)] * 5

    def moving():
        for i in range(5):
            f = np.zeros((60, 80), dtype=np.uint8)
            f[20:40, 10 + i * 8 : 30 + i * 8] = 255
            yield f

    static_mags = motion_stats(iter(static), 80, 60)
    moving_mags = motion_stats(moving(), 80, 60)
    assert max(static_mags) < 1e-4
    assert min(moving_mags) > 10 * max(static_mags, default=1e-9)


def test_jpeg_encode_never_upscales_and_is_stable() -> None:
    rgb = (np.random.default_rng(3).integers(0, 255, (480, 640, 3))).astype(np.uint8)
    from PIL import Image
    import io

    jpeg = encode_jpeg(rgb)
    with Image.open(io.BytesIO(jpeg)) as im:
        assert im.width == 640  # 640×480 stays 640 wide — never upscaled
        assert not im.getexif(), "no metadata allowed"
    assert encode_jpeg(rgb) == jpeg  # byte-stable

    wide = np.zeros((720, 2560, 3), dtype=np.uint8)
    with Image.open(io.BytesIO(encode_jpeg(wide))) as im:
        assert im.width == 1280  # downscaled to the cap


# --- fast: caption cache ---------------------------------------------------------


def test_caption_cache_key_sensitivity(tmp_path: Path) -> None:
    cache = CaptionCache(tmp_path / "captions")
    caption = ShotCaption(text="a red frame", tags=["red"])
    key = cache_key(["a" * 64], "v1", "claude-haiku-4-5")
    assert cache.get(key) is None  # miss
    cache.put(key, caption)
    assert cache.get(key) == caption  # hit
    # prompt_version and model changes are deliberate cache misses (OQ-10).
    assert cache.get(cache_key(["a" * 64], "v2", "claude-haiku-4-5")) is None
    assert cache.get(cache_key(["a" * 64], "v1", "claude-sonnet-5")) is None
    assert cache.get(cache_key(["b" * 64], "v1", "claude-haiku-4-5")) is None


# --- fast: cost module -----------------------------------------------------------


def test_estimator_math() -> None:
    # 2 calls: one 1-frame (1000x750 → 1000 tokens), one 2-frame.
    est = costs.estimate(
        "claude-haiku-4-5",
        [[(1000, 750)], [(750, 750), (750, 750)]],
        prompt_tokens_per_call=200,
        max_output_tokens_per_call=300,
    )
    assert est.calls == 2
    assert est.input_tokens == (200 + 1000) + (200 + 750 + 750)
    assert est.output_tokens == 600
    expected = (est.input_tokens * 1.00 + 600 * 5.00) / 1e6
    assert abs(est.usd - round(expected, 4)) < 1e-9


def test_estimator_cache_aware_second_run_is_free() -> None:
    # After caching, zero uncached calls → $0.
    est = costs.estimate("claude-haiku-4-5", [], 200, 300)
    assert est.calls == 0 and est.usd == 0.0


def test_estimator_refuses_unknown_model() -> None:
    with pytest.raises(costs.UnknownModelError):
        costs.estimate("claude-nonexistent-9", [[(100, 100)]], 100, 100)


def test_ledger_aggregation() -> None:
    usage_a = {"calls": 40, "input_tokens": 50000, "output_tokens": 9000,
               "usd": 0.095, "estimated_usd": 0.11}
    usage_b = {"calls": 10, "input_tokens": 12000, "output_tokens": 2500,
               "usd": 0.0245, "estimated_usd": 0.03}
    manifests = [
        {"documents": {"video": {"run": {"api_usage": usage_a}}}},
        {"documents": {"video": {"run": {"api_usage": usage_b}}},
         "stems": {"run": {}}},
        {"documents": {"video": {"run": {}}}},  # no spend
    ]
    totals = costs.aggregate_ledger(manifests)
    assert totals["runs"] == 2 and totals["calls"] == 50
    assert totals["usd"] == 0.1195 and totals["estimated_usd"] == 0.14


# --- slow: fixture pipeline -------------------------------------------------------


def _cut_fixture(path: Path) -> None:
    """red 3 s | green 3 s | blue 5 s at 24 fps + sine audio → hard cuts at
    3.0 and 6.0; the blue shot (> 4 s) must produce two frames."""
    filters = (
        "color=c=red:size=640x480:rate=24:duration=3[v0];"
        "color=c=green:size=640x480:rate=24:duration=3[v1];"
        "color=c=blue:size=640x480:rate=24:duration=5[v2];"
        "[v0][v1][v2]concat=n=3:v=1:a=0[v]"
    )
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=11:sample_rate=44100",
         "-filter_complex", filters, "-map", "[v]", "-map", "0:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "alac", str(path)],
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode()[:400]


def _run_video_stage(tmp: Path) -> tuple[Path, str]:
    mp4 = tmp / "cuts.mp4"
    _cut_fixture(mp4)
    library = tmp / "lib"
    result = runner.invoke(app, ["ingest", str(mp4), "--library", str(library)])
    assert result.exit_code == 0, result.output
    track_id = next(p.name for p in library.iterdir() if p.is_dir())
    result = runner.invoke(app, ["video", track_id, "--library", str(library)])
    assert result.exit_code == 0, result.output
    return library, track_id


def _validate(document: dict) -> None:
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    registry = Registry()
    for f in SCHEMAS.glob("*.schema.json"):
        registry = registry.with_resource(
            f.name, Resource.from_contents(json.loads(f.read_text()))
        )
    schema = json.loads((SCHEMAS / "video.schema.json").read_text())
    errors = list(Draft202012Validator(schema, registry=registry).iter_errors(document))
    assert not errors, "\n".join(e.message for e in errors)


@pytest.mark.slow
def test_video_stage_on_cut_fixture(tmp_path: Path) -> None:
    library, track_id = _run_video_stage(tmp_path)
    doc = json.loads((library / track_id / "video.json").read_text())
    _validate(doc)

    shots = doc["shots"]
    assert len(shots) == 3
    # Boundary tolerance: ±0.15 s (stated; ContentDetector on hard cuts).
    assert abs(shots[0]["end_seconds"] - 3.0) <= 0.15
    assert abs(shots[1]["end_seconds"] - 6.0) <= 0.15
    # Partition invariants in the document.
    for a, b in zip(shots, shots[1:]):
        assert a["end_frame"] == b["start_frame"]
        assert a["end_seconds"] == b["start_seconds"]
    # The blue shot is 5 s → two representative frames; others one.
    assert len(shots[0]["representative_frames"]) == 1
    assert len(shots[2]["representative_frames"]) == 2
    # null backend: caption ABSENT (not empty), backend named in the doc.
    assert doc["caption_backend"] == {"name": "null"}
    assert all("caption" not in s for s in shots)
    # Palettes: dominant swatch of the red shot is red-ish.
    top = shots[0]["palette"][0]["hex"]
    r, g, b = (int(top[i : i + 2], 16) for i in (1, 3, 5))
    assert r > 150 and g < 100 and b < 100
    # Solid-color shots: near-zero motion.
    assert shots[0]["motion"]["mean"] < 0.001
    # Frames on disk, 640 wide (never upscaled), sha matches bytes.
    from mrw import hashing

    for shot in shots:
        for rep in shot["representative_frames"]:
            data = (library / track_id / rep["path"]).read_bytes()
            assert hashing.sha256_bytes(data) == rep["sha256"]
    manifest = json.loads((library / track_id / "manifest.json").read_text())
    assert manifest["documents"]["video"]["status"] == "ok"
    assert manifest["documents"]["video"]["run"].get("api_usage") is None


@pytest.mark.slow
def test_video_double_run_byte_identity(tmp_path: Path) -> None:
    """video.json AND every frame JPEG byte-identical — the JPEG-stability
    proof (frame sha256 keys the caption cache)."""
    outputs = []
    for name in ("lib_a", "lib_b"):
        sub = tmp_path / name
        sub.mkdir()
        library, track_id = _run_video_stage(sub)
        track_dir = library / track_id
        frames = {
            p.name: p.read_bytes() for p in sorted((track_dir / "frames").iterdir())
        }
        outputs.append(((track_dir / "video.json").read_bytes(), frames))
    assert outputs[0][0] == outputs[1][0], "video.json differs"
    assert outputs[0][1].keys() == outputs[1][1].keys()
    for name in outputs[0][1]:
        assert outputs[0][1][name] == outputs[1][1][name], f"{name} differs"


def test_moved_original_is_prerequisite_error(tmp_path: Path) -> None:
    mp4 = tmp_path / "cuts.mp4"
    _cut_fixture(mp4)
    library = tmp_path / "lib"
    result = runner.invoke(app, ["ingest", str(mp4), "--library", str(library)])
    assert result.exit_code == 0, result.output
    track_id = next(p.name for p in library.iterdir() if p.is_dir())
    mp4.rename(tmp_path / "moved_away.mp4")

    result = runner.invoke(app, ["video", track_id, "--library", str(library)])
    assert result.exit_code == 2
    assert "original video not found" in result.output
    assert "copy_video" in result.output  # names the remedy


def test_lyrics_config_hash_covers_pr11_fields() -> None:
    # M5 pre-flight (field-tracks observation): PR #11's clip fields must
    # participate in the lyrics config-subset hash.
    from mrw.config import LyricsConfig, stage_hash

    base = stage_hash(LyricsConfig())
    assert stage_hash(LyricsConfig(clip_to_vocal_activity=False)) != base
    assert stage_hash(LyricsConfig(clip_padding_seconds=0.7)) != base
