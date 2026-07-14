"""Stage 5 — video analysis (PLAN §4 stage 5); contract:
schemas/video.schema.json v1.0.0 (frozen). Video inputs only.

Shot detection: PySceneDetect ContentDetector (OQ-8). Per shot:
representative frames (midpoint <4 s; 1/3 and 2/3 otherwise), CIELab
k-means palette (k=5, fixed seed, sklearn), Farnebäck motion (12 fps,
320 px pass, diagonal-normalized), pluggable captions (mrw/captions.py).

Determinism (D5 extension — JPEG byte stability is load-bearing):
- Frame extraction is by exact source FRAME INDEX (ffmpeg select filter on
  frame number, single sequential decode), never time-seek.
- Exactly one encoder touches pixels: ffmpeg decodes to raw RGB; Pillow
  does all resizing (Lanczos) and the JPEG encode with pinned settings
  (quality 90, 4:2:0 subsampling, no EXIF/metadata/timestamps). Frames are
  downscaled to ≤1280 px wide and NEVER upscaled.
- k-means: fixed seed, n_init=1; rounding per the precision contract
  (proportions/motion 3 dp; palette hex exact).
- Captions are deterministic-given-cache (OQ-10; see mrw/captions.py) and
  budget-gated through mrw/costs.py before any spend.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from . import TOOL_VERSION, canonical, config, costs, hashing
from .library import Library, PrerequisiteError
from .models import (
    ApiUsage,
    CaptionBackendInfo,
    Detector,
    DocumentEntry,
    Interval,
    Motion,
    PaletteSwatch,
    RepresentativeFrame,
    RunMetadata,
    Shot,
    VideoDocument,
)

JPEG_QUALITY = 90
MAX_FRAME_WIDTH = 1280
MOTION_FPS = 12
MOTION_WIDTH = 320
PALETTE_K = 5
PALETTE_MAX_DIM = 256
LONG_SHOT_SECONDS = 4.0


class VideoError(RuntimeError):
    """Stage failure — recorded in the manifest as status=failed; exit 1."""


class BudgetExceeded(PrerequisiteError):
    """Pre-flight estimate over caption_budget_usd_per_run; nothing spent."""


@dataclass
class VideoResult:
    track_id: str
    n_shots: int
    backend: str
    estimate: costs.CostEstimate | None
    usage: ApiUsage | None
    already_done: bool
    estimated_only: bool = False


# --- pure helpers (fast-tested without any video) -----------------------------


def representative_times(start: float, end: float) -> list[float]:
    """Midpoint for shots < 4 s; 1/3 and 2/3 for longer (schema)."""
    length = end - start
    if length < LONG_SHOT_SECONDS:
        return [start + length / 2.0]
    return [start + length / 3.0, start + 2.0 * length / 3.0]


def normalize_shots(
    boundaries: list[tuple[int, int]], fps: float
) -> list[tuple[int, int, float, float]]:
    """(start_frame, end_frame) list → sorted, contiguous, gap-free
    partition with exclusive end_frame and frame-derived times."""
    out = []
    for i, (s, e) in enumerate(sorted(boundaries)):
        if out and s != out[-1][1]:
            raise VideoError(f"shot partition has a gap/overlap at frame {s}")
        if e <= s:
            raise VideoError(f"empty shot at frame {s}")
        out.append((s, e, canonical.round_seconds(s / fps), canonical.round_seconds(e / fps)))
    return out


def palette_from_rgb(rgb) -> list[PaletteSwatch]:
    """k-means (k=5, seed 0, n_init=1) in CIELab; swatches as sRGB hex +
    proportion, sorted by proportion desc (ties: darker hex first)."""
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans

    h, w = rgb.shape[:2]
    scale = PALETTE_MAX_DIM / max(h, w)
    if scale < 1.0:
        rgb = cv2.resize(
            rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    # cv2 uint8 Lab is uint8-scaled (L*255/100, a/b offset +128); keep the
    # clustering AND the reverse conversion in that same space.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).reshape(-1, 3).astype("float64")
    k = min(PALETTE_K, len(np.unique(lab, axis=0)))
    km = KMeans(n_clusters=k, n_init=1, random_state=0).fit(lab)
    counts = np.bincount(km.labels_, minlength=k).astype("float64")
    centers_u8 = np.clip(np.round(km.cluster_centers_), 0, 255).astype("uint8")
    centers_rgb = cv2.cvtColor(
        centers_u8.reshape(1, -1, 3), cv2.COLOR_Lab2RGB
    ).reshape(-1, 3).astype(int)
    swatches = []
    total = counts.sum()
    for i in range(k):
        r, g, b = (int(x) for x in centers_rgb[i])
        swatches.append(
            PaletteSwatch(
                hex=f"#{r:02x}{g:02x}{b:02x}",
                proportion=canonical.round_ratio(counts[i] / total),
            )
        )
    swatches.sort(key=lambda s: (-s.proportion, s.hex))
    return swatches


def motion_stats(gray_frames, width: int, height: int) -> list[float]:
    """Per-pair Farnebäck mean flow magnitude, diagonal-normalized (pure
    over an iterable of grayscale frames)."""
    import cv2
    import numpy as np

    diagonal = float(np.hypot(width, height))
    mags = []
    prev = None
    for frame in gray_frames:
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mags.append(float(np.linalg.norm(flow, axis=2).mean()) / diagonal)
        prev = frame
    return mags


def shot_motion(pair_values: list[float], pair_times: list[float],
                start: float, end: float) -> Motion:
    import numpy as np

    inside = [v for v, t in zip(pair_values, pair_times) if start <= t < end]
    if not inside:
        # H2 degenerate convention (PR #12 review): a shot too short to
        # contain any 12 fps motion-sample pair (< ~1/6 s) carries no
        # motion evidence and reports 0.0/0.0 — deliberately identical to
        # a static shot, and stated in the schema's motion description.
        return Motion(mean=0.0, p95=0.0)
    arr = np.array(inside)
    return Motion(
        mean=canonical.round_ratio(float(arr.mean())),
        p95=canonical.round_ratio(float(np.percentile(arr, 95))),
    )


def encode_jpeg(rgb) -> bytes:
    """The single pinned JPEG encoder (D5): Pillow, quality 90, 4:2:0,
    optimize off, progressive off, NO metadata of any kind."""
    import io

    from PIL import Image

    img = Image.fromarray(rgb, mode="RGB")
    if img.width > MAX_FRAME_WIDTH:  # never upscale
        new_h = max(1, round(img.height * MAX_FRAME_WIDTH / img.width))
        img = img.resize((MAX_FRAME_WIDTH, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, subsampling=2,
             optimize=False, progressive=False, exif=b"")
    return buf.getvalue()


# --- decode helpers ------------------------------------------------------------


def _video_stream_info(path: Path) -> tuple[float, int, int, int]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-print_format",
         "json", "-show_streams", "-show_format", str(path)],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise VideoError(f"ffprobe failed: {proc.stderr.decode()[:300]}")
    info = json.loads(proc.stdout)
    stream = next(
        s for s in info["streams"]
        if s.get("codec_type") == "video"
        and not s.get("disposition", {}).get("attached_pic")
    )
    fps = float(Fraction(stream["avg_frame_rate"]))
    width, height = int(stream["width"]), int(stream["height"])
    nb = stream.get("nb_frames")
    frame_count = int(nb) if nb else int(round(float(info["format"]["duration"]) * fps))
    return fps, width, height, frame_count


def _extract_frames_rgb(path: Path, indices: list[int], width: int, height: int):
    """Single sequential decode; exact frame indices via the select filter."""
    import numpy as np

    if not indices:
        return {}
    expr = "+".join(f"eq(n\\,{i})" for i in sorted(set(indices)))
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"select='{expr}'", "-vsync", "0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise VideoError(f"frame extraction failed: {proc.stderr.decode()[:300]}")
    frame_bytes = width * height * 3
    raw = proc.stdout
    frames = {}
    for slot, idx in enumerate(sorted(set(indices))):
        chunk = raw[slot * frame_bytes : (slot + 1) * frame_bytes]
        if len(chunk) < frame_bytes:
            raise VideoError(f"decode returned too few frames (wanted {idx})")
        frames[idx] = np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)
    return frames


def _motion_pass(path: Path) -> tuple[list[float], list[float]]:
    """12 fps / 320 px grayscale sequential decode → per-pair magnitudes."""
    import numpy as np

    probe_h_proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True,
    )
    w0, h0 = (int(x) for x in probe_h_proc.stdout.decode().strip().split(",")[:2])
    height = max(2, round(MOTION_WIDTH * h0 / w0 / 2) * 2)
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"fps={MOTION_FPS},scale={MOTION_WIDTH}:{height}",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise VideoError(f"motion pass failed: {proc.stderr.decode()[:300]}")
    frame_bytes = MOTION_WIDTH * height
    n = len(proc.stdout) // frame_bytes
    frames = (
        np.frombuffer(proc.stdout[i * frame_bytes : (i + 1) * frame_bytes],
                      dtype=np.uint8).reshape(height, MOTION_WIDTH)
        for i in range(n)
    )
    values = motion_stats(frames, MOTION_WIDTH, height)
    pair_times = [(i + 0.5) / MOTION_FPS for i in range(len(values))]
    return values, pair_times


def _detect_shots(path: Path, vcfg: config.VideoConfig, fps: float,
                  frame_count: int) -> list[tuple[int, int]]:
    from scenedetect import ContentDetector, detect

    min_len_frames = max(1, round(vcfg.min_shot_seconds * fps))
    scenes = detect(
        str(path),
        ContentDetector(threshold=vcfg.detector_threshold,
                        min_scene_len=min_len_frames),
    )
    if not scenes:
        return [(0, frame_count)]
    bounds = [(s.frame_num, e.frame_num) for s, e in scenes]
    # Guarantee full coverage of the stream.
    if bounds[0][0] != 0:
        bounds[0] = (0, bounds[0][1])
    if bounds[-1][1] < frame_count:
        bounds[-1] = (bounds[-1][0], frame_count)
    return bounds


# --- the stage -----------------------------------------------------------------


def _resolve_video_path(track_dir: Path, source_doc: dict) -> Path:
    copies = sorted(track_dir.glob("source_video.*"))
    if copies:
        return copies[0]
    original = Path(source_doc["file"]["original_path"])
    if original.is_file():
        return original
    raise PrerequisiteError(
        f"original video not found at {original} (OQ-2: it is referenced, "
        "not copied) — restore it there, or re-ingest with "
        "ingest.copy_video = true"
    )


def run_video(
    track: str, library: Library, cfg: config.Config, estimate_only: bool = False
) -> VideoResult:
    track_id = library.resolve_track_id(track)
    manifest = library.read_manifest(track_id)
    if manifest is None or manifest.documents.source.status != "ok":
        raise PrerequisiteError(f"track {track_id} has no successful ingest")
    if not manifest.has_video:
        raise PrerequisiteError(
            f"track {track_id} is audio-only — video stage is not applicable"
        )
    track_dir = library.track_dir(track_id)
    source_doc = json.loads((track_dir / "source.json").read_text(encoding="utf-8"))
    video_path = _resolve_video_path(track_dir, source_doc)

    vcfg = cfg.video
    video_hash = config.stage_hash(vcfg)
    prior = manifest.documents.video
    if (
        not estimate_only
        and prior.status == "ok"
        and prior.config_hash == video_hash
        and (track_dir / "video.json").is_file()
    ):
        return VideoResult(track_id, -1, vcfg.caption_backend, None, None, True)

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    backend = None  # visible to the failure path for partial-spend recording

    def _record(entry: DocumentEntry) -> None:
        manifest.documents.video = entry
        library.write_manifest(track_id, manifest)

    def _accrued_usage(estimated: costs.CostEstimate | None):
        """Real spend so far — recorded even on failure (PR #12 round 3):
        paid caption calls must never vanish from the cost ledger."""
        if backend is None or getattr(backend, "calls", 0) == 0:
            return None
        in_price, out_price = costs.PRICES_PER_MTOK[vcfg.caption_model]
        usd = (
            backend.input_tokens * in_price + backend.output_tokens * out_price
        ) / 1_000_000
        return ApiUsage(
            calls=backend.calls,
            input_tokens=backend.input_tokens,
            output_tokens=backend.output_tokens,
            usd=round(usd, 4),
            estimated_usd=estimated.usd if estimated else None,
        )

    estimate = None
    try:
        from .captions import (
            PROMPT_TOKENS_ESTIMATE,
            CaptionCache,
            cache_key,
            make_backend,
        )

        fps, src_w, src_h, frame_count = _video_stream_info(video_path)
        bounds = _detect_shots(video_path, vcfg, fps, frame_count)
        shots_raw = normalize_shots(bounds, fps)

        # PR #12 round 4 [minor]: a $0 null-backend estimate needs only the
        # shot count — return before any frame extraction/JPEG work.
        if estimate_only and vcfg.caption_backend == "null":
            return VideoResult(
                track_id, len(shots_raw), "null", None, None, False,
                estimated_only=True,
            )

        # Representative frame plan (indices, exact).
        plan: list[list[int]] = []
        for s_frame, e_frame, s_t, e_t in shots_raw:
            times = representative_times(s_t, e_t)
            plan.append(
                [min(int(round(t * fps)), e_frame - 1) for t in times]
            )
        all_indices = sorted({i for shot in plan for i in shot})
        frames_rgb = _extract_frames_rgb(video_path, all_indices, src_w, src_h)
        jpegs = {i: encode_jpeg(frames_rgb[i]) for i in all_indices}
        shas = {i: hashing.sha256_bytes(jpegs[i]) for i in all_indices}

        # Estimation and the budget gate need only the cache — the backend
        # (which requires ANTHROPIC_API_KEY) is constructed after the gate,
        # so `--estimate` never needs a key.
        cache = CaptionCache(track_dir / ".cache" / "captions")

        # Pre-flight cost estimate over UNCACHED shots only (M5 addendum).
        if vcfg.caption_backend != "null":
            from PIL import Image
            import io

            uncached_dims: list[list[tuple[int, int]]] = []
            for shot_indices in plan:
                key = cache_key(
                    [shas[i] for i in shot_indices],
                    vcfg.caption_prompt_version,
                    vcfg.caption_model,
                )
                if cache.get(key) is None:
                    dims = []
                    for i in shot_indices:
                        with Image.open(io.BytesIO(jpegs[i])) as im:
                            dims.append((im.width, im.height))
                    uncached_dims.append(dims)
            estimate = costs.estimate(
                vcfg.caption_model,
                uncached_dims,
                PROMPT_TOKENS_ESTIMATE,
                vcfg.caption_max_output_tokens,
            )
            if estimate_only:
                return VideoResult(
                    track_id, len(shots_raw), vcfg.caption_backend,
                    estimate, None, False, estimated_only=True,
                )
            if estimate.usd > vcfg.caption_budget_usd_per_run:
                raise BudgetExceeded(
                    f"caption estimate {estimate.describe()} exceeds "
                    f"caption_budget_usd_per_run "
                    f"(${vcfg.caption_budget_usd_per_run:.2f}) — raise "
                    "video.caption_budget_usd_per_run in mrw.toml to "
                    "proceed, or run with --estimate to inspect"
                )

        backend, cache = make_backend(vcfg, track_dir / ".cache" / "captions")
        motion_values, motion_times = _motion_pass(video_path)

        # PR #12 review [major]: frames are written to a temp dir and
        # swapped in whole on success (the stems temp-dir-and-rename
        # pattern) — a failed run leaves the previous frames/ intact, and
        # a re-run can never leave stale JPEGs from a prior config beside
        # the new video.json.
        frames_dir = track_dir / "frames"
        frames_tmp = track_dir / ".frames.tmp"
        if frames_tmp.exists():
            import shutil as _shutil

            _shutil.rmtree(frames_tmp)
        frames_tmp.mkdir()
        shots: list[Shot] = []
        for idx, ((s_frame, e_frame, s_t, e_t), shot_indices) in enumerate(
            zip(shots_raw, plan)
        ):
            reps = []
            for n, frame_idx in enumerate(shot_indices, start=1):
                name = f"shot_{idx + 1:04d}_f{n}.jpg"
                (frames_tmp / name).write_bytes(jpegs[frame_idx])
                reps.append(
                    RepresentativeFrame(
                        time_seconds=canonical.round_seconds(frame_idx / fps),
                        path=f"frames/{name}",
                        sha256=shas[frame_idx],
                    )
                )
            import numpy as np

            pooled = np.concatenate(
                [frames_rgb[i].reshape(-1, 3) for i in shot_indices]
            ).reshape(-1, 1, 3)
            caption = backend.caption(
                [jpegs[i] for i in shot_indices],
                [shas[i] for i in shot_indices],
            )
            shots.append(
                Shot(
                    index=idx,
                    start_seconds=s_t,
                    end_seconds=e_t,
                    start_frame=s_frame,
                    end_frame=e_frame,
                    representative_frames=reps,
                    palette=palette_from_rgb(pooled),
                    motion=shot_motion(motion_values, motion_times, s_t, e_t),
                    caption=caption,
                )
            )

        document = VideoDocument(
            track_id=track_id,
            video_span=Interval(
                start_seconds=0.0, end_seconds=shots_raw[-1][3]
            ),
            detector=Detector(
                name="pyscenedetect_content",
                threshold=vcfg.detector_threshold,
                min_shot_seconds=vcfg.min_shot_seconds,
            ),
            caption_backend=CaptionBackendInfo(
                name=vcfg.caption_backend,
                model=vcfg.caption_model if vcfg.caption_backend != "null" else None,
                prompt_version=(
                    vcfg.caption_prompt_version
                    if vcfg.caption_backend != "null"
                    else None
                ),
            ),
            shots=shots,
        )
        # PR #12 round 5 [major]: the document is built, validated and
        # serialized BEFORE the frames swap, so any modelling/serialization
        # failure leaves the previous frames/ AND video.json fully intact.
        # The swap and the (atomic temp-and-rename) document write are then
        # adjacent renames — the only remaining inconsistency window is a
        # hard crash between the two, which the next run heals by
        # rewriting both (the entry records failed/stale either way).
        from .models import doc_dump as _doc_dump

        canonical.dumps(_doc_dump(document))  # validate + serialize first
        import shutil as _shutil

        if frames_dir.exists():
            _shutil.rmtree(frames_dir)
        frames_tmp.rename(frames_dir)
        content_sha = library.write_document(track_id, "video.json", document)

        usage = None
        if vcfg.caption_backend != "null":
            usage = _accrued_usage(estimate) or ApiUsage(
                calls=0, input_tokens=0, output_tokens=0, usd=0.0,
                estimated_usd=estimate.usd if estimate else None,
            )
        _record(
            DocumentEntry(
                status="ok",
                path="video.json",
                schema_version=document.schema_version,
                content_sha256=content_sha,
                config_hash=video_hash,
                run=RunMetadata(
                    started_at=started_at,
                    duration_seconds=round(time.monotonic() - t0, 2),
                    tool_version=TOOL_VERSION,
                    api_usage=usage,
                ),
            )
        )
        return VideoResult(
            track_id, len(shots), vcfg.caption_backend, estimate, usage, False
        )
    except PrerequisiteError:
        raise
    except Exception as exc:
        import shutil as _shutil

        _shutil.rmtree(track_dir / ".frames.tmp", ignore_errors=True)
        partial = _accrued_usage(estimate)
        _record(
            DocumentEntry(
                status="failed",
                config_hash=video_hash,
                run=(
                    RunMetadata(
                        started_at=started_at,
                        duration_seconds=round(time.monotonic() - t0, 2),
                        tool_version=TOOL_VERSION,
                        api_usage=partial,
                    )
                    if partial
                    else None
                ),
                error=str(exc)[:500],
            )
        )
        raise VideoError(str(exc)[:500]) from exc
