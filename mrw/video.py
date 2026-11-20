"""Stage 5 — video analysis (PLAN §4 stage 5); contract:
schemas/video.schema.json v1.1.0. Video inputs only.

Timeline (VID-2): every time in video.json is on the unified audio
timeline — t = 0 is the first sample of source_audio.flac, i.e. the
presentation time of the first decoded frame of the audio stream ingest
chose (ingest.select_audio_stream). Video frame times come from a
presentation-time table of the analysed video stream (ffprobe packet pts,
demux only, sorted): frame index i (decode/presentation order, the `n` of
ffmpeg's select filter) is at pts[i] − t0. Nothing assumes a constant
frame rate, so VFR sources get exact indices and times; the offset applied
and a vfr indicator are recorded in the document. Frames presented before
t = 0 (video that starts before the audio) are outside the timeline and
excluded from the shots.

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

import bisect
import itertools
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from . import TOOL_VERSION, canonical, config, costs, hashing, versions
from .library import Library, PrerequisiteError, mark_dependents_stale
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
    VideoTiming,
)

JPEG_QUALITY = 90
MAX_FRAME_WIDTH = 1280
MOTION_FPS = 12
MOTION_WIDTH = 320
PALETTE_K = 5
PALETTE_MAX_DIM = 256
LONG_SHOT_SECONDS = 4.0
# A frame-interval deviation from the median beyond this fraction marks the
# stream VFR. Container timebase rounding (e.g. Matroska's 1 ms: 41/42 ms at
# 24 fps) stays far below it; a 30→15 fps switch or a dropped frame does not.
VFR_TOLERANCE = 0.1
# Frames up to half a millisecond before t = 0 round to 0.000 and stay on
# the timeline; anything earlier precedes the audio and is excluded.
ORIGIN_SLOP_SECONDS = 0.0005


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
    boundaries: list[tuple[int, int]], frame_times: list[float]
) -> list[tuple[int, int, float, float]]:
    """(start_frame, end_frame) list → sorted, contiguous, gap-free
    partition with exclusive end_frame. Times come from `frame_times`
    (seconds on the unified timeline, len = frame count + 1: entry i is
    frame i's presentation time, the last entry the end of the last frame),
    never from frame/fps — the stream may be VFR (VID-2)."""
    out = []
    for s, e in sorted(boundaries):
        if out and s != out[-1][1]:
            raise VideoError(f"shot partition has a gap/overlap at frame {s}")
        if e <= s:
            raise VideoError(f"empty shot at frame {s}")
        if e >= len(frame_times):
            raise VideoError(f"shot ends at frame {e}, past the last frame")
        out.append((s, e, frame_times[s], frame_times[e]))
    return out


def frame_table_from_packets(
    packets: list[dict], fallback_fps: float
) -> tuple[list[float], float]:
    """(sorted presentation times, end of the last frame) on the container
    clock, from ffprobe `packet` entries of ONE video stream.

    Demux only — no decode. Packets flagged discard ('D', e.g. before an
    MP4 edit-list start) are never output by the decoder and are skipped.
    Packets without pts fall back to dts; a stream with no timestamps at
    all is synthesised at `fallback_fps` from 0 (CFR assumption).
    """
    frames: list[tuple[float, float]] = []  # (pts, duration or 0)
    n_output = 0
    missing = False  # any packet without pts/dts → synthesise the table
    for pkt in packets:
        if "D" in (pkt.get("flags") or ""):
            continue
        n_output += 1
        t = pkt.get("pts_time", pkt.get("dts_time"))
        if t in (None, "N/A"):
            missing = True
            continue
        d = pkt.get("duration_time")
        frames.append((float(t), float(d) if d not in (None, "N/A") else 0.0))
    if missing or not frames:
        if n_output == 0 or fallback_fps <= 0:
            raise VideoError("video stream has no frames with timestamps")
        return [i / fallback_fps for i in range(n_output)], n_output / fallback_fps
    frames.sort()
    times = [t for t, _ in frames]
    # Length of the last frame: its own packet duration, else the last
    # frame interval (VFR-safe — never an average rate), else 1/avg fps.
    last = frames[-1][1]
    if last <= 0:
        if len(times) > 1 and times[-1] > times[-2]:
            last = times[-1] - times[-2]
        else:
            last = 1.0 / fallback_fps if fallback_fps > 0 else 0.04
    return times, times[-1] + last


def is_vfr(pts: list[float]) -> bool:
    """True when frame intervals are not constant (beyond timebase
    rounding): any interval deviates from the median by more than
    VFR_TOLERANCE of it."""
    intervals = [b - a for a, b in itertools.pairwise(pts)]
    if len(intervals) < 2:
        return False
    median = sorted(intervals)[len(intervals) // 2]
    if median <= 0:
        return True
    return any(abs(i - median) > VFR_TOLERANCE * median for i in intervals)


@dataclass(frozen=True)
class Timeline:
    """The analysed video stream's frames placed on the unified timeline.

    pts: sorted presentation times (container clock), one per frame;
    end: end of the last frame (container clock);
    origin: container time of the first sample of source_audio.flac (t0).
    """

    pts: list[float]
    end: float
    origin: float

    @property
    def n_frames(self) -> int:
        return len(self.pts)

    @property
    def first_frame(self) -> int:
        """First frame presented at or after t = 0 (earlier frames precede
        the audio and are outside the timeline)."""
        return bisect.bisect_left(self.pts, self.origin - ORIGIN_SLOP_SECONDS)

    @property
    def offset_seconds(self) -> float:
        """First video frame's time minus the audio origin (may be < 0)."""
        return canonical.round_seconds(self.pts[0] - self.origin)

    def frame_times(self) -> list[float]:
        """Document times (3 dp, clamped at 0) for frames 0..n−1 plus the
        end of the last frame — the table normalize_shots consumes."""
        return [
            canonical.round_seconds(max(0.0, t - self.origin))
            for t in [*self.pts, self.end]
        ]

    def index_near(self, container_time: float) -> int:
        """Frame whose pts is nearest to a container-clock time (ties →
        earlier frame)."""
        i = bisect.bisect_left(self.pts, container_time)
        if i == 0:
            return 0
        if i == len(self.pts):
            return len(self.pts) - 1
        before, after = self.pts[i - 1], self.pts[i]
        return i - 1 if container_time - before <= after - container_time else i


def shot_bounds(cut_indices: list[int], first: int, n_frames: int) -> list[tuple[int, int]]:
    """Frame-index partition [first, n_frames) split at the given cuts.

    The last shot ends at n_frames — the frames that exist in the stream —
    never at a container-duration estimate (VID-11)."""
    if first >= n_frames:
        raise VideoError("no video frames at or after audio t = 0")
    cuts = sorted({c for c in cut_indices if first < c < n_frames})
    edges = [first, *cuts, n_frames]
    return list(itertools.pairwise(edges))


def representative_index(frame_times: list[float], s: int, e: int, t: float) -> int:
    """Frame in [s, e) whose time is nearest t (ties → earlier frame).
    Clamped to the shot, hence to the frames that exist."""
    i = bisect.bisect_left(frame_times, t, s, e)
    if i >= e:
        return e - 1
    if i > s and t - frame_times[i - 1] <= frame_times[i] - t:
        return i - 1
    return i


def palette_sample(frame):
    """One frame's palette pixels: the H×W×3 frame downscaled (INTER_AREA)
    to <= PALETTE_MAX_DIM on its long edge, flattened to (N, 3).

    Downscaling must happen on the 2-D image. Pooling full-resolution
    pixels into an N×1 column first and resizing that (the pre-fix code)
    averaged runs of N/256 raster-order pixels — whole image rows — so a
    half-red, half-blue frame clustered as purples.
    """
    import cv2

    h, w = frame.shape[:2]
    scale = PALETTE_MAX_DIM / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(
            frame, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return frame.reshape(-1, 3)


def palette_from_rgb(rgb) -> list[PaletteSwatch]:
    """Palette of a single H×W×3 frame (see palette_from_samples)."""
    return palette_from_samples(palette_sample(rgb))


def palette_from_samples(samples) -> list[PaletteSwatch]:
    """k-means (k=5, seed 0, n_init=1) in CIELab over (N, 3) uint8 RGB
    samples (from palette_sample, pooled across a shot's frames); swatches
    as sRGB hex + proportion, sorted by proportion desc (ties: darker hex
    first)."""
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans

    rgb = np.ascontiguousarray(samples, dtype=np.uint8).reshape(-1, 1, 3)
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


def shot_motion(pair_values: list[float], sample_times: list[float],
                start: float, end: float) -> Motion:
    """Motion of one shot from the 12 fps pass.

    sample_times[k] is motion sample k's time on the unified timeline
    (len = len(pair_values) + 1); pair k is the flow between samples k and
    k+1. Each SAMPLE belongs to the shot whose [start, end) contains its
    own time — the shots' partition — and a pair counts only when both of
    its samples are in this shot (VID-3). The pair straddling a cut is
    flow between two unrelated images and belongs to no shot; assigning
    pairs by midpoint used to put one such spike into a neighbouring shot
    per cut, dominating p95 of short static shots.
    """
    import numpy as np

    inside = [
        v for k, v in enumerate(pair_values)
        if start <= sample_times[k] < end and start <= sample_times[k + 1] < end
    ]
    if not inside:
        # H2 degenerate convention (PR #12 review): a shot too short to
        # contain two 12 fps motion samples (< ~1/6 s) carries no motion
        # evidence and reports 0.0/0.0 — deliberately identical to a
        # static shot, and stated in the schema's motion description.
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


def _rotation_degrees(stream: dict) -> int:
    """Display rotation of a video stream (display-matrix side data, or the
    legacy `rotate` tag), normalised to 0/90/180/270."""
    rot = None
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            rot = sd["rotation"]
            break
    if rot is None:
        rot = (stream.get("tags") or {}).get("rotate")
    try:
        return int(round(float(rot))) % 360 if rot is not None else 0
    except (TypeError, ValueError):
        return 0


def _display_dims(stream: dict) -> tuple[int, int]:
    """(width, height) of decoded frames. ffmpeg auto-rotates on decode, so
    a ±90°/270° stream (portrait phone video stored as landscape) decodes
    with its stored dimensions swapped (VID-4)."""
    width, height = int(stream["width"]), int(stream["height"])
    if _rotation_degrees(stream) in (90, 270):
        width, height = height, width
    return width, height


@dataclass(frozen=True)
class VideoStreamInfo:
    index: int  # container stream index of the analysed video stream
    width: int  # display (post-rotation) dimensions of decoded frames
    height: int
    fps: float  # average frame rate — sizing hints only, never timing
    start_time: float  # stream start_time (container clock)


def _ffprobe_json(args: list[str], what: str) -> dict:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", *args],
            capture_output=True,
            check=False,
            timeout=300,  # demux/header work only; longer means a hung process
        )
    except subprocess.TimeoutExpired as e:
        raise VideoError(f"ffprobe ({what}) timed out after 300 s") from e
    if proc.returncode != 0:
        raise VideoError(f"ffprobe ({what}) failed: {proc.stderr.decode()[:300]}")
    return json.loads(proc.stdout)


def _video_stream_info(path: Path) -> VideoStreamInfo:
    info = _ffprobe_json(["-show_streams", str(path)], "streams")
    stream = next(
        (
            s for s in info["streams"]
            if s.get("codec_type") == "video"
            and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    if stream is None:
        raise VideoError("no video stream (cover art is not video)")
    try:
        fps = float(Fraction(stream.get("avg_frame_rate", "0/1")))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    width, height = _display_dims(stream)
    start = stream.get("start_time")
    return VideoStreamInfo(
        index=int(stream["index"]),
        width=width,
        height=height,
        fps=fps,
        start_time=float(start) if start not in (None, "N/A") else 0.0,
    )


def _frame_table(path: Path, stream: VideoStreamInfo) -> tuple[list[float], float]:
    """Presentation-time table of the analysed stream: ffprobe packet pts,
    demux only (no decode), sorted — see frame_table_from_packets."""
    info = _ffprobe_json(
        ["-select_streams", str(stream.index),
         "-show_entries", "packet=pts_time,dts_time,duration_time,flags",
         str(path)],
        "packets",
    )
    return frame_table_from_packets(info.get("packets", []), stream.fps)


def _audio_origin(path: Path, audio_index: int) -> float:
    """Container time of the first sample of source_audio.flac: the pts of
    the first frame the decoder outputs for the audio stream ingest decoded.

    Decoded, not the stream's start_time: priming samples dropped through
    discard packets / skip_samples shift the first output sample, and the
    FLAC (audio-only output) starts exactly there with no padding. Only the
    first few packets are read."""
    info = _ffprobe_json(
        ["-select_streams", str(audio_index), "-read_intervals", "%+#16",
         "-show_frames", "-show_streams", str(path)],
        "audio origin",
    )
    times = []
    for fr in info.get("frames", []):
        t = fr.get("best_effort_timestamp_time", fr.get("pts_time"))
        if t not in (None, "N/A"):
            times.append(float(t))
    if times:
        return min(times)
    for st in info.get("streams", []):
        if st.get("start_time") not in (None, "N/A"):
            return float(st["start_time"])
    return 0.0


def _audio_stream_index(path: Path, source_doc: dict) -> int:
    """The audio stream ingest decoded: recorded in source.json from 1.1.0;
    for older tracks, the same selection rule re-applied to the file."""
    recorded = source_doc.get("media", {}).get("audio", {}).get("stream_index")
    if recorded is not None:
        return int(recorded)
    from .ingest import select_audio_stream

    streams = _ffprobe_json(["-show_streams", str(path)], "streams")["streams"]
    chosen, _ = select_audio_stream(streams)
    if chosen is None:
        raise VideoError("no audio stream — the unified timeline has no origin")
    return int(chosen["index"])


def _extract_frames_rgb(path: Path, indices: list[int], width: int, height: int,
                        stream_index: int | None = None):
    """Single sequential decode; exact frame indices via the select filter
    (its `n` is the decoder's output order — the frame table's order)."""
    import numpy as np

    if not indices:
        return {}
    expr = "+".join(f"eq(n\\,{i})" for i in sorted(set(indices)))
    mapping = ["-map", f"0:{stream_index}"] if stream_index is not None else []
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), *mapping,
         "-vf", f"select='{expr}'", "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise VideoError(f"frame extraction failed: {proc.stderr.decode()[:300]}")
    frame_bytes = width * height * 3
    raw = proc.stdout
    wanted = len(set(indices))
    if len(raw) != wanted * frame_bytes:
        # A size mismatch means the frame geometry is wrong (rotation,
        # aspect, mid-stream resize) or the stream decoded fewer frames
        # than its packet table — reshaping would scramble pixels.
        raise VideoError(
            f"decode returned {len(raw)} bytes for {wanted} frames of "
            f"{width}x{height}; expected {wanted * frame_bytes}"
        )
    frames = {}
    for slot, idx in enumerate(sorted(set(indices))):
        chunk = raw[slot * frame_bytes : (slot + 1) * frame_bytes]
        frames[idx] = np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)
    return frames


def _motion_pass(path: Path, w0: int, h0: int, stream_index: int | None = None,
                 first_frame_seconds: float = 0.0) -> tuple[list[float], list[float]]:
    """12 fps / 320 px grayscale sequential decode → per-pair magnitudes
    and every sample's time on the unified timeline (one more time than
    magnitudes; pair k joins samples k and k+1 — see shot_motion).

    (w0, h0) are the display (post-rotation) dimensions from
    _video_stream_info, so the scale keeps the true aspect ratio.
    `first_frame_seconds` is the analysed stream's first frame on the
    unified timeline (Timeline.pts[0] − origin, unrounded). The decode is
    re-based to that frame (setpts=PTS-STARTPTS) and timestamps pass
    through untouched (-fps_mode passthrough), so ffmpeg never pads the
    head with copies of the first frame to reach the file's t = 0 (the
    VID-2 motion/shot origin skew): sample k is at first_frame_seconds +
    k / MOTION_FPS — the same clock as the shots.
    """
    import numpy as np

    height = max(2, round(MOTION_WIDTH * h0 / w0 / 2) * 2)
    mapping = ["-map", f"0:{stream_index}"] if stream_index is not None else []
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), *mapping,
         "-vf", f"setpts=PTS-STARTPTS,fps={MOTION_FPS},scale={MOTION_WIDTH}:{height}",
         "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True,
        check=False,
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
    sample_times = [first_frame_seconds + k / MOTION_FPS for k in range(n)]
    return values, sample_times


def _detect_cuts(path: Path, vcfg: config.VideoConfig, stream: VideoStreamInfo,
                 timeline: Timeline) -> list[int]:
    """Frame indices where ContentDetector starts a new shot.

    Scenedetect 0.7 positions are PTS-derived (OpenCV CAP_PROP_POS_MSEC,
    relative to the stream's start_time); their frame_num is only an
    average-rate approximation on VFR input, so it is never used. Each cut
    time is mapped back to the container clock and to the nearest frame of
    the packet table."""
    from scenedetect import ContentDetector, detect

    fps = stream.fps if stream.fps > 0 else 24.0
    min_len_frames = max(1, round(vcfg.min_shot_seconds * fps))
    scenes = detect(
        str(path),
        ContentDetector(threshold=vcfg.detector_threshold,
                        min_scene_len=min_len_frames),
        start_in_scene=True,
    )
    return [
        timeline.index_near(stream.start_time + start.seconds)
        for start, _ in scenes[1:]
    ]


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
        and versions.is_current("video", prior.algorithm_version)
        and (track_dir / "video.json").is_file()
    ):
        return VideoResult(track_id, -1, vcfg.caption_backend, None, None, True)

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    backend = None  # visible to the failure path for partial-spend recording

    def _record(entry: DocumentEntry) -> None:
        def apply(m) -> None:
            m.documents.video = entry
            if entry.status == "ok" and entry.content_sha256 != prior.content_sha256:
                mark_dependents_stale(m, "video")

        apply(manifest)
        library.update_manifest(track_id, apply)

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

        stream = _video_stream_info(video_path)
        src_w, src_h = stream.width, stream.height
        pts, end = _frame_table(video_path, stream)
        timeline = Timeline(
            pts=pts,
            end=end,
            origin=_audio_origin(
                video_path, _audio_stream_index(video_path, source_doc)
            ),
        )
        frame_times = timeline.frame_times()
        cuts = _detect_cuts(video_path, vcfg, stream, timeline)
        bounds = shot_bounds(cuts, timeline.first_frame, timeline.n_frames)
        shots_raw = normalize_shots(bounds, frame_times)

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
                [representative_index(frame_times, s_frame, e_frame, t)
                 for t in times]
            )
        all_indices = sorted({i for shot in plan for i in shot})
        frames_rgb = _extract_frames_rgb(
            video_path, all_indices, src_w, src_h, stream.index
        )
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
        motion_values, motion_times = _motion_pass(
            video_path, src_w, src_h, stream.index,
            timeline.pts[0] - timeline.origin,
        )

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
                        time_seconds=frame_times[frame_idx],
                        path=f"frames/{name}",
                        sha256=shas[frame_idx],
                    )
                )
            import numpy as np

            pooled = np.concatenate(
                [palette_sample(frames_rgb[i]) for i in shot_indices]
            )
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
                    palette=palette_from_samples(pooled),
                    motion=shot_motion(motion_values, motion_times, s_t, e_t),
                    caption=caption,
                )
            )

        document = VideoDocument(
            track_id=track_id,
            video_span=Interval(
                start_seconds=shots_raw[0][2], end_seconds=shots_raw[-1][3]
            ),
            timing=VideoTiming(
                offset_seconds=timeline.offset_seconds,
                vfr=is_vfr(timeline.pts),
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
                algorithm_version=versions.current("video"),
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
