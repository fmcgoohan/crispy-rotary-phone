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
import contextlib
import itertools
import json
import subprocess
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from . import TOOL_VERSION, canonical, config, costs, hashing, inputs, versions
from .library import (
    Library,
    PrerequisiteError,
    append_api_ledger,
    failure_message,
    mark_dependents_stale,
)
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
# A frame-interval deviation from the median beyond this fraction — and
# beyond VFR_MIN_DEVIATION_SECONDS — marks the stream VFR. The absolute floor
# absorbs container timebase rounding: Matroska/WebM store 1 ms ticks, so a
# CFR stream's intervals alternate between two values one tick apart
# (41/42 ms at 24 fps, 8/9 ms at 120 fps — 12.5 % of the median, review C-3).
# A 30→15 fps switch or a dropped frame deviates by a whole frame interval.
VFR_TOLERANCE = 0.1
VFR_MIN_DEVIATION_SECONDS = 0.0011
# Motion samples and shot starts are compared with this slack, so a sample
# that lands exactly on a frame's pts (the fps filter shows that frame) is
# not lost to float error in pts − origin arithmetic.
MOTION_TIE_SLACK_SECONDS = 1e-6
# Frame lookup by timestamp (extraction, first-decoded trim): pts_time is
# printed to 1 µs; frames are always far more than this apart.
PTS_MATCH_SECONDS = 1e-4
# Packets read past the first keyframe to find the first decoded frame.
FIRST_DECODE_PACKETS = 32
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
    VFR_TOLERANCE of it and by more than one container tick
    (VFR_MIN_DEVIATION_SECONDS)."""
    intervals = [b - a for a, b in itertools.pairwise(pts)]
    if len(intervals) < 2:
        return False
    median = sorted(intervals)[len(intervals) // 2]
    if median <= 0:
        return True
    tolerance = max(VFR_TOLERANCE * median, VFR_MIN_DEVIATION_SECONDS)
    return any(abs(i - median) > tolerance for i in intervals)


def trim_to_first_decoded(pts: list[float], first_decoded: float | None) -> list[float]:
    """Drop table entries presented before the first frame the decoder
    actually outputs (review C-2). A stream cut with `-c copy -copyinkf`
    starts with non-keyframes whose packets exist but which the decoder
    discards; counting them shifts every index and spans frames that never
    decode. `first_decoded` None (nothing decoded in the probe) keeps the
    table as is — extraction then fails loudly if frames are missing."""
    if first_decoded is None:
        return pts
    kept = [t for t in pts if t >= first_decoded - PTS_MATCH_SECONDS]
    if not kept:
        raise VideoError("decoder outputs no frame of the video stream's packet table")
    return kept


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

    def raw_time(self, i: int) -> float:
        """Unrounded unified-timeline time of frame i (i == n_frames: the
        end of the last frame) — for comparisons against motion samples,
        which are on the same unrounded clock."""
        t = self.pts[i] if i < len(self.pts) else self.end
        return t - self.origin

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
    its samples are in this shot (VID-3). start/end must be the shot's
    UNROUNDED frame times (Timeline.raw_time): the motion pass samples
    with round=up, so sample k shows the latest frame presented at or
    before its time, and membership by time is membership by the frame
    it shows (review C-1). The pair straddling a cut is
    flow between two unrelated images and belongs to no shot; assigning
    pairs by midpoint used to put one such spike into a neighbouring shot
    per cut, dominating p95 of short static shots.
    """
    import numpy as np

    lo, hi = start - MOTION_TIE_SLACK_SECONDS, end - MOTION_TIE_SLACK_SECONDS
    inside = [
        v for k, v in enumerate(pair_values)
        if lo <= sample_times[k] < hi and lo <= sample_times[k + 1] < hi
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
    # Video streams in the file, attached pictures included. Above 1, a
    # reader that picks its own stream (OpenCV, under scenedetect) may not
    # pick this one — see _detector_input.
    n_video_streams: int = 1


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
    n_video = sum(1 for s in info["streams"] if s.get("codec_type") == "video")
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
        n_video_streams=n_video,
    )


def _frame_table(path: Path, stream: VideoStreamInfo) -> tuple[list[float], float]:
    """Presentation-time table of the analysed stream: ffprobe packet pts,
    demux only (no decode), sorted — see frame_table_from_packets — then
    trimmed to the first frame the decoder outputs (review C-2)."""
    info = _ffprobe_json(
        ["-select_streams", str(stream.index),
         "-show_entries", "packet=pts_time,dts_time,duration_time,flags",
         str(path)],
        "packets",
    )
    packets = info.get("packets", [])
    pts, end = frame_table_from_packets(packets, stream.fps)
    if any(
        p.get("pts_time", p.get("dts_time")) in (None, "N/A")
        for p in packets if "D" not in (p.get("flags") or "")
    ):
        return pts, end  # synthesised CFR table: no clock to align with
    first_key = next(
        (i for i, p in enumerate(packets) if "K" in (p.get("flags") or "")), 0
    )
    first = _first_decoded_pts(path, stream.index, first_key + FIRST_DECODE_PACKETS)
    return trim_to_first_decoded(pts, first), end


def _first_decoded_pts(path: Path, stream_index: int, n_packets: int) -> float | None:
    """Earliest pts among the frames the decoder outputs from the first
    `n_packets` packets of the stream (None: no frame, or no timestamps).
    Reading up to the first keyframe plus a margin covers the frames a
    decoder drops before it (leading non-keyframes) and B-frame reorder."""
    info = _ffprobe_json(
        ["-select_streams", str(stream_index), "-read_intervals", f"%+#{n_packets}",
         "-show_entries", "frame=pts_time,best_effort_timestamp_time",
         "-show_frames", str(path)],
        "first decoded frame",
    )
    times = []
    for fr in info.get("frames", []):
        t = fr.get("pts_time", fr.get("best_effort_timestamp_time"))
        if t in (None, "N/A"):
            t = fr.get("best_effort_timestamp_time")
        if t not in (None, "N/A"):
            times.append(float(t))
    return min(times) if times else None


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
    """The audio stream ingest decoded: recorded in source.json from 1.1.0.

    Older tracks (source.json 1.0.0) were decoded with ffmpeg's automatic
    stream choice, which ranks streams by channel count and demux state —
    not the 1.1.0 rule and not reproducible from ffprobe. With a single
    audio stream there is nothing to choose; with several, guessing could
    measure t0 on a different stream than source_audio.flac holds and
    silently shift every time, so the track must be re-ingested (review
    C-4)."""
    recorded = source_doc.get("media", {}).get("audio", {}).get("stream_index")
    if recorded is not None:
        return int(recorded)
    streams = _ffprobe_json(["-show_streams", str(path)], "streams")["streams"]
    audio = [st for st in streams if st.get("codec_type") == "audio"]
    if not audio:
        raise VideoError("no audio stream — the unified timeline has no origin")
    if len(audio) > 1:
        raise PrerequisiteError(
            f"{len(audio)} audio streams, and this track's source.json predates "
            "recording which one ingest decoded (media.audio.stream_index) — "
            "the audio clock cannot be determined. Re-ingest the track: remove "
            "its directory from the library and run `mrw ingest` again"
        )
    return int(audio[0]["index"])


def _extract_frames_rgb(path: Path, indices: list[int], width: int, height: int,
                        stream_index: int | None = None,
                        pts: list[float] | None = None):
    """Single sequential decode of the requested frames.

    With `pts` (the frame table), each frame is selected by its own
    presentation time on the container clock (-copyts keeps the demuxer's
    timestamps, the ones ffprobe reported), so a decoder that outputs
    fewer frames than the packet table can never hand back a different
    frame under the requested index (review C-2) — a frame that does not
    decode makes the byte-count check below fail instead. Without `pts`,
    the select filter's `n` (decoder output order) is used."""
    import numpy as np

    if not indices:
        return {}
    wanted_idx = sorted(set(indices))
    if pts is not None:
        expr = "+".join(
            f"between(t\\,{pts[i] - PTS_MATCH_SECONDS!r}\\,{pts[i] + PTS_MATCH_SECONDS!r})"
            for i in wanted_idx
        )
        copyts = ["-copyts"]
    else:
        expr = "+".join(f"eq(n\\,{i})" for i in wanted_idx)
        copyts = []
    mapping = ["-map", f"0:{stream_index}"] if stream_index is not None else []
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", *copyts, "-i", str(path), *mapping,
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
         # round=up: sample k shows the latest frame presented at or
         # before k / MOTION_FPS — never a frame from its future, so a
         # sample's time always lies inside the shot of the frame it shows
         # (review C-1; the default round=near let the first sample of a
         # new shot carry a time up to half a sample before the cut).
         "-vf", (f"setpts=PTS-STARTPTS,fps={MOTION_FPS}:round=up,"
                 f"scale={MOTION_WIDTH}:{height}"),
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


@contextlib.contextmanager
def _detector_input(path: Path, stream: VideoStreamInfo) -> Iterator[Path]:
    """A file whose only video stream is the analysed one, for scenedetect.

    OpenCV (scenedetect's backend) takes the file's first video stream,
    attached pictures included, and has no stream selector; frame table,
    extraction and motion use `stream.index` (-map 0:<index>). When the
    file holds any other video stream, the analysed one is stream-copied
    (no re-encode: packets and their relative timing are unchanged) into a
    temporary Matroska file that is removed afterwards. A file with a
    single video stream is passed through untouched, with no extra work."""
    if stream.n_video_streams <= 1:
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="mrw-detect-") as tmp:
        out = Path(tmp) / "stream.mkv"
        try:
            proc = subprocess.run(
                ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
                 "-map", f"0:{stream.index}", "-c", "copy",
                 "-fflags", "+bitexact", "-f", "matroska", str(out)],
                capture_output=True,
                check=False,
                timeout=600,  # stream copy only — no decode
            )
        except subprocess.TimeoutExpired as e:
            raise VideoError("isolating the video stream timed out after 600 s") from e
        if proc.returncode != 0:
            raise VideoError(
                f"isolating video stream {stream.index} for shot detection "
                f"failed: {proc.stderr.decode()[:300]}"
            )
        yield out


def _detect_cuts(path: Path, vcfg: config.VideoConfig, stream: VideoStreamInfo,
                 timeline: Timeline) -> list[int]:
    """Frame indices where ContentDetector starts a new shot.

    The detector reads exactly the analysed stream (_detector_input).
    Scenedetect 0.7 positions are PTS-derived (OpenCV CAP_PROP_POS_MSEC,
    relative to the stream's start_time — unchanged by a stream copy, which
    shifts every timestamp of the stream alike); their frame_num is only an
    average-rate approximation on VFR input, so it is never used. Each cut
    time is mapped back to the container clock and to the nearest frame of
    the frame table."""
    from scenedetect import ContentDetector, detect

    fps = stream.fps if stream.fps > 0 else 24.0
    min_len_frames = max(1, round(vcfg.min_shot_seconds * fps))
    with _detector_input(path, stream) as detector_path:
        scenes = detect(
            str(detector_path),
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
    """The file the stage analyses: the first candidate whose bytes are the
    ingested video (DET-3/ORCH-2/VID-5). Library copies source_video.* are
    tried first, then original_path. Only a candidate of the recorded size
    is hashed (~1 s/GB, small next to the stage's decode passes), so a
    truncated leftover copy costs nothing (SEC-9).

    A copy that fails the hash check does not hide an intact original
    (review A-3): the stage falls back to it. When nothing verifies, the
    error names each rejected file and the remedy that fits it.
    """
    expected = source_doc["file"]["sha256"]
    size = source_doc["file"]["size_bytes"]
    original = Path(source_doc["file"]["original_path"])
    copies = [p for p in sorted(track_dir.glob("source_video.*"))
              if p.is_file() and p.stat().st_size == size]
    problems: list[str] = []
    for copy in copies:
        actual = hashing.sha256_file(copy)
        if actual == expected:
            return copy
        problems.append(
            f"the library copy {copy} is damaged (sha256 {actual}, expected "
            f"{expected}) — delete it, or replace it with `mrw ingest "
            f"<the original video> --update`"
        )
    if original.is_file():
        actual = (
            hashing.sha256_file(original)
            if original.stat().st_size == size
            else f"of a {original.stat().st_size}-byte file"
        )
        if actual == expected:
            return original
        problems.append(
            f"the file at {original} is not the ingested video (sha256 "
            f"{actual}, expected {expected}) — restore the original there, "
            "or ingest the new file as a new track"
        )
    elif not problems:
        raise PrerequisiteError(
            f"original video not found at {original} (OQ-2: it is referenced, "
            "not copied) — restore it there, or relink it with `mrw ingest "
            "<its new path> --update` (with ingest.copy_video = true that also "
            "keeps a verified copy in the library)"
        )
    else:
        problems.append(f"the original is not at {original}")
    raise PrerequisiteError("; ".join(problems))


def run_video(
    track: str, library: Library, cfg: config.Config, estimate_only: bool = False
) -> VideoResult:
    track_id = library.resolve_track_id(track)
    if estimate_only:
        # Read-only (records nothing, spends nothing): no run lock needed.
        return _run_video(track_id, library, cfg, estimate_only)
    with library.stage_lock(track_id, "video"):
        return _run_video(track_id, library, cfg, estimate_only)


def _run_video(
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

    vcfg = cfg.video
    video_hash = config.stage_hash(vcfg)
    inputs_sha = inputs.current_inputs("video", manifest, source_doc)
    prior = manifest.documents.video
    if (
        not estimate_only
        and prior.status == "ok"
        and prior.config_hash == video_hash
        and versions.is_current("video", prior.algorithm_version)
        and inputs.inputs_match(prior.inputs_sha256, inputs_sha)
        and (track_dir / "video.json").is_file()
    ):
        return VideoResult(track_id, -1, vcfg.caption_backend, None, None, True)

    # Resolved and verified only when the stage will read the file (DET-3):
    # an up-to-date track whose original moved stays a no-op, and bytes
    # that are not the ingested video are refused before any analysis.
    video_path = _resolve_video_path(track_dir, source_doc)

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    backend = None  # visible to the failure path for partial-spend recording
    recorded = False
    # Identity of the document on disk — what a failed entry carries
    # forward (review A-1); the new one once it has been written.
    on_disk_sha = prior.content_sha256

    def _record(entry: DocumentEntry) -> None:
        def apply(m) -> None:
            m.documents.video = entry
            append_api_ledger(m, "video", entry.status, entry.run)
            # Keyed on the document on disk changing, not on status: a
            # failed entry carries the on-disk identity forward (review A-1).
            if entry.content_sha256 != prior.content_sha256:
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
            video_path, all_indices, src_w, src_h, stream.index, timeline.pts
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
                    motion=shot_motion(
                        motion_values, motion_times,
                        timeline.raw_time(s_frame), timeline.raw_time(e_frame),
                    ),
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
        on_disk_sha = content_sha

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
                inputs_sha256=inputs_sha,
                run=RunMetadata(
                    started_at=started_at,
                    duration_seconds=round(time.monotonic() - t0, 2),
                    tool_version=TOOL_VERSION,
                    api_usage=usage,
                ),
            )
        )
        recorded = True
        return VideoResult(
            track_id, len(shots), vcfg.caption_backend, estimate, usage, False
        )
    except PrerequisiteError:
        raise
    except BaseException as exc:
        # BaseException (SEC-9): an interrupted run still records failed
        # with its partial spend (and ledger entry), then the interruption
        # propagates unwrapped. Not after the ok entry is committed: that
        # would mark a good video.json failed and append its spend to the
        # ledger a second time (review A-2).
        if recorded:
            raise
        import shutil as _shutil

        _shutil.rmtree(track_dir / ".frames.tmp", ignore_errors=True)
        partial = _accrued_usage(estimate)
        _record(
            DocumentEntry(
                status="failed",
                # The document on disk is unchanged by the failure, so its
                # identity carries forward (review A-1): dependents built
                # from it stay current, and an identical re-run does not
                # mark them stale.
                content_sha256=on_disk_sha,
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
                error=failure_message(exc)[:500],
            )
        )
        if not isinstance(exc, Exception):
            raise
        raise VideoError(str(exc)[:500]) from exc
