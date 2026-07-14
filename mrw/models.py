"""Pydantic models for the analysis documents (M0 scope: source + manifest).

Field order here IS the canonical key order in written documents — keep it in
sync with the hand-drafted schemas in schemas/. Serialization convention:
absent optional fields are omitted (never null) — dump with exclude_none=True.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

# Per-document schema versions (PR #6 review, S2: never share a version
# constant across documents — schemas evolve independently).
SOURCE_SCHEMA_VERSION = "1.0.0"
AUDIO_FEATURES_SCHEMA_VERSION = "1.0.0"
# lyrics 1.1.0: additive — engine.language_source, outside_vocal_activity
# flag, untranscribed_regions sharpened to uncovered spans (review 007).
LYRICS_SCHEMA_VERSION = "1.1.0"
VIDEO_SCHEMA_VERSION = "1.0.0"
# 1.1.0: additive — `error` on the manifest stems block (M2).
# 1.2.0: additive — `warning` on the stems block (PR #5 review: persist the
# MPS-fallback reason for batch runs).
# 1.3.0: additive — optional `api_usage` on run metadata (M5 cost ledger).
MANIFEST_SCHEMA_VERSION = "1.3.0"

Status = Literal["ok", "pending", "failed", "stale", "not_applicable"]


def doc_dump(model: BaseModel) -> dict:
    """Standard serialization for documents: JSON mode, absent-not-null."""
    return model.model_dump(mode="json", exclude_none=True)


# --- source.json ------------------------------------------------------------


class SourceFile(BaseModel):
    original_path: str
    filename: str
    size_bytes: int
    sha256: str


class NormalizedAudio(BaseModel):
    path: str
    sample_rate: int
    channels: int
    duration_seconds: float
    audio_stream_sha256: str


class MediaTags(BaseModel):
    title: Optional[str] = None
    artist: Optional[str] = None


class MediaAudio(BaseModel):
    codec: str
    sample_rate: int
    channels: int
    bit_rate: Optional[int] = None


class MediaVideo(BaseModel):
    codec: str
    width: int
    height: int
    fps: float
    frame_count: int


class Media(BaseModel):
    container: str
    duration_seconds: float
    tags: Optional[MediaTags] = None
    audio: MediaAudio
    video: Optional[MediaVideo] = None


class LyricsInput(BaseModel):
    path: str
    format: Literal["lrc", "txt"]
    sha256: str
    line_count: int
    has_timestamps: bool


class SourceDocument(BaseModel):
    schema_version: str = SOURCE_SCHEMA_VERSION
    track_id: str
    file: SourceFile
    normalized_audio: NormalizedAudio
    media: Media
    lyrics_input: Optional[LyricsInput] = None


# --- audio_features.json ----------------------------------------------------


class Timeseries(BaseModel):
    unit: str
    start_seconds: float
    hop_seconds: float
    values: list[float]


class Tempo(BaseModel):
    bpm_global: float
    confidence: float


class Beats(BaseModel):
    times: list[float]
    beats_per_bar: int
    downbeat_offset: int
    downbeat_method: Literal["assumed_4_4_phase_fit", "model"]


class Loudness(BaseModel):
    integrated_lufs: float


class Onsets(BaseModel):
    times: list[float]
    strengths: list[float]
    strength_reference: float


class ChannelFeatures(BaseModel):
    rms_db: Timeseries
    spectral_centroid_hz: Timeseries
    onsets: Onsets


class StemFeatures(BaseModel):
    vocals: ChannelFeatures
    drums: ChannelFeatures
    bass: ChannelFeatures
    other: ChannelFeatures


class VocalActivityParams(BaseModel):
    enter_db: float
    exit_db: float
    min_region_seconds: float
    min_gap_seconds: float


class VocalRegion(BaseModel):
    start_seconds: float
    end_seconds: float
    mean_rms_db: float


class VocalActivity(BaseModel):
    params: VocalActivityParams
    regions: list[VocalRegion]


class AudioFeaturesDocument(BaseModel):
    schema_version: str = AUDIO_FEATURES_SCHEMA_VERSION
    track_id: str
    duration_seconds: float
    tempo: Tempo
    beats: Beats
    loudness: Loudness
    mix: ChannelFeatures
    stems: StemFeatures
    vocal_activity: VocalActivity


# --- video.json ---------------------------------------------------------------


class Interval(BaseModel):
    start_seconds: float
    end_seconds: float


class Detector(BaseModel):
    name: str
    threshold: float
    min_shot_seconds: float


class CaptionBackendInfo(BaseModel):
    name: str
    model: Optional[str] = None
    prompt_version: Optional[str] = None


class RepresentativeFrame(BaseModel):
    time_seconds: float
    path: str
    sha256: str


class PaletteSwatch(BaseModel):
    hex: str
    proportion: float


class Motion(BaseModel):
    mean: float
    p95: float


class ShotCaption(BaseModel):
    text: str
    tags: list[str]


class Shot(BaseModel):
    index: int
    start_seconds: float
    end_seconds: float
    start_frame: int
    end_frame: int
    representative_frames: list[RepresentativeFrame]
    palette: list[PaletteSwatch]
    motion: Motion
    caption: Optional[ShotCaption] = None


class VideoDocument(BaseModel):
    schema_version: str = VIDEO_SCHEMA_VERSION
    track_id: str
    video_span: Interval
    detector: Detector
    caption_backend: CaptionBackendInfo
    shots: list[Shot]


# --- lyrics.json ------------------------------------------------------------


class LyricsEngine(BaseModel):
    name: str
    model: str
    language_source: Optional[
        Literal["pinned", "detected_vocal_window", "default_no_vocal_activity"]
    ] = None


class LyricsWord(BaseModel):
    text: str
    start_seconds: float
    end_seconds: float
    confidence: float


class LyricsLine(BaseModel):
    text: str
    start_seconds: float
    end_seconds: float
    confidence: Optional[float] = None
    flags: list[str]
    words: list[LyricsWord]


class SuppliedMarkup(BaseModel):
    text: str
    label: Optional[
        Literal["intro", "verse", "chorus", "bridge", "outro", "other"]
    ] = None
    source_line_index: int
    hint_seconds: Optional[float] = None


class UntranscribedRegion(BaseModel):
    start_seconds: float
    end_seconds: float


class LyricsCoverage(BaseModel):
    vocal_activity_covered_ratio: float
    lines_flagged_ratio: float


class LyricsDocument(BaseModel):
    schema_version: str = LYRICS_SCHEMA_VERSION
    track_id: str
    mode: Literal["aligned", "transcribed"]
    language: str
    engine: LyricsEngine
    lines: list[LyricsLine]
    supplied_markup: Optional[list[SuppliedMarkup]] = None
    untranscribed_regions: list[UntranscribedRegion]
    coverage: LyricsCoverage


# --- manifest.json ----------------------------------------------------------


class ApiUsage(BaseModel):
    calls: int
    input_tokens: int
    output_tokens: int
    usd: float
    estimated_usd: Optional[float] = None


class RunMetadata(BaseModel):
    started_at: str
    duration_seconds: float
    tool_version: str
    device: Optional[Literal["cpu", "mps"]] = None
    api_usage: Optional[ApiUsage] = None


class DocumentEntry(BaseModel):
    status: Status
    path: Optional[str] = None
    schema_version: Optional[str] = None
    content_sha256: Optional[str] = None
    config_hash: Optional[str] = None
    run: Optional[RunMetadata] = None
    error: Optional[str] = None


class ManifestDocuments(BaseModel):
    source: DocumentEntry
    audio_features: DocumentEntry
    lyrics: DocumentEntry
    video: DocumentEntry
    structure: DocumentEntry


class StemsState(BaseModel):
    status: Status
    retained: bool
    config_hash: Optional[str] = None
    run: Optional[RunMetadata] = None
    error: Optional[str] = None
    warning: Optional[str] = None


class Manifest(BaseModel):
    schema_version: str = MANIFEST_SCHEMA_VERSION
    track_id: str
    title: str
    artist: Optional[str] = None
    has_video: bool
    created_at: str
    documents: ManifestDocuments
    stems: Optional[StemsState] = None
