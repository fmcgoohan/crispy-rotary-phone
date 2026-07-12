"""Pydantic models for the analysis documents (M0 scope: source + manifest).

Field order here IS the canonical key order in written documents — keep it in
sync with the hand-drafted schemas in schemas/. Serialization convention:
absent optional fields are omitted (never null) — dump with exclude_none=True.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

SCHEMA_VERSION = "1.0.0"
# 1.1.0: additive — `error` on the manifest stems block (M2).
MANIFEST_SCHEMA_VERSION = "1.1.0"

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
    schema_version: str = SCHEMA_VERSION
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
    schema_version: str = SCHEMA_VERSION
    track_id: str
    duration_seconds: float
    tempo: Tempo
    beats: Beats
    loudness: Loudness
    mix: ChannelFeatures
    stems: StemFeatures
    vocal_activity: VocalActivity


# --- manifest.json ----------------------------------------------------------


class RunMetadata(BaseModel):
    started_at: str
    duration_seconds: float
    tool_version: str
    device: Optional[Literal["cpu", "mps"]] = None


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


class Manifest(BaseModel):
    schema_version: str = MANIFEST_SCHEMA_VERSION
    track_id: str
    title: str
    artist: Optional[str] = None
    has_video: bool
    created_at: str
    documents: ManifestDocuments
    stems: Optional[StemsState] = None
