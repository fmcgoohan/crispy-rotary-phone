"""Frozen pipeline configuration with per-stage subset hashing.

Config comes from mrw.toml (or --config). Each stage's manifest entry records
a hash of only that stage's config subset, so unrelated config edits don't
mark documents stale.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from . import canonical


class IngestConfig(BaseModel, frozen=True):
    sample_rate: int = 44100
    channels: int = 2
    copy_video: bool = False  # OQ-2 / review 001 R-7


class StemsConfig(BaseModel, frozen=True):
    retain: bool = True  # OQ-3
    model: str = "htdemucs"
    device: str = "auto"  # auto | cpu | mps
    # Stem-file byte identity (OQ-13) is guaranteed at cpu_threads = 1;
    # raise for faster CPU separation when byte identity doesn't matter
    # (review 005, field finding F-2). Validated here, not clamped deep
    # in the wrapper.
    cpu_threads: int = Field(default=1, ge=1)


class FeaturesConfig(BaseModel, frozen=True):
    # Vocal-activity hysteresis on the vocal stem's RMS envelope (OQ-5);
    # the values are embedded in the document's vocal_activity.params (R-5).
    vocal_enter_db: float = -35.0
    vocal_exit_db: float = -45.0
    vocal_min_region_seconds: float = 0.3
    vocal_min_gap_seconds: float = 0.2
    # Downbeat phase-fit onset-sampling half-window (review 006 F-1): a
    # tuning knob, so it participates in the features config_hash.
    phase_fit_window_seconds: float = Field(default=0.03, gt=0)


class LyricsConfig(BaseModel, frozen=True):
    # OQ-7: faster-whisper on the isolated vocal stem; `small` default.
    model: str = "small"
    # Optional language pin; None = detect once and record (schema).
    language: str | None = None
    # Flag thresholds (schema-documented vocabulary; review 006 F-1 policy:
    # output-affecting tunables are config so they join the config_hash).
    confidence_threshold: float = 0.5  # low_confidence: mean word conf below
    long_word_seconds: float = 2.5  # long_word_duration (melisma)
    # Renamed from no_speech_threshold (PR #10 nit): flag_* thresholds
    # feed line flags; decode_* settings feed the engine.
    flag_no_speech_threshold: float = 0.5  # possibly_non_lexical (transcribed)
    compression_ratio_threshold: float = 2.4  # possibly_non_lexical
    overlap_rms_db: float = -15.0  # overlapping_vocals: sustained loud stem
    min_anchor_score: float = 0.6  # aligned mode: fuzzy-anchor acceptance
    # Review 007 field finding 3: minimum uncovered-span length emitted as
    # an untranscribed region.
    uncovered_min_seconds: float = 1.0
    # Degenerate-case convention (H2): a track with zero vocal-activity
    # regions skips language detection (meaningless and numerically
    # knife-edge on silence) and records this fallback instead.
    fallback_language: str = "en"
    # Review 007 finding 4 (closed): full-file transcription feeds quiet
    # verses in badly-conditioned 30 s windows mixed with near-silence —
    # an 18 s clean verse decoded to nothing/garbage while the identical
    # audio decoded perfectly in isolation. Clip transcription to padded,
    # merged vocal-activity windows instead (features is a prerequisite).
    clip_to_vocal_activity: bool = True
    # ge=0: a negative pad would invert a clip window (PR #11 nit).
    clip_padding_seconds: float = Field(default=0.5, ge=0)

    # Review 007 field finding 4 investigation knob: whisper's no-speech
    # gate, exposed so decode-setting changes are config (and hash) events.
    decode_no_speech_threshold: float = 0.6


class VideoConfig(BaseModel, frozen=True):
    # OQ-8: PySceneDetect ContentDetector.
    detector_threshold: float = 27.0
    min_shot_seconds: float = Field(default=0.4, gt=0)
    # Captions (OQ-10): "null" = offline no-op default, caption absent.
    # Validated at the config boundary (PR #12 round 4 nit) — a typo'd
    # backend fails at load, not deep inside the stage after decode work.
    caption_backend: Literal["null", "anthropic"] = "null"
    caption_model: str = "claude-haiku-4-5"  # pinned; priced in mrw/costs.py
    caption_prompt_version: str = "v1"
    caption_max_output_tokens: int = Field(default=300, ge=1)
    # Budget gate (M5 addendum): abort before spending if the pre-flight
    # estimate exceeds this; batch-safe (no interactive prompt).
    caption_budget_usd_per_run: float = Field(default=0.50, ge=0)


class Config(BaseModel, frozen=True):
    ingest: IngestConfig = IngestConfig()
    stems: StemsConfig = StemsConfig()
    features: FeaturesConfig = FeaturesConfig()
    lyrics: LyricsConfig = LyricsConfig()
    video: VideoConfig = VideoConfig()
    # Later milestones add: structure section.


# Which config subset governs each pipeline document / artifact.
STAGE_SECTION: dict[str, str] = {
    "source": "ingest",
    "stems": "stems",
    "audio_features": "features",
    "lyrics": "lyrics",
    "video": "video",
}


def load(path: Path | None = None) -> Config:
    """Load mrw.toml if present (cwd default), else built-in defaults."""
    candidate = path or Path("mrw.toml")
    if candidate.is_file():
        with open(candidate, "rb") as f:
            return Config.model_validate(tomllib.load(f))
    if path is not None:
        raise FileNotFoundError(f"config file not found: {path}")
    return Config()


def stage_hash(section: BaseModel) -> str:
    """16-hex-char hash of one stage's config subset (canonical JSON)."""
    payload = canonical.dumps(section.model_dump(mode="json"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
