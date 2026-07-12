"""Frozen pipeline configuration with per-stage subset hashing.

Config comes from mrw.toml (or --config). Each stage's manifest entry records
a hash of only that stage's config subset, so unrelated config edits don't
mark documents stale.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

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


class Config(BaseModel, frozen=True):
    ingest: IngestConfig = IngestConfig()
    stems: StemsConfig = StemsConfig()
    # Later milestones add: features, lyrics, video, structure sections.


# Which config subset governs each pipeline document / artifact.
STAGE_SECTION: dict[str, str] = {
    "source": "ingest",
    "stems": "stems",
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
