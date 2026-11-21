"""Consumed-input identities (DET-1 fuller form, DET-3/ORCH-2/VID-5).

Each analysis stage records `inputs_sha256` on its manifest entry: a short
hash over the identities of exactly what it consumed. A stage's no-op check
and `mrw status` compare the stored value with the identities current now,
so a document is never reported up to date after an upstream input changed —
whichever command changed it, and even if the `stale` flip written by
mark_dependents_stale was lost or never happened.

Every per-track identity is readable from manifest.json + source.json
alone, so `mrw status` can derive staleness without hashing media. The one
exception is configuration-wide, not per-track: a local Whisper model
directory, which a stage run or `mrw status` fingerprints once and passes in
(a pinned model's identity is already its config value):

- audio_features ← stems content_sha256 + source audio_stream_sha256
- lyrics ← audio_features content_sha256 + stems content_sha256 +
  source lyrics_input.sha256 (absent when transcribing) + the Whisper
  model fingerprint when lyrics.model is a local directory
  (whisper_models.local_model_fingerprint; joins only then, so entries
  made with a pinned model keep their value)
- video ← source file.sha256 (the stage also verifies the bytes it opens
  against it before analysing them)

Path-only facts (original_path, filename) are deliberately excluded, so
relinking a moved original never cascades.

Compatibility: an entry without inputs_sha256 (written before manifest
1.5.0) counts as matching — existing libraries see no churn on upgrade; its
next real run records the value. An identity that is itself unknown (e.g. a
stems block from before content_sha256 existed) hashes as the empty string,
so it changes — and the dependent re-runs — once the upstream records one.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from .models import Manifest

INPUTS_HASH_LEN = 16


def stems_content_sha256(file_digests: list[str]) -> str:
    """sha256 over the stem files' sha256 hex digests concatenated in
    STEM_NAMES order (the stems block's content_sha256)."""
    return hashlib.sha256("".join(file_digests).encode("ascii")).hexdigest()


def stems_dir_sha256(stems_dir: Path, names: tuple[str, ...]) -> str:
    from . import hashing

    return stems_content_sha256(
        [hashing.sha256_file(stems_dir / f"{n}.flac") for n in names]
    )


def _hash(parts: list[tuple[str, Optional[str]]]) -> str:
    text = "\n".join(f"{name}={value or ''}" for name, value in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:INPUTS_HASH_LEN]


def current_inputs(
    stage: str,
    manifest: Manifest,
    source_doc: dict,
    *,
    model_fingerprint: Optional[str] = None,
) -> Optional[str]:
    """The inputs_sha256 `stage` would record if it ran now (None for stages
    without tracked inputs). `model_fingerprint`: the local Whisper model
    directory's fingerprint (lyrics only; None for a pinned model)."""
    stems_sha = manifest.stems.content_sha256 if manifest.stems else None
    if stage == "audio_features":
        return _hash([
            ("stems", stems_sha),
            ("audio_stream", source_doc["normalized_audio"]["audio_stream_sha256"]),
        ])
    if stage == "lyrics":
        lyrics_input = source_doc.get("lyrics_input") or {}
        parts = [
            ("audio_features", manifest.documents.audio_features.content_sha256),
            ("stems", stems_sha),
            ("lyrics_input", lyrics_input.get("sha256")),
        ]
        if model_fingerprint is not None:
            parts.append(("whisper_model", model_fingerprint))
        return _hash(parts)
    if stage == "video":
        return _hash([("source_file", source_doc["file"]["sha256"])])
    return None


def inputs_match(stored: Optional[str], current: Optional[str]) -> bool:
    """Absent stored value = recorded before 1.5.0 → treated as matching."""
    return stored is None or current is None or stored == current
