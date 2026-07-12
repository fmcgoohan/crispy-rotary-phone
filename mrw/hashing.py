"""SHA-256 helpers. track_id = first 16 hex chars of the source file hash."""

from __future__ import annotations

import hashlib
from pathlib import Path

TRACK_ID_LEN = 16
_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def track_id_from_sha(sha256_hex: str) -> str:
    return sha256_hex[:TRACK_ID_LEN]
