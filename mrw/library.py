"""Track directories and manifest I/O.

The manifest is the mutable envelope (run metadata lives there and only
there); analysis documents are written via canonical.write and never carry
volatile data.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from . import canonical, hashing
from .models import Manifest, doc_dump


class Library:
    def __init__(self, root: Path):
        self.root = Path(root)

    def track_dir(self, track_id: str) -> Path:
        return self.root / track_id

    def manifest_path(self, track_id: str) -> Path:
        return self.track_dir(track_id) / "manifest.json"

    def track_ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            p.name for p in self.root.iterdir() if (p / "manifest.json").is_file()
        )

    def read_manifest(self, track_id: str) -> Manifest | None:
        path = self.manifest_path(track_id)
        if not path.is_file():
            return None
        return Manifest.model_validate_json(path.read_text(encoding="utf-8"))

    def write_manifest(self, track_id: str, manifest: Manifest) -> None:
        self.track_dir(track_id).mkdir(parents=True, exist_ok=True)
        canonical.write(self.manifest_path(track_id), doc_dump(manifest))

    def write_document(self, track_id: str, filename: str, document: BaseModel) -> str:
        """Write an analysis document atomically; returns its content sha256."""
        path = self.track_dir(track_id) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        text = canonical.write(path, doc_dump(document))
        return hashing.sha256_bytes(text.encode("utf-8"))
