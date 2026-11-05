"""Track directories and manifest I/O.

The manifest is the mutable envelope (run metadata lives there and only
there); analysis documents are written via canonical.write and never carry
volatile data.
"""

from __future__ import annotations

import contextlib
import fcntl
import re
from pathlib import Path
from typing import Callable, Iterator

from pydantic import BaseModel

from . import canonical, hashing
from .models import Manifest, doc_dump


# Which documents consume each upstream artifact (PLAN §3 stage graph).
# structure (M6) consumes features, lyrics and video.
DEPENDENTS: dict[str, tuple[str, ...]] = {
    "stems": ("audio_features", "lyrics"),
    "audio_features": ("lyrics", "structure"),
    "lyrics": ("structure",),
    "video": ("structure",),
}


def mark_dependents_stale(manifest: Manifest, stage: str) -> list[str]:
    """After `stage` produced new output, flip every `ok` document derived
    from it (transitively) to `stale`, in the same manifest write.

    Freshness used to be keyed on each stage's own config only, so
    re-running stems or features left lyrics.json "up to date" although it
    was computed from inputs that no longer exist (DET-1). A stale entry
    fails its stage's no-op check (so it re-runs) and is refused as a
    prerequisite downstream (H3). Returns the names flipped.
    """
    flipped: list[str] = []
    queue = list(DEPENDENTS.get(stage, ()))
    while queue:
        name = queue.pop(0)
        entry = getattr(manifest.documents, name)
        if entry.status == "ok":
            setattr(manifest.documents, name, entry.model_copy(update={"status": "stale"}))
            flipped.append(name)
            queue.extend(DEPENDENTS.get(name, ()))
    return flipped


class PrerequisiteError(RuntimeError):
    """Bad invocation / missing prerequisites — nothing recorded; exit 2."""


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

    @contextlib.contextmanager
    def _manifest_lock(self, track_id: str) -> Iterator[None]:
        lock_path = self.track_dir(track_id) / ".manifest.lock"
        with open(lock_path, "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def update_manifest(
        self, track_id: str, mutate: Callable[[Manifest], None]
    ) -> Manifest:
        """Apply one stage's change to the manifest ON DISK, atomically.

        Stages used to read the manifest at start and rewrite the whole file
        at the end, so two commands on one track (stems in one terminal,
        video in another) silently undid each other's entries, API spend
        included (ORCH-4). This re-reads under an exclusive flock, applies
        only `mutate`, and writes before releasing. flock is advisory and
        reliable on local filesystems, which is where a library lives;
        fcntl is POSIX-only, matching the macOS target. The empty
        `.manifest.lock` file stays in the track dir; nothing enumerates or
        hashes track-dir contents (only manifest.json and named documents).
        """
        with self._manifest_lock(track_id):
            manifest = self.read_manifest(track_id)
            if manifest is None:
                raise PrerequisiteError(f"no manifest for track {track_id}")
            mutate(manifest)
            self.write_manifest(track_id, manifest)
            return manifest

    def write_document(self, track_id: str, filename: str, document: BaseModel) -> str:
        """Write an analysis document atomically; returns its content sha256."""
        path = self.track_dir(track_id) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        text = canonical.write(path, doc_dump(document))
        return hashing.sha256_bytes(text.encode("utf-8"))

    def resolve_track_id(self, track: str) -> str:
        """Accept a track_id or a media path (re-resolved by content hash)."""
        if re.fullmatch(r"[0-9a-f]{16}", track):
            if self.manifest_path(track).is_file():
                return track
            raise PrerequisiteError(f"no such track in library: {track}")
        path = Path(track)
        if path.is_file():
            track_id = hashing.track_id_from_sha(hashing.sha256_file(path))
            if self.manifest_path(track_id).is_file():
                return track_id
            raise PrerequisiteError(
                f"{path.name} (track {track_id}) is not ingested — "
                "run `mrw ingest` first"
            )
        raise PrerequisiteError(f"not a track_id or existing file: {track}")
