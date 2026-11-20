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
from .models import ApiLedgerEntry, Manifest, RunMetadata, doc_dump


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


def _ledger_entry(stage: str, status: str, run: RunMetadata) -> ApiLedgerEntry:
    usage = run.api_usage
    assert usage is not None
    return ApiLedgerEntry(
        stage=stage,
        started_at=run.started_at,
        status="failed" if status == "failed" else "ok",
        calls=usage.calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        usd=usage.usd,
        estimated_usd=usage.estimated_usd,
    )


def seed_api_ledger(manifest: Manifest) -> None:
    """Carry pre-1.5.0 spend into the append-only ledger (ORCH-7).

    Before api_ledger existed, the only record of spend was each entry's
    run.api_usage, which the next run of that stage replaces. On the first
    write to such a manifest, copy every existing run.api_usage into
    api_ledger, so the spend survives the replacement. No-op once the
    ledger exists, or when nothing was ever spent (the field stays absent).
    """
    if manifest.api_ledger is not None:
        return
    seeded: list[ApiLedgerEntry] = []
    for name in type(manifest.documents).model_fields:
        entry = getattr(manifest.documents, name)
        if entry.run is not None and entry.run.api_usage is not None:
            seeded.append(_ledger_entry(name, entry.status, entry.run))
    stems = manifest.stems
    if stems is not None and stems.run is not None and stems.run.api_usage is not None:
        seeded.append(_ledger_entry("stems", stems.status, stems.run))
    if seeded:
        manifest.api_ledger = seeded


def append_api_ledger(
    manifest: Manifest, stage: str, status: str, run: RunMetadata | None
) -> None:
    """Append one run's spend to the ledger (only runs that made calls).
    Call inside an update_manifest mutate, i.e. under the manifest lock."""
    if run is None or run.api_usage is None or run.api_usage.calls == 0:
        return
    if manifest.api_ledger is None:
        manifest.api_ledger = []
    manifest.api_ledger.append(_ledger_entry(stage, status, run))


def failure_message(exc: BaseException) -> str:
    """The `error` text for a failed run. Interruptions (KeyboardInterrupt,
    which the CLI also raises for SIGTERM/SIGHUP — SEC-9) carry no message
    of their own, so they are named explicitly."""
    if not isinstance(exc, Exception):
        return f"interrupted ({type(exc).__name__})"
    return str(exc)


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

    @contextlib.contextmanager
    def stage_lock(self, track_id: str, stage: str) -> Iterator[None]:
        """Refuse a second concurrent run of one stage on one track.

        Two `mrw stems T` runs shared the fixed .stems.tmp dir: the second
        deleted the first's output mid-write and then recorded failed over
        a good result (ORCH-4). A non-blocking exclusive flock on
        `<track>/.<stage>.lock` makes the second run exit 2 instead. The
        lock dies with the process (crash, SIGKILL), so no stale-lock
        cleanup is ever needed; the empty file stays, like .manifest.lock.
        """
        lock_path = self.track_dir(track_id) / f".{stage}.lock"
        with open(lock_path, "a") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise PrerequisiteError(
                    f"stage {stage} already running for {track_id}"
                ) from None
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
            seed_api_ledger(manifest)
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
