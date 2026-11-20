"""Stage 1 — ingest: probe, hash, normalize audio, register the track.

source.json is a pure function of (input file bytes, ingest config); all run
metadata goes to manifest.json. Re-ingesting an already-registered file is a
no-op (the M0 contract) that names every option it ignored; `--update`
applies them to the existing track (ORCH-6/ARCH-9/DET-4/VID-10/LYR-13).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from . import TOOL_VERSION, canonical, config, hashing, lrc
from .library import Library
from .models import (
    DocumentEntry,
    LyricsInput,
    Manifest,
    ManifestDocuments,
    Media,
    MediaAudio,
    MediaTags,
    MediaVideo,
    NormalizedAudio,
    RunMetadata,
    SourceDocument,
    SourceFile,
    StemsState,
)



class IngestError(RuntimeError):
    pass


LYRICS_SUFFIXES = (".lrc", ".txt")


@dataclass
class IngestResult:
    track_id: str
    title: str
    has_video: bool
    already_ingested: bool
    # Re-ingest of a registered track: options a plain re-ingest ignored,
    # and what --update changed.
    ignored: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)


# ffprobe only reads headers; anything slower than this is a hung process
# (damaged or pathological input), not a big file.
PROBE_TIMEOUT_SECONDS = 120


def _run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise IngestError(f"{cmd[0]} timed out after {timeout:.0f} s") from e
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise IngestError(f"{cmd[0]} failed: {stderr[:500]}")
    return proc


def probe(path: Path) -> dict:
    import json

    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout=PROBE_TIMEOUT_SECONDS,
    )
    return json.loads(proc.stdout)


def _first_stream(streams: list[dict], codec_type: str) -> dict | None:
    for s in streams:
        if s.get("codec_type") != codec_type:
            continue
        # Embedded cover art shows up as a video stream marked attached_pic.
        if codec_type == "video" and s.get("disposition", {}).get("attached_pic"):
            continue
        return s
    return None


def select_audio_stream(streams: list[dict]) -> tuple[dict | None, int]:
    """(the audio stream ingest analyses, number of audio streams).

    The rule is explicit and recorded (ORCH-13), never left to ffmpeg's
    automatic selection (which ranks by default disposition, then channel
    count): the first audio stream flagged disposition.default, else the
    first audio stream. Both ffmpeg calls decode exactly this stream via
    `-map 0:<index>`, so media.audio describes the stream that was decoded.
    The video stage applies the same rule to find the audio clock origin.
    """
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        return None, 0
    for s in audio:
        if (s.get("disposition") or {}).get("default"):
            return s, len(audio)
    return audio[0], len(audio)


def _decode_hash_and_duration(
    path: Path, sample_rate: int, channels: int, stream_index: int
) -> tuple[str, float]:
    """Hash the decoded PCM (s16le) of audio stream `stream_index` and
    derive the exact decoded duration."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        f"0:{stream_index}",
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    h = hashlib.sha256()
    total = 0
    # stderr goes to a temp file, not a pipe: with both piped and only
    # stdout drained, ffmpeg blocks once it has written ~64 KB of decode
    # errors (damaged media) and this loop waits on it for ever (SEC-5).
    with tempfile.TemporaryFile() as err, subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=err
    ) as proc:
        assert proc.stdout is not None
        while chunk := proc.stdout.read(1 << 20):
            h.update(chunk)
            total += len(chunk)
        proc.wait()
        err.seek(0)
        stderr = err.read().decode("utf-8", "replace")
    if proc.returncode != 0:
        raise IngestError(f"ffmpeg decode failed: {stderr.strip()[:500]}")
    bytes_per_second = sample_rate * channels * 2
    return h.hexdigest(), canonical.round_seconds(total / bytes_per_second)


def _normalize_audio(
    src: Path, dest: Path, sample_rate: int, channels: int, stream_index: int
) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(src),
            "-map",
            f"0:{stream_index}",
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-sample_fmt",
            "s16",
            str(dest),
        ]
    )


def _lyrics_input(lyrics_path: Path, track_dir: Path, staged: Path | None = None) -> LyricsInput:
    """Copy the lyrics into the track dir and describe them. With `staged`,
    the bytes go there instead (the caller moves them to the returned
    `path` later); the description is the same."""
    fmt = "lrc" if lyrics_path.suffix.lower() == ".lrc" else "txt"
    dest = track_dir / f"lyrics_input.{fmt}"
    target = staged or dest
    shutil.copyfile(lyrics_path, target)
    text = lrc.decode_text(target.read_bytes())
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return LyricsInput(
        path=dest.name,
        format=fmt,
        sha256=hashing.sha256_file(target),
        line_count=len(lines),
        has_timestamps=any(lrc.LEADING_TIME_TAGS.match(ln.strip()) for ln in lines),
    )


LYRICS_STAGING = ".lyrics_input.tmp"


def _registered_lyrics_intact(track_dir: Path, source_doc: dict) -> bool:
    """The registered lyrics_input.* still hashes to source.json's sha256
    (False after a crash mid-`--update`, so a re-run repairs it)."""
    li = source_doc.get("lyrics_input")
    if not li:
        return True
    path = track_dir / li["path"]
    return path.is_file() and hashing.sha256_file(path) == li["sha256"]


VIDEO_COPY_TMP = ".source_video.tmp"


def copy_video_atomic(media_path: Path, track_dir: Path, expected_sha: str) -> Path:
    """Copy the video into the track dir as source_video<ext>, atomically.

    The copy used to be written straight to its final name, so an ENOSPC
    or Ctrl-C part-way through left a truncated source_video.* that the
    video stage later preferred over the intact original (ORCH-2/SEC-9).
    Now it goes to a temp name in the track dir (same filesystem) and is
    os.replace'd into place only when complete; the temp file is removed
    on any failure or interruption. The temp copy is hashed and must match
    expected_sha (the ingested file's) before it is moved into place.
    """
    dest = track_dir / f"source_video{media_path.suffix.lower()}"
    tmp = track_dir / VIDEO_COPY_TMP
    try:
        shutil.copyfile(media_path, tmp)
        if hashing.sha256_file(tmp) != expected_sha:
            raise IngestError(
                f"copy of {media_path} does not hash to the ingested video "
                "(file changed during the copy?)"
            )
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def _read_source(track_dir: Path) -> dict | None:
    import json

    try:
        return json.loads((track_dir / "source.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _library_copy_state(track_dir: Path, source_doc: dict) -> str:
    """'ok' when some source_video.* hashes to the ingested video (the video
    stage's own test), 'damaged' when copies exist but none does, 'missing'
    when there is none. A size match alone is not enough (review A-3): a
    same-size copy with rotted bytes must be reported and repaired."""
    expected = source_doc["file"]["sha256"]
    size = source_doc["file"]["size_bytes"]
    copies = [p for p in sorted(track_dir.glob("source_video.*")) if p.is_file()]
    if any(
        p.stat().st_size == size and hashing.sha256_file(p) == expected
        for p in copies
    ):
        return "ok"
    return "damaged" if copies else "missing"


def _lyrics_changed(lyrics_path: Path | None, source_doc: dict | None) -> bool:
    if lyrics_path is None:
        return False
    registered = ((source_doc or {}).get("lyrics_input") or {}).get("sha256")
    return hashing.sha256_file(lyrics_path) != registered


def _ignored_options(
    existing: Manifest,
    track_dir: Path,
    cfg: config.Config,
    lyrics_path: Path | None,
    title: str | None,
) -> list[str]:
    """What a plain re-ingest would leave unapplied — reported, never
    silently dropped (H1)."""
    source_doc = _read_source(track_dir)
    ignored: list[str] = []
    if _lyrics_changed(lyrics_path, source_doc):
        registered = (source_doc or {}).get("lyrics_input")
        ignored.append(
            f"--lyrics {lyrics_path} ("
            + ("differs from the registered lyrics" if registered
               else "no lyrics are registered")
            + ")"
        )
    elif (
        lyrics_path is not None
        and source_doc is not None
        and not _registered_lyrics_intact(track_dir, source_doc)
    ):
        ignored.append(f"--lyrics {lyrics_path} (the registered copy is damaged)")
    if title is not None and title != existing.title:
        ignored.append(f"--title {title!r} (current title {existing.title!r})")
    if existing.has_video and source_doc is not None:
        copy_state = _library_copy_state(track_dir, source_doc)
        if copy_state == "damaged":
            ignored.append("damaged library video copy (would be replaced)")
        elif cfg.ingest.copy_video and copy_state == "missing":
            ignored.append("ingest.copy_video = true (no copy in the library yet)")
    if not (track_dir / "source_audio.flac").is_file():
        ignored.append("missing source_audio.flac (would be regenerated)")
    return ignored


def _regenerate_flac(media_path: Path, track_dir: Path, source_doc: dict) -> None:
    """Rebuild a deleted source_audio.flac and prove it is the same audio:
    its decoded PCM must hash to the recorded audio_stream_sha256."""
    na = source_doc["normalized_audio"]
    sr, ch = na["sample_rate"], na["channels"]
    dest = track_dir / na["path"]
    tmp = track_dir / ".source_audio.tmp.flac"
    try:
        _normalize_audio(media_path, tmp, sr, ch)
        stream_sha, _ = _decode_hash_and_duration(tmp, sr, ch)
        if stream_sha != na["audio_stream_sha256"]:
            raise IngestError(
                f"regenerated {dest.name} does not match the recorded audio "
                "stream hash (different ffmpeg build?) — not replaced"
            )
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _update_existing(
    existing: Manifest,
    media_path: Path,
    file_sha: str,
    library: Library,
    cfg: config.Config,
    lyrics_path: Path | None,
    title: str | None,
) -> IngestResult:
    """`mrw ingest --update`: apply new inputs to a registered track.

    Touches only ingest's own artifacts (source.json, lyrics_input.*,
    source_audio.flac, source_video.*) and the manifest's title and source
    entry; a newly registered lyrics file resets lyrics and structure to
    pending (the mode or the text changed). No other stage's document or
    entry is modified. media_path hashes to the track by construction, so
    recording it as original_path relinks a moved original without
    cascading (inputs_sha256 excludes paths).
    """
    track_id = existing.track_id
    track_dir = library.track_dir(track_id)
    source_doc = _read_source(track_dir)
    if source_doc is None:
        raise IngestError(
            f"track {track_id}: source.json is missing or unreadable — "
            "the track cannot be updated in place"
        )
    updated: list[str] = []

    # Order (review A-4): all fallible work that leaves the registered state
    # alone comes first — flac regeneration, the video copy, staging the new
    # lyrics under a temp name. Only then does the registered state change,
    # in an order where every interruption point is either consistent or
    # detected: lyrics/structure reset to pending, source.json rewritten,
    # the staged lyrics moved into place, the old format removed, and the
    # manifest's source entry updated last. A crash after source.json is
    # rewritten but before the move is caught by the lyrics stage's own
    # input check, and a re-run of --update repairs it
    # (_registered_lyrics_intact).
    if not (track_dir / source_doc["normalized_audio"]["path"]).is_file():
        _regenerate_flac(media_path, track_dir, source_doc)
        updated.append("regenerated source_audio.flac")

    copy_state = (
        _library_copy_state(track_dir, source_doc) if existing.has_video else "n/a"
    )
    if copy_state == "damaged" or (cfg.ingest.copy_video and copy_state == "missing"):
        # A damaged copy is replaced whatever copy_video says: the library
        # chose to keep one, and the video stage would otherwise have to
        # skip it on every run (review A-3).
        dest = copy_video_atomic(media_path, track_dir, file_sha)
        for stray in track_dir.glob("source_video.*"):
            if stray != dest:
                stray.unlink()  # damaged copies under another extension
        updated.append(
            "replaced the damaged library video copy" if copy_state == "damaged"
            else "copied the video into the library"
        )

    staged = track_dir / LYRICS_STAGING
    try:
        lyrics_changed = lyrics_path is not None and (
            _lyrics_changed(lyrics_path, source_doc)
            or not _registered_lyrics_intact(track_dir, source_doc)
        )
        new_lyrics: LyricsInput | None = None
        if lyrics_changed:
            assert lyrics_path is not None
            new_lyrics = _lyrics_input(lyrics_path, track_dir, staged=staged)
            source_doc["lyrics_input"] = new_lyrics.model_dump(mode="json")
            updated.append(
                f"registered lyrics {lyrics_path.name} ({new_lyrics.format})"
            )

            def reset(m: Manifest) -> None:
                m.documents.lyrics = DocumentEntry(status="pending")
                m.documents.structure = DocumentEntry(status="pending")

            library.update_manifest(track_id, reset)

        if source_doc["file"]["original_path"] != str(media_path):
            source_doc["file"]["original_path"] = str(media_path)
            source_doc["file"]["filename"] = media_path.name
            updated.append(f"original_path -> {media_path}")

        source = SourceDocument.model_validate(source_doc)
        source_sha = library.write_document(track_id, "source.json", source)
        if new_lyrics is not None:
            os.replace(staged, track_dir / new_lyrics.path)
            for fmt in ("lrc", "txt"):
                if f"lyrics_input.{fmt}" != new_lyrics.path:
                    (track_dir / f"lyrics_input.{fmt}").unlink(missing_ok=True)
    finally:
        staged.unlink(missing_ok=True)

    ingest_hash = config.stage_hash(cfg.ingest)
    new_title = title if title is not None and title != existing.title else None
    if new_title is not None:
        updated.append(f"title -> {new_title!r}")
    if existing.documents.source.config_hash != ingest_hash:
        updated.append("source config_hash -> current ingest config")

    def apply(m: Manifest) -> None:
        if new_title is not None:
            m.title = new_title
        m.documents.source = m.documents.source.model_copy(
            update={"content_sha256": source_sha, "config_hash": ingest_hash}
        )

    manifest = library.update_manifest(track_id, apply)
    return IngestResult(
        track_id=track_id,
        title=manifest.title,
        has_video=manifest.has_video,
        already_ingested=True,
        updated=updated,
    )


def ingest(
    media_path: Path,
    library: Library,
    cfg: config.Config,
    lyrics_path: Path | None = None,
    title: str | None = None,
    update: bool = False,
) -> IngestResult:
    if lyrics_path is not None and lyrics_path.suffix.lower() not in LYRICS_SUFFIXES:
        raise ValueError(f"lyrics must be .lrc or .txt: {lyrics_path}")
    media_path = media_path.resolve()
    if not media_path.is_file():
        raise IngestError(f"no such file: {media_path}")

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()

    file_sha = hashing.sha256_file(media_path)
    track_id = hashing.track_id_from_sha(file_sha)

    existing = library.read_manifest(track_id)
    if existing is not None and existing.documents.source.status == "ok":
        if update:
            return _update_existing(
                existing, media_path, file_sha, library, cfg, lyrics_path, title
            )
        return IngestResult(
            track_id=track_id,
            title=existing.title,
            has_video=existing.has_video,
            already_ingested=True,
            ignored=_ignored_options(
                existing, library.track_dir(track_id), cfg, lyrics_path, title
            ),
        )

    info = probe(media_path)
    fmt = info.get("format", {})
    streams = info.get("streams", [])
    audio_stream, audio_stream_count = select_audio_stream(streams)
    if audio_stream is None:
        raise IngestError(f"no audio stream in {media_path.name}")
    video_stream = _first_stream(streams, "video")
    has_video = video_stream is not None

    track_dir = library.track_dir(track_id)
    track_dir.mkdir(parents=True, exist_ok=True)

    sr, ch = cfg.ingest.sample_rate, cfg.ingest.channels
    flac_path = track_dir / "source_audio.flac"
    audio_index = int(audio_stream["index"])
    _normalize_audio(media_path, flac_path, sr, ch, audio_index)
    stream_sha, duration = _decode_hash_and_duration(media_path, sr, ch, audio_index)

    tags_raw = fmt.get("tags", {}) or {}
    tags = None
    if tags_raw.get("title") or tags_raw.get("artist"):
        tags = MediaTags(title=tags_raw.get("title"), artist=tags_raw.get("artist"))

    video = None
    if video_stream is not None:
        fps = float(Fraction(video_stream.get("avg_frame_rate", "0/1")))
        fps = round(fps, 3)
        nb_frames = video_stream.get("nb_frames")
        frame_count = (
            int(nb_frames) if nb_frames else int(round(duration * fps)) if fps else 0
        )
        video = MediaVideo(
            codec=video_stream.get("codec_name", "unknown"),
            width=int(video_stream["width"]),
            height=int(video_stream["height"]),
            fps=fps,
            frame_count=frame_count,
        )

    lyrics_input = _lyrics_input(lyrics_path, track_dir) if lyrics_path else None

    # OQ-2 / review 001 R-7: opt-in self-contained library. The copy is an
    # artifact (like stems), not a document — the video stage prefers
    # source_video.* over file.original_path when present.
    if has_video and cfg.ingest.copy_video:
        copy_video_atomic(media_path, track_dir, file_sha)

    source_doc = SourceDocument(
        track_id=track_id,
        file=SourceFile(
            original_path=str(media_path),
            filename=media_path.name,
            size_bytes=media_path.stat().st_size,
            sha256=file_sha,
        ),
        normalized_audio=NormalizedAudio(
            path=flac_path.name,
            sample_rate=sr,
            channels=ch,
            duration_seconds=duration,
            audio_stream_sha256=stream_sha,
        ),
        media=Media(
            container=fmt.get("format_name", "unknown"),
            duration_seconds=canonical.round_seconds(float(fmt.get("duration", duration))),
            tags=tags,
            audio=MediaAudio(
                codec=audio_stream.get("codec_name", "unknown"),
                sample_rate=int(audio_stream.get("sample_rate", sr)),
                channels=int(audio_stream.get("channels", ch)),
                bit_rate=int(audio_stream["bit_rate"])
                if audio_stream.get("bit_rate")
                else None,
                stream_index=audio_index,
                stream_count=audio_stream_count,
            ),
            video=video,
        ),
        lyrics_input=lyrics_input,
    )

    content_sha = library.write_document(track_id, "source.json", source_doc)

    resolved_title = (
        title or (tags.title if tags and tags.title else None) or media_path.stem
    )
    run = RunMetadata(
        started_at=started_at,
        duration_seconds=round(time.monotonic() - t0, 2),
        tool_version=TOOL_VERSION,
    )
    pending = DocumentEntry(status="pending")
    manifest = Manifest(
        track_id=track_id,
        title=resolved_title,
        artist=tags.artist if tags else None,
        has_video=has_video,
        created_at=started_at,
        documents=ManifestDocuments(
            source=DocumentEntry(
                status="ok",
                path="source.json",
                schema_version=source_doc.schema_version,
                content_sha256=content_sha,
                config_hash=config.stage_hash(cfg.ingest),
                run=run,
            ),
            audio_features=pending,
            lyrics=pending,
            video=pending if has_video else DocumentEntry(status="not_applicable"),
            structure=pending,
        ),
        stems=StemsState(status="pending", retained=cfg.stems.retain),
    )
    library.write_manifest(track_id, manifest)

    return IngestResult(
        track_id=track_id,
        title=resolved_title,
        has_video=has_video,
        already_ingested=False,
    )
