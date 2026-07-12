"""Stage 1 — ingest: probe, hash, normalize audio, register the track.

source.json is a pure function of (input file bytes, ingest config); all run
metadata goes to manifest.json. Re-ingesting an already-registered file is a
no-op.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from . import TOOL_VERSION, canonical, config, hashing
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

_LRC_TIMESTAMP = re.compile(r"^\[\d{1,3}:\d{2}")


class IngestError(RuntimeError):
    pass


@dataclass
class IngestResult:
    track_id: str
    title: str
    has_video: bool
    already_ingested: bool


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True)
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
        ]
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


def _decode_hash_and_duration(
    path: Path, sample_rate: int, channels: int
) -> tuple[str, float]:
    """Hash the decoded PCM (s16le) and derive the exact decoded duration."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
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
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        assert proc.stdout is not None
        while chunk := proc.stdout.read(1 << 20):
            h.update(chunk)
            total += len(chunk)
        stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
    if proc.returncode != 0:
        raise IngestError(f"ffmpeg decode failed: {stderr.strip()[:500]}")
    bytes_per_second = sample_rate * channels * 2
    return h.hexdigest(), canonical.round_seconds(total / bytes_per_second)


def _normalize_audio(src: Path, dest: Path, sample_rate: int, channels: int) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(src),
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


def _lyrics_input(lyrics_path: Path, track_dir: Path) -> LyricsInput:
    fmt = "lrc" if lyrics_path.suffix.lower() == ".lrc" else "txt"
    dest = track_dir / f"lyrics_input.{fmt}"
    shutil.copyfile(lyrics_path, dest)
    text = dest.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return LyricsInput(
        path=dest.name,
        format=fmt,
        sha256=hashing.sha256_file(dest),
        line_count=len(lines),
        has_timestamps=any(_LRC_TIMESTAMP.match(ln) for ln in lines),
    )


def ingest(
    media_path: Path,
    library: Library,
    cfg: config.Config,
    lyrics_path: Path | None = None,
    title: str | None = None,
) -> IngestResult:
    media_path = media_path.resolve()
    if not media_path.is_file():
        raise IngestError(f"no such file: {media_path}")

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()

    file_sha = hashing.sha256_file(media_path)
    track_id = hashing.track_id_from_sha(file_sha)

    existing = library.read_manifest(track_id)
    if existing is not None and existing.documents.source.status == "ok":
        return IngestResult(
            track_id=track_id,
            title=existing.title,
            has_video=existing.has_video,
            already_ingested=True,
        )

    info = probe(media_path)
    fmt = info.get("format", {})
    streams = info.get("streams", [])
    audio_stream = _first_stream(streams, "audio")
    if audio_stream is None:
        raise IngestError(f"no audio stream in {media_path.name}")
    video_stream = _first_stream(streams, "video")
    has_video = video_stream is not None

    track_dir = library.track_dir(track_id)
    track_dir.mkdir(parents=True, exist_ok=True)

    sr, ch = cfg.ingest.sample_rate, cfg.ingest.channels
    flac_path = track_dir / "source_audio.flac"
    _normalize_audio(media_path, flac_path, sr, ch)
    stream_sha, duration = _decode_hash_and_duration(media_path, sr, ch)

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
        shutil.copyfile(
            media_path, track_dir / f"source_video{media_path.suffix.lower()}"
        )

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
