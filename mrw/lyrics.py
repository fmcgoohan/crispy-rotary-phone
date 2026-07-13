"""Stage 4 — lyrics per PLAN §4 stage 4; contract: schemas/lyrics.schema.json
v1.0.0 (frozen).

Mode `aligned` when the track was ingested with a lyrics file, else
`transcribed`. Transcription: faster-whisper (`small`, int8, CPU) on the
ISOLATED vocal stem (OQ-7), deterministic decode (greedy, temperature 0).
Alignment: rapidfuzz anchoring of supplied lines to the Whisper word stream
(OQ-6, mrw/align.py — pure logic, unit-tested without Whisper); `.lrc`
timestamps are search-window hints only. Supplied markup handled per R-3.

Honesty machinery: the six schema line flags with thresholds from
LyricsConfig; `untranscribed_regions` = vocal-activity regions with zero
word overlap; both coverage ratios. Unreadable passages surface as flagged
or untranscribed — never silently dropped, never fake-precise.

Determinism (D5): the engine decodes greedily at temperature 0 and this
stage emits documents only, so the precision-contract rounding is the
jitter policy; the double-run test (stubbed engine) is the evidence.
Prerequisites are validated before any heavy import (T5).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import TOOL_VERSION, canonical, config
from .align import AnchoredLine, SuppliedLine, WordObs, align_lines
from .library import Library, PrerequisiteError
from .models import (
    DocumentEntry,
    LyricsCoverage,
    LyricsDocument,
    LyricsEngine,
    LyricsLine,
    LyricsWord,
    RunMetadata,
    SuppliedMarkup,
    UntranscribedRegion,
)

_LRC_TS = re.compile(r"\[(\d{1,3}):(\d{2})(?:\.(\d{1,3}))?\]")
_MARKUP = re.compile(r"^\[[^\]]+\]$")
_LABELS = ("intro", "verse", "chorus", "bridge", "outro")

# Structural (rides tool_version): fraction of a line's span that must sit
# above overlap_rms_db for the overlapping_vocals flag.
_OVERLAP_FRACTION = 0.8
_HOP_SECONDS = 0.01


class LyricsError(RuntimeError):
    """Stage failure — recorded in the manifest as status=failed; exit 1."""


@dataclass
class LyricsResult:
    track_id: str
    mode: str
    n_lines: int
    n_untranscribed: int
    already_done: bool


@dataclass
class Segment:
    """One engine segment: text, span, hallucination stats, words."""

    text: str
    start: float
    end: float
    no_speech_prob: float
    compression_ratio: float
    words: list[WordObs]


def _transcribe(
    vocals_path: Path, cfg: config.LyricsConfig
) -> tuple[str, list[Segment]]:
    """Run faster-whisper on the vocal stem. Isolated so tests stub it."""
    from faster_whisper import WhisperModel

    model = WhisperModel(cfg.model, device="cpu", compute_type="int8")
    segments_iter, info = model.transcribe(
        str(vocals_path),
        word_timestamps=True,
        temperature=0.0,
        beam_size=1,
        best_of=1,
        language=cfg.language,
        condition_on_previous_text=False,
    )
    segments = []
    for seg in segments_iter:
        segments.append(
            Segment(
                text=seg.text.strip(),
                start=float(seg.start),
                end=float(seg.end),
                no_speech_prob=float(seg.no_speech_prob),
                compression_ratio=float(seg.compression_ratio),
                words=[
                    WordObs(
                        text=w.word.strip(),
                        start=float(w.start),
                        end=float(w.end),
                        probability=float(w.probability),
                    )
                    for w in (seg.words or [])
                    if w.word.strip()
                ],
            )
        )
    return info.language, segments


def fetch_whisper_model(model_name: str) -> Path:
    """Download the Whisper model if absent; returns its local directory."""
    from faster_whisper import download_model

    return Path(download_model(model_name))


def parse_supplied_lyrics(
    raw_text: str,
) -> tuple[list[SuppliedLine], list[SuppliedMarkup]]:
    """Split a supplied .lrc/.txt into singable lines + markup entries (R-3).

    Markup = a bracketed-only line that is not an .lrc timestamp; excluded
    from lines[] and coverage math, label normalized when recognizable.
    """
    lines: list[SuppliedLine] = []
    markup: list[SuppliedMarkup] = []
    for index, raw in enumerate(raw_text.splitlines()):
        stripped = raw.strip()
        if not stripped:
            continue
        hint = None
        match = _LRC_TS.match(stripped)
        if match:
            minutes, seconds = int(match.group(1)), int(match.group(2))
            frac = (match.group(3) or "0").ljust(3, "0")[:3]
            hint = canonical.round_seconds(minutes * 60 + seconds + int(frac) / 1000)
        text = _LRC_TS.sub("", stripped).strip()
        if not text:
            continue
        if _MARKUP.fullmatch(text):
            inner = text[1:-1].strip().lower()
            label = next((l for l in _LABELS if inner.startswith(l)), "other")
            markup.append(
                SuppliedMarkup(
                    text=text,
                    label=label,
                    source_line_index=index,
                    hint_seconds=hint,
                )
            )
            continue
        lines.append(SuppliedLine(text=text, source_line_index=index, hint_seconds=hint))
    return lines, markup


def _vocals_rms_lookup(features_doc: dict):
    values = features_doc["stems"]["vocals"]["rms_db"]["values"]

    def mean_above_fraction(start: float, end: float, threshold_db: float) -> float:
        i0, i1 = int(start / _HOP_SECONDS), max(int(end / _HOP_SECONDS), 0)
        span = values[max(i0, 0) : min(i1 + 1, len(values))]
        if not span:
            return 0.0
        return sum(1 for v in span if v > threshold_db) / len(span)

    return mean_above_fraction


def _line_flags(
    words: list[LyricsWord],
    span: tuple[float, float],
    base_flags: list[str],
    cfg: config.LyricsConfig,
    overlap_fraction_fn,
    segment: Segment | None = None,
) -> list[str]:
    flags = list(base_flags)
    if words:
        mean_conf = sum(w.confidence for w in words) / len(words)
        if mean_conf < cfg.confidence_threshold and "low_confidence" not in flags:
            flags.append("low_confidence")
        if any(w.end_seconds - w.start_seconds > cfg.long_word_seconds for w in words):
            flags.append("long_word_duration")
    if segment is not None and (
        segment.no_speech_prob > cfg.no_speech_threshold
        or segment.compression_ratio > cfg.compression_ratio_threshold
    ):
        flags.append("possibly_non_lexical")
    if overlap_fraction_fn(span[0], span[1], cfg.overlap_rms_db) >= _OVERLAP_FRACTION:
        flags.append("overlapping_vocals")
    return flags


def _coverage_and_untranscribed(
    lines: list[LyricsLine], vocal_regions: list[dict]
) -> tuple[LyricsCoverage, list[UntranscribedRegion]]:
    spans = sorted(
        (w.start_seconds, w.end_seconds) for line in lines for w in line.words
    )
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    def overlap(a0: float, a1: float) -> float:
        return sum(max(0.0, min(a1, e) - max(a0, s)) for s, e in merged)

    total = sum(r["end_seconds"] - r["start_seconds"] for r in vocal_regions)
    covered = sum(
        overlap(r["start_seconds"], r["end_seconds"]) for r in vocal_regions
    )
    untranscribed = [
        UntranscribedRegion(
            start_seconds=r["start_seconds"], end_seconds=r["end_seconds"]
        )
        for r in vocal_regions
        if overlap(r["start_seconds"], r["end_seconds"]) == 0.0
    ]
    # No vocal activity at all → vacuously covered (degenerate-case
    # convention, H2): 1.0, not a division by zero.
    covered_ratio = covered / total if total > 0 else 1.0
    flagged_ratio = (
        sum(1 for line in lines if line.flags) / len(lines) if lines else 0.0
    )
    return (
        LyricsCoverage(
            vocal_activity_covered_ratio=canonical.round_ratio(min(covered_ratio, 1.0)),
            lines_flagged_ratio=canonical.round_ratio(flagged_ratio),
        ),
        untranscribed,
    )


def _words_from_obs(observed: list[WordObs]) -> list[LyricsWord]:
    return [
        LyricsWord(
            text=w.text,
            start_seconds=canonical.round_seconds(w.start),
            end_seconds=canonical.round_seconds(w.end),
            confidence=canonical.round_ratio(w.probability),
        )
        for w in observed
    ]


def _words_from_pairs(pairs) -> list[LyricsWord]:
    return [
        LyricsWord(
            text=text,
            start_seconds=canonical.round_seconds(start),
            end_seconds=canonical.round_seconds(end),
            confidence=canonical.round_ratio(conf),
        )
        for text, start, end, conf in pairs
    ]


def run_lyrics(track: str, library: Library, cfg: config.Config) -> LyricsResult:
    track_id = library.resolve_track_id(track)
    manifest = library.read_manifest(track_id)
    if manifest is None or manifest.documents.source.status != "ok":
        raise PrerequisiteError(f"track {track_id} has no successful ingest")
    track_dir = library.track_dir(track_id)

    # Cheap prerequisite validation before any heavy import (T5).
    if manifest.documents.audio_features.status != "ok" or not (
        track_dir / "audio_features.json"
    ).is_file():
        raise PrerequisiteError(
            f"track {track_id}: audio features missing — run `mrw features "
            f"{track_id}` first (vocal_activity is consumed here)"
        )
    vocals_path = track_dir / "stems" / "vocals.flac"
    stems_state = manifest.stems
    if (
        stems_state is None
        or stems_state.status != "ok"
        or not stems_state.retained
        or not vocals_path.is_file()
    ):
        raise PrerequisiteError(
            f"track {track_id}: vocal stem not on disk — re-run `mrw stems "
            f"{track_id}` with stems.retain = true"
        )

    lyrics_hash = config.stage_hash(cfg.lyrics)
    prior = manifest.documents.lyrics
    if (
        prior.status == "ok"
        and prior.config_hash == lyrics_hash
        and (track_dir / "lyrics.json").is_file()
    ):
        return LyricsResult(track_id, "?", -1, -1, already_done=True)

    source_doc = json.loads((track_dir / "source.json").read_text(encoding="utf-8"))
    features_doc = json.loads(
        (track_dir / "audio_features.json").read_text(encoding="utf-8")
    )
    lyrics_input = source_doc.get("lyrics_input")
    mode = "aligned" if lyrics_input else "transcribed"

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()

    def _record(entry: DocumentEntry) -> None:
        manifest.documents.lyrics = entry
        library.write_manifest(track_id, manifest)

    try:
        language, segments = _transcribe(vocals_path, cfg.lyrics)
        if cfg.lyrics.language:
            language = cfg.lyrics.language
        overlap_fn = _vocals_rms_lookup(features_doc)
        all_words = [w for seg in segments for w in seg.words]

        supplied_markup = None
        lines: list[LyricsLine] = []
        if mode == "aligned":
            raw = (track_dir / lyrics_input["path"]).read_text(
                encoding="utf-8", errors="replace"
            )
            supplied_lines, markup_entries = parse_supplied_lyrics(raw)
            supplied_markup = markup_entries or None
            anchored = align_lines(
                supplied_lines, all_words, cfg.lyrics.min_anchor_score
            )
            for a in anchored:
                words = _words_from_pairs(a.words)
                span = (a.start, a.end)
                lines.append(
                    LyricsLine(
                        text=a.text,
                        start_seconds=canonical.round_seconds(a.start),
                        end_seconds=canonical.round_seconds(a.end),
                        confidence=a.confidence,
                        flags=_line_flags(
                            words, span, a.flags, cfg.lyrics, overlap_fn
                        ),
                        words=words,
                    )
                )
        else:
            for seg in segments:
                if not seg.text:
                    continue
                words = _words_from_obs(seg.words)
                mean_conf = (
                    sum(w.confidence for w in words) / len(words) if words else None
                )
                lines.append(
                    LyricsLine(
                        text=seg.text,
                        start_seconds=canonical.round_seconds(seg.start),
                        end_seconds=canonical.round_seconds(seg.end),
                        confidence=(
                            canonical.round_ratio(mean_conf)
                            if mean_conf is not None
                            else None
                        ),
                        flags=_line_flags(
                            words,
                            (seg.start, seg.end),
                            [],
                            cfg.lyrics,
                            overlap_fn,
                            segment=seg,
                        ),
                        words=words,
                    )
                )

        coverage, untranscribed = _coverage_and_untranscribed(
            lines, features_doc["vocal_activity"]["regions"]
        )
        document = LyricsDocument(
            track_id=track_id,
            mode=mode,
            language=language,
            engine=LyricsEngine(name="faster-whisper", model=cfg.lyrics.model),
            lines=lines,
            supplied_markup=supplied_markup,
            untranscribed_regions=untranscribed,
            coverage=coverage,
        )
        content_sha = library.write_document(track_id, "lyrics.json", document)
        _record(
            DocumentEntry(
                status="ok",
                path="lyrics.json",
                schema_version=document.schema_version,
                content_sha256=content_sha,
                config_hash=lyrics_hash,
                run=RunMetadata(
                    started_at=started_at,
                    duration_seconds=round(time.monotonic() - t0, 2),
                    tool_version=TOOL_VERSION,
                    device="cpu",
                ),
            )
        )
        return LyricsResult(
            track_id=track_id,
            mode=mode,
            n_lines=len(lines),
            n_untranscribed=len(untranscribed),
            already_done=False,
        )
    except PrerequisiteError:
        raise
    except Exception as exc:
        _record(
            DocumentEntry(
                status="failed", config_hash=lyrics_hash, error=str(exc)[:500]
            )
        )
        raise LyricsError(str(exc)[:500]) from exc
