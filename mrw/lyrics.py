"""Stage 4 — lyrics per PLAN §4 stage 4; contract: schemas/lyrics.schema.json
v1.1.0.

Mode `aligned` when the track was ingested with a lyrics file, else
`transcribed`. Transcription: faster-whisper (`small`, int8, CPU) on the
ISOLATED vocal stem (OQ-7), deterministic decode (greedy, temperature 0);
language detection runs on a window of the earliest vocal-activity audio,
never a silent stem head (review 007 finding 1), with provenance in
engine.language_source. Alignment: rapidfuzz anchoring of supplied lines to
the Whisper word stream (OQ-6, mrw/align.py — pure logic, unit-tested
without Whisper); `.lrc` timestamps are search-window hints only. Supplied
markup handled per R-3.

Honesty machinery: the seven schema line flags with thresholds from
LyricsConfig; `untranscribed_regions` = uncovered spans — word-covered
intervals subtracted from vocal-activity regions, spans ≥
uncovered_min_seconds emitted (007 finding 3); both coverage ratios.
Unreadable passages surface as flagged or untranscribed — never silently
dropped, never fake-precise.

Determinism (D5): the engine decodes greedily at temperature 0 and this
stage emits documents only, so the precision-contract rounding is the
jitter policy; double-run tests cover both the stubbed and the real engine
path. Prerequisites are validated before any heavy import (T5).
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


def vocal_window_slices(
    regions: list[tuple[float, float]], max_seconds: float = 30.0
) -> list[tuple[float, float]]:
    """Earliest-first vocal-activity slices totalling ≤ max_seconds (pure).

    Review 007 field finding 1: language detection on a silent stem head
    locks onto noise-attractor languages (observed: Welsh) — detect on
    audio that audio_features says is actually vocal.
    """
    out: list[tuple[float, float]] = []
    total = 0.0
    for start, end in sorted(regions):
        if total >= max_seconds:
            break
        take = min(end - start, max_seconds - total)
        if take > 0:
            out.append((start, start + take))
            total += take
    return out


def _detection_window_audio(vocals_path: Path, regions: list[tuple[float, float]]):
    """16 kHz mono window assembled from the earliest vocal-activity
    regions; the file head when no regions exist."""
    import librosa
    import numpy as np
    import soundfile as sf

    data, sample_rate = sf.read(vocals_path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    slices = vocal_window_slices(regions)
    if slices:
        parts = [
            mono[int(s * sample_rate) : int(e * sample_rate)] for s, e in slices
        ]
        window = np.concatenate(parts)
    else:
        window = mono[: 30 * sample_rate]
    return librosa.resample(window, orig_sr=sample_rate, target_sr=16000)


def _transcribe(
    vocals_path: Path,
    cfg: config.LyricsConfig,
    vocal_regions: list[tuple[float, float]],
) -> tuple[str, str, list[Segment]]:
    """Run faster-whisper on the vocal stem. Isolated so tests stub it.

    Returns (language, language_source, segments).
    """
    from faster_whisper import WhisperModel

    # D5 (PR #9 review): pin threads like the stems precedent — documents
    # get the rounding layer, but near-threshold decodes on quiet audio can
    # flip with thread order; one thread keeps same-machine runs stable.
    model = WhisperModel(
        cfg.model, device="cpu", compute_type="int8", cpu_threads=1, num_workers=1
    )
    decode = dict(
        word_timestamps=True,
        temperature=0.0,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
        no_speech_threshold=cfg.decode_no_speech_threshold,
    )
    language = cfg.language
    language_source = "pinned"
    if language is None:
        if not vocal_regions:
            # H2 degenerate convention (PR #10 review): with zero vocal
            # activity, detection would run on silence — meaningless, and
            # numerically knife-edge (near-tied language probabilities can
            # flip on last-ulp differences between runs). Record the
            # configured fallback with its own provenance instead.
            language = cfg.fallback_language
            language_source = "default_no_vocal_activity"
        else:
            # Review 007 field finding 1: never detect on the (possibly
            # silent) stem head — use the earliest vocal-activity audio.
            language_source = "detected_vocal_window"
            window = _detection_window_audio(vocals_path, vocal_regions)
            _, info = model.transcribe(window, language=None, **decode)
            language = info.language

    segments_iter, info = model.transcribe(
        str(vocals_path), language=language, **decode
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
    return language, language_source, segments


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
    series = features_doc["stems"]["vocals"]["rms_db"]
    values = series["values"]
    # S4 (PR #9 review): the hop comes from the document's own embedded
    # hop_seconds — never a parallel constant that could drift from it.
    hop = series["hop_seconds"]

    def mean_above_fraction(start: float, end: float, threshold_db: float) -> float:
        i0, i1 = int(start / hop), max(int(end / hop), 0)
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
    vocal_regions: list[tuple[float, float]],
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
        segment.no_speech_prob > cfg.flag_no_speech_threshold
        or segment.compression_ratio > cfg.compression_ratio_threshold
    ):
        flags.append("possibly_non_lexical")
    if overlap_fraction_fn(span[0], span[1], cfg.overlap_rms_db) >= _OVERLAP_FRACTION:
        flags.append("overlapping_vocals")
    # Review 007 field finding 2: a line whose span overlaps no vocal-
    # activity region is likely hallucinated over instrumental audio —
    # flagged, never dropped.
    if not any(
        min(span[1], r_end) > max(span[0], r_start)
        for r_start, r_end in vocal_regions
    ):
        flags.append("outside_vocal_activity")
    return flags


def uncovered_spans(
    word_spans: list[tuple[float, float]],
    regions: list[tuple[float, float]],
    min_seconds: float,
) -> list[tuple[float, float]]:
    """Vocal-activity minus word coverage, spans ≥ min_seconds (pure).

    Review 007 field finding 3: region granularity hid an 18 s missed
    verse inside a partially-covered merged region — subtract coverage
    instead of testing regions whole.
    """
    merged: list[list[float]] = []
    for s, e in sorted(word_spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out: list[tuple[float, float]] = []
    for r_start, r_end in sorted(regions):
        gaps: list[tuple[float, float]] = []
        pos = r_start
        for s, e in merged:
            if e <= pos or s >= r_end:
                continue
            if s > pos:
                gaps.append((pos, min(s, r_end)))
            pos = max(pos, e)
        if pos < r_end:
            gaps.append((pos, r_end))
        # PR #10 review [major S2]: a WHOLE zero-coverage region is always
        # emitted regardless of length — exactly the 1.0.0 guarantee, which
        # keeps this a strict superset (additive minor). min_seconds
        # filters only fragments of partially-covered regions.
        whole_region_uncovered = len(gaps) == 1 and gaps[0] == (r_start, r_end)
        for s, e in gaps:
            if whole_region_uncovered or e - s >= min_seconds:
                out.append((s, e))
    return out


def _coverage_and_untranscribed(
    lines: list[LyricsLine],
    vocal_regions: list[dict],
    min_uncovered_seconds: float,
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
    region_tuples = [(r["start_seconds"], r["end_seconds"]) for r in vocal_regions]
    untranscribed = [
        UntranscribedRegion(
            start_seconds=canonical.round_seconds(s),
            end_seconds=canonical.round_seconds(e),
        )
        for s, e in uncovered_spans(
            [(m[0], m[1]) for m in merged], region_tuples, min_uncovered_seconds
        )
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
        region_tuples = [
            (r["start_seconds"], r["end_seconds"])
            for r in features_doc["vocal_activity"]["regions"]
        ]
        language, language_source, segments = _transcribe(
            vocals_path, cfg.lyrics, region_tuples
        )
        if cfg.lyrics.language:
            language = cfg.lyrics.language
        # Whisper can hallucinate timestamps past the end of quiet audio
        # (observed on CI: end 21.98 on a 15 s clip). Times outside the
        # unified timeline domain are clamped to [0, duration]; content
        # honesty is the flags' job, timeline validity is ours (PR #9).
        duration = float(features_doc["duration_seconds"])
        for seg in segments:
            seg.start = min(max(seg.start, 0.0), duration)
            seg.end = min(max(seg.end, seg.start), duration)
            for w in seg.words:
                w.start = min(max(w.start, 0.0), duration)
                w.end = min(max(w.end, w.start), duration)
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
                            words, span, a.flags, cfg.lyrics, overlap_fn,
                            region_tuples,
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
                            region_tuples,
                            segment=seg,
                        ),
                        words=words,
                    )
                )

        coverage, untranscribed = _coverage_and_untranscribed(
            lines,
            features_doc["vocal_activity"]["regions"],
            cfg.lyrics.uncovered_min_seconds,
        )
        document = LyricsDocument(
            track_id=track_id,
            mode=mode,
            language=language,
            engine=LyricsEngine(
                name="faster-whisper",
                model=cfg.lyrics.model,
                language_source=language_source,
            ),
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
