"""Stage 4 — lyrics per PLAN §4 stage 4; contract: schemas/lyrics.schema.json
v1.2.0.

Mode `aligned` when the track was ingested with a lyrics file, else
`transcribed`. Transcription: faster-whisper (`small`, int8, CPU) on the
ISOLATED vocal stem (OQ-7), deterministic decode (greedy, temperature 0);
language detection runs on a window of the earliest vocal-activity audio,
never a silent stem head (review 007 finding 1), with provenance in
engine.language_source. Alignment: global monotone rapidfuzz alignment of
supplied lines to the Whisper word stream (OQ-6, mrw/align.py — pure logic,
unit-tested without Whisper); `.lrc` timestamps are offset-corrected cost
hints, never trusted outright. Supplied markup handled per R-3.

Honesty machinery: the seven schema line flags with thresholds from
LyricsConfig; `untranscribed_regions` = uncovered spans — word-covered
intervals subtracted from vocal-activity regions, spans ≥
uncovered_min_seconds emitted (007 finding 3); both coverage ratios.
Unreadable passages surface as flagged or untranscribed — never silently
dropped, never fake-precise.

Transcription is clipped to padded, merged vocal-activity windows
(clip_to_vocal_activity, default on — review 007 finding 4): full-file 30 s
windows that mix near-silence with quiet singing decode to nothing, while
the same audio decodes cleanly in well-conditioned windows.

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

from . import TOOL_VERSION, canonical, config, hashing, inputs, lrc, versions
from .align import AnchoredLine, SuppliedLine, WordObs, align
from .library import Library, PrerequisiteError, failure_message, mark_dependents_stale
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


def clips_from_regions(
    regions: list[tuple[float, float]], padding: float
) -> list[tuple[float, float]]:
    """Padded, merged vocal-activity windows for clipped transcription
    (pure; review 007 finding 4)."""
    clips: list[list[float]] = []
    for s, e in sorted(regions):
        s, e = max(0.0, s - padding), e + padding
        if clips and s <= clips[-1][1]:
            clips[-1][1] = max(clips[-1][1], e)
        else:
            clips.append([s, e])
    return [(s, e) for s, e in clips]


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
        resolve_whisper_model(cfg.model),
        device="cpu", compute_type="int8", cpu_threads=1, num_workers=1,
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

    if cfg.clip_to_vocal_activity and vocal_regions:
        # Review 007 finding 4: align decode windows to actual vocal
        # content. Full-file 30 s windows that mix near-silence with a
        # quiet verse decode to nothing/garbage; the same audio decodes
        # cleanly in a well-conditioned window. Timestamps stay
        # file-relative.
        clips = clips_from_regions(vocal_regions, cfg.clip_padding_seconds)
        decode["clip_timestamps"] = ",".join(
            str(canonical.round_seconds(x)) for pair in clips for x in pair
        )
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


def resolve_whisper_model(model_name: str) -> str:
    """Local directory of a cached Whisper model — never touches the network.

    Passing a size name straight to WhisperModel makes faster-whisper call
    huggingface_hub.snapshot_download(revision="main"), which queries the
    Hub on every run while online and silently pulls a new snapshot if the
    upstream branch moved (LYR-3). CLAUDE.md permits first-use weight
    downloads only, via `mrw models fetch`; transcription runs from the
    cache, at the revision pinned in mrw/whisper_models.py.
    """
    import os

    if os.path.isdir(model_name):
        return model_name
    from faster_whisper.utils import download_model

    _, revision = pinned_whisper_model(model_name)
    try:
        return download_model(model_name, local_files_only=True, revision=revision)
    except FileNotFoundError as e:  # huggingface_hub LocalEntryNotFoundError
        raise LyricsError(
            f"whisper model '{model_name}' (revision {revision[:12]}) is not in "
            "the local model cache — run `mrw models fetch` first "
            "(transcription never downloads)"
        ) from e


def pinned_whisper_model(model_name: str) -> tuple[str, str]:
    """(repo id, pinned commit) for a faster-whisper size name."""
    from .whisper_models import WHISPER_MODELS

    try:
        return WHISPER_MODELS[model_name]
    except KeyError:
        raise LyricsError(
            f"whisper model '{model_name}' has no pinned revision; use one of "
            f"{', '.join(sorted(WHISPER_MODELS))} or a local model directory"
        ) from None


def fetch_whisper_model(model_name: str) -> Path:
    """Download the Whisper model at its pinned revision if absent; returns
    its local directory. A local model directory is returned as is."""
    import os

    if os.path.isdir(model_name):
        return Path(model_name)
    from faster_whisper.utils import download_model

    _, revision = pinned_whisper_model(model_name)
    return Path(download_model(model_name, revision=revision))


def parse_supplied_lyrics(
    raw_text: str,
) -> tuple[list[SuppliedLine], list[SuppliedMarkup]]:
    """Split a supplied .lrc/.txt into singable lines + markup entries (R-3).

    Markup = a bracketed-only line that is not an .lrc timestamp; excluded
    from lines[] and coverage math, label normalized when recognizable.
    LRC ID tags ([ar:], [ti:], [offset:] …) are metadata and are skipped;
    [offset:] shifts every hint. A line carrying several leading time tags
    (compressed LRC) yields one SuppliedLine per tag, sharing its text and
    source_line_index; such files are then ordered by hint so alignment
    sees the timeline in order.
    """
    lines: list[SuppliedLine] = []
    markup: list[SuppliedMarkup] = []
    raw_lines = raw_text.splitlines()
    offset = lrc.offset_seconds(raw_lines)
    expanded = False
    for index, raw in enumerate(raw_lines):
        stripped = raw.strip()
        if not stripped or lrc.ID_TAG.match(stripped):
            continue
        hints: list[float | None] = [
            canonical.round_seconds(max(0.0, t - offset))
            for t in lrc.leading_time_tags(stripped)
        ] or [None]
        text = lrc.TIME_TAG.sub("", stripped).strip()
        if not text:
            continue
        if _MARKUP.fullmatch(text):
            inner = text[1:-1].strip().lower()
            label = next((l for l in _LABELS if inner.startswith(l)), "other")
            # Same rule as sung lines: one entry per leading time tag, so a
            # compressed "[00:45][01:30][Chorus]" marks every occurrence.
            for hint in hints:
                markup.append(
                    SuppliedMarkup(
                        text=text,
                        label=label,
                        source_line_index=index,
                        hint_seconds=hint,
                    )
                )
            continue
        expanded = expanded or len(hints) > 1
        for hint in hints:
            lines.append(
                SuppliedLine(text=text, source_line_index=index, hint_seconds=hint)
            )
    if expanded and all(l.hint_seconds is not None for l in lines):
        lines.sort(key=lambda l: l.hint_seconds)  # stable: file order on ties
    # One rule regardless of expansion: hint order when every entry has a
    # hint (an LRC timeline), else file order. Stable, so ties keep file order.
    if markup and all(m.hint_seconds is not None for m in markup):
        markup.sort(key=lambda m: m.hint_seconds)
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
    # CRIT-5: a zero-length span (a word Whisper timed start == end, or one
    # clamped to (duration, duration)) covers nothing, but it would still
    # split a region's gaps and defeat the whole-region test below — so a
    # short region holding only such a word vanished from the ledger.
    # Drop them before merging; coverage is unchanged (they add 0 s).
    for s, e in sorted(sp for sp in word_spans if sp[1] > sp[0]):
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
    unmatched_words: list[LyricsWord] | None = None,
) -> tuple[LyricsCoverage, list[UntranscribedRegion]]:
    # CRIT-1: heard words no supplied line accounts for still cover their
    # audio — untranscribed means "content unknown", not "not in the file".
    spans = sorted(
        [(w.start_seconds, w.end_seconds) for line in lines for w in line.words]
        + [(w.start_seconds, w.end_seconds) for w in unmatched_words or []]
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
    with library.stage_lock(track_id, "lyrics"):
        return _run_lyrics(track_id, library, cfg)


def _run_lyrics(track: str, library: Library, cfg: config.Config) -> LyricsResult:
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
            f"track {track_id}: audio features missing or "
            f"{manifest.documents.audio_features.status} — run `mrw features "
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
    source_doc = json.loads((track_dir / "source.json").read_text(encoding="utf-8"))
    inputs_sha = inputs.current_inputs("lyrics", manifest, source_doc)
    prior = manifest.documents.lyrics
    if (
        prior.status == "ok"
        and prior.config_hash == lyrics_hash
        and versions.is_current("lyrics", prior.algorithm_version)
        and inputs.inputs_match(prior.inputs_sha256, inputs_sha)
        and (track_dir / "lyrics.json").is_file()
    ):
        return LyricsResult(track_id, "?", -1, -1, already_done=True)

    features_doc = json.loads(
        (track_dir / "audio_features.json").read_text(encoding="utf-8")
    )
    lyrics_input = source_doc.get("lyrics_input")
    mode = "aligned" if lyrics_input else "transcribed"
    if lyrics_input:
        # The registered text must be the one on disk (review A-4): an
        # interrupted `mrw ingest --update` can leave source.json naming
        # lyrics that were never moved into place. Checked before any heavy
        # work; nothing is recorded.
        lyrics_file = track_dir / lyrics_input["path"]
        if not lyrics_file.is_file() or (
            hashing.sha256_file(lyrics_file) != lyrics_input["sha256"]
        ):
            raise PrerequisiteError(
                f"track {track_id}: {lyrics_input['path']} is missing or does "
                "not match the lyrics registered in source.json — re-register "
                "them with `mrw ingest <the media file> --lyrics <lyrics file> "
                "--update`"
            )

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    recorded = False
    # Identity of the document on disk — what a failed entry carries
    # forward (review A-1); the new one once it has been written.
    on_disk_sha = prior.content_sha256

    def _record(entry: DocumentEntry) -> None:
        def apply(m) -> None:
            m.documents.lyrics = entry
            # Keyed on the document on disk changing, not on status: a
            # failed entry carries the on-disk identity forward (review A-1).
            if entry.content_sha256 != prior.content_sha256:
                mark_dependents_stale(m, "lyrics")

        apply(manifest)
        library.update_manifest(track_id, apply)

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
        unmatched_words = None
        lines: list[LyricsLine] = []
        if mode == "aligned":
            raw = lrc.decode_text((track_dir / lyrics_input["path"]).read_bytes())
            supplied_lines, markup_entries = parse_supplied_lyrics(raw)
            supplied_markup = markup_entries or None
            alignment = align(
                supplied_lines,
                all_words,
                cfg.lyrics.min_anchor_score,
                duration=duration,
            )
            unmatched_words = _words_from_obs(
                sorted(alignment.unmatched, key=lambda w: w.start)  # stable
            )
            for a in alignment.lines:
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
                        source_line_index=a.source_line_index,
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
            unmatched_words,
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
            unmatched_words=unmatched_words,
            untranscribed_regions=untranscribed,
            coverage=coverage,
        )
        content_sha = library.write_document(track_id, "lyrics.json", document)
        on_disk_sha = content_sha
        _record(
            DocumentEntry(
                status="ok",
                path="lyrics.json",
                schema_version=document.schema_version,
                content_sha256=content_sha,
                config_hash=lyrics_hash,
                algorithm_version=versions.current("lyrics"),
                inputs_sha256=inputs_sha,
                run=RunMetadata(
                    started_at=started_at,
                    duration_seconds=round(time.monotonic() - t0, 2),
                    tool_version=TOOL_VERSION,
                    device="cpu",
                ),
            )
        )
        recorded = True
        return LyricsResult(
            track_id=track_id,
            mode=mode,
            n_lines=len(lines),
            n_untranscribed=len(untranscribed),
            already_done=False,
        )
    except PrerequisiteError:
        raise
    except BaseException as exc:
        # BaseException (SEC-9): interruptions are recorded as failed too,
        # then propagate unwrapped.
        if recorded:  # ok entry already committed (review A-2)
            raise
        message = failure_message(exc)[:500]
        _record(
            DocumentEntry(
                status="failed",
                # The document on disk is unchanged by the failure, so its
                # identity carries forward (review A-1): dependents built
                # from it stay current, and an identical re-run does not
                # mark them stale.
                content_sha256=on_disk_sha,
                config_hash=lyrics_hash,
                error=message,
            )
        )
        if not isinstance(exc, Exception):
            raise
        raise LyricsError(message) from exc
