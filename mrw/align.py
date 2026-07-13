"""Pure lyric-line alignment: (supplied lines, word stream) → anchored lines.

No Whisper, no I/O — the interface review 006's architecture requirement
asked for, so alignment logic gets fast deterministic unit tests with
synthetic transcripts (OQ-6). The word stream is whatever observed
(text, start, end, probability) sequence the caller provides.

Algorithm: sequential fuzzy anchoring. For each supplied line (in file
order), scan a bounded window of the word stream from the current cursor
(or around the line's `.lrc` hint when present) for the contiguous word
span whose normalized text best matches the line under
rapidfuzz.token_sort_ratio. Accepted anchors advance the cursor;
unanchored lines get `unaligned` + `timing_interpolated` flags and timing
from their hint or neighbor interpolation — never fake word-level precision
(empty `words`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import canonical

_PUNCT = re.compile(r"[^\w']+", re.UNICODE)

# Search bounds (structural, ride tool_version): how far past the cursor an
# anchor may land, and the hint window in seconds around an .lrc timestamp.
LOOKAHEAD_WORDS = 400
HINT_BEFORE_SECONDS = 5.0
HINT_AFTER_SECONDS = 20.0


@dataclass
class WordObs:
    """One observed word from any transcription engine."""

    text: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class SuppliedLine:
    text: str
    source_line_index: int
    hint_seconds: float | None = None


@dataclass
class AnchoredLine:
    text: str
    start: float
    end: float
    anchored: bool
    confidence: float | None = None
    flags: list[str] = field(default_factory=list)
    # (text, start, end, confidence) — supplied text, observed timing.
    words: list[tuple[str, float, float, float]] = field(default_factory=list)


def normalize(token: str) -> str:
    return _PUNCT.sub("", token.lower())


def _score(line_tokens: list[str], window_tokens: list[str]) -> float:
    from rapidfuzz import fuzz

    return fuzz.token_sort_ratio(" ".join(line_tokens), " ".join(window_tokens)) / 100.0


def _pair_words(
    supplied: list[str], window: list[WordObs]
) -> list[tuple[str, float, float, float]]:
    """Map supplied words onto observed timings positionally; when counts
    differ, indices are spread linearly. Confidence is the per-word fuzzy
    match (aligned mode's schema semantics: 'token match score')."""
    from rapidfuzz import fuzz

    out = []
    n_sup, n_obs = len(supplied), len(window)
    for i, word in enumerate(supplied):
        j = 0 if n_sup == 1 else round(i * (n_obs - 1) / (n_sup - 1))
        obs = window[j]
        conf = fuzz.ratio(normalize(word), normalize(obs.text)) / 100.0
        out.append((word, obs.start, obs.end, canonical.round_ratio(conf)))
    return out


def align_lines(
    lines: list[SuppliedLine],
    words: list[WordObs],
    min_anchor_score: float = 0.6,
) -> list[AnchoredLine]:
    norm_words = [normalize(w.text) for w in words]
    results: list[AnchoredLine] = []
    cursor = 0

    for line in lines:
        supplied_words = [t for t in line.text.split() if normalize(t)]
        tokens = [normalize(t) for t in supplied_words]
        best = None  # (score, start, length)

        if tokens and words:
            if line.hint_seconds is not None:
                lo_t = line.hint_seconds - HINT_BEFORE_SECONDS
                hi_t = line.hint_seconds + HINT_AFTER_SECONDS
                candidates = [
                    i for i, w in enumerate(words) if lo_t <= w.start <= hi_t
                ]
                span = (
                    (candidates[0], candidates[-1] + 1) if candidates else (0, 0)
                )
            else:
                span = (cursor, min(len(words), cursor + LOOKAHEAD_WORDS))

            k = len(tokens)
            for start in range(span[0], span[1]):
                for length in range(max(1, k - 2), k + 4):
                    if start + length > len(words):
                        break
                    score = _score(tokens, norm_words[start : start + length])
                    if best is None or score > best[0]:
                        best = (score, start, length)

        if best is not None and best[0] >= min_anchor_score:
            _, start, length = best
            window = words[start : start + length]
            paired = _pair_words(supplied_words, window)
            confidence = canonical.round_ratio(
                sum(p[3] for p in paired) / len(paired)
            )
            results.append(
                AnchoredLine(
                    text=line.text,
                    start=window[0].start,
                    end=window[-1].end,
                    anchored=True,
                    confidence=confidence,
                    flags=[],
                    words=paired,
                )
            )
            cursor = start + length
        else:
            results.append(
                AnchoredLine(
                    text=line.text,
                    start=line.hint_seconds if line.hint_seconds is not None else -1.0,
                    end=-1.0,
                    anchored=False,
                    confidence=None,
                    flags=["low_confidence", "unaligned", "timing_interpolated"],
                    words=[],
                )
            )

    _interpolate_unanchored(results)
    return results


def _interpolate_unanchored(results: list[AnchoredLine]) -> None:
    """Give unanchored lines neighbor-interpolated timing (schema: 'timing
    absent-quality, taken from lrc hint or neighbor interpolation')."""
    n = len(results)
    for i, line in enumerate(results):
        if line.anchored:
            continue
        prev_end = next(
            (results[j].end for j in range(i - 1, -1, -1) if results[j].anchored),
            0.0,
        )
        next_start = next(
            (results[j].start for j in range(i + 1, n) if results[j].anchored),
            None,
        )
        if line.start >= 0.0:  # had an .lrc hint — keep it as the start
            start = line.start
        else:
            start = prev_end
        if next_start is not None and next_start > start:
            end = min(next_start, start + 10.0)
        else:
            end = start + 4.0  # bounded guess; words stay empty regardless
        line.start, line.end = start, end
