"""Pure lyric-line alignment: (supplied lines, word stream) → anchored lines.

No Whisper, no I/O — the interface review 006's architecture requirement
asked for, so alignment logic gets fast deterministic unit tests with
synthetic transcripts (OQ-6). The word stream is whatever observed
(text, start, end, probability) sequence the caller provides.

Algorithm (lyrics algorithm version 4; LYR-1, LYR-2, LYR-8, LYR-11):

1. Line order. Lines are aligned in the order the caller supplies them —
   file order (parse_supplied_lyrics orders an expanded multi-tag `.lrc` by
   hint). When hinted lines are out of hint order, the hint-ordered sequence
   (hinted lines permuted among their own slots; unhinted lines keep their
   positions) is solved too, and taken only when its alignment total is
   strictly higher — i.e. the transcript corroborates the hint order. So a
   malformed, out-of-order `.lrc` still aligns, while one mistyped
   timestamp (whose permutation anchors fewer lines) never moves its line
   away from where it is sung.
2. Global monotone alignment at line granularity — dynamic programming
   over (line i, transcript word j), the Needleman-Wunsch-style alignment
   PLAN §4 describes. Each line either anchors a contiguous word window
   that starts at or after the end of the previous anchored window, or is
   skipped. A transcript word is never used by two lines. Candidate windows
   score ≥ min_anchor_score under rapidfuzz token_sort_ratio (the
   acceptance threshold, unchanged), have no internal inter-word gap over
   MAX_INTERNAL_GAP_SECONDS and last at most MAX_WINDOW_SECONDS. Anchoring
   gains SKIP_PENALTY + score − hint cost; skipping gains 0. So the DP
   first maximises the number of anchored lines, then their total score:
   a cleaner later chorus can no longer capture an earlier chorus's lines
   (LYR-1). Exact ties break toward the earliest window, line by line.
3. `.lrc` hints are a cost term, not a search window (LYR-2): an unhinted
   pass first estimates a global hint offset = median(anchor − hint) over
   confidently anchored lines that are unambiguous (text unique in the file
   and matched strongly in one place in the transcript) (handles
   files synced to a different version of the song); the hinted pass then
   charges HINT_COST_PER_SECOND × |window start − (hint + offset)|, capped
   at HINT_COST_CAP_SECONDS, so a strong match far from a wrong hint can
   still anchor.
4. Inside an anchored window, supplied words are paired with transcript
   words by content (a token-level alignment, fuzz.ratio ≥ WORD_MATCH_RATIO
   = match; two supplied words may merge into one heard word and one
   supplied word may be heard as two) — never by position (LYR-8). Pairing
   may reach PAIRING_MARGIN_WORDS past the window over words no other line
   uses, so an edge word the window score dropped is still observed. Matched
   words take observed timings; unmatched supplied words are interpolated
   between matched neighbours by character share, and the line gets
   `timing_interpolated`. Two supplied words never share one observed span
   without that flag.
5. Transcript words that no anchored line paired are returned as
   `unmatched` (CRIT-1) so the caller can keep them and count them toward
   coverage.
6. Unanchored lines get `unaligned` + `timing_interpolated` and line
   timing only (empty `words` — never fake word-level precision): a hinted
   line starts at its offset-corrected hint; a run of k consecutive
   unhinted lines gets starts evenly spaced over the gap between its
   neighbours, each lasting gap/k capped at UNANCHORED_LINE_CAP_SECONDS
   (the rest of a slot stays unclaimed), or laid out OPEN_ENDED_LINE_SECONDS
   each when nothing follows (less, evenly, when the track ends sooner).
   Zero-anchor convention (H2): with no anchors and no hints, lines are laid
   out in order from 0 s, OPEN_ENDED_LINE_SECONDS each (squeezed evenly
   into the track when it is shorter). Every emitted time is clamped to
   [0, duration] (LYR-11, SCH-3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import canonical

_PUNCT = re.compile(r"[^\w']+", re.UNICODE)

# Tunables classification (review 007 F-1, per the PR #7 convention):
# min_anchor_score is the acceptance threshold and lives in LyricsConfig
# (config_hash member). The constants below are STRUCTURAL — they shape how
# alignment searches and scores, not what it accepts — and ride the lyrics
# ALGORITHM_VERSIONS entry. Promoting any of them to config later is a
# conscious config-evolution event, not a silent edit.
WINDOW_SHORTER_BY = 2  # candidate windows: k-2 .. k+3 words for a k-word line
WINDOW_LONGER_BY = 3
MAX_INTERNAL_GAP_SECONDS = 8.0  # sustained notes yes, a whole break no
MAX_WINDOW_SECONDS = 30.0
SKIP_PENALTY = 1.0  # > any hint cost: anchoring always beats skipping
HINT_COST_PER_SECOND = 0.01
HINT_COST_CAP_SECONDS = 30.0  # cost ≤ 0.3: a far, strong match still wins
OFFSET_MIN_SCORE = 0.8  # anchors trusted for the global hint-offset estimate
WORD_MATCH_RATIO = 0.8
_MERGE_WEIGHT = 0.9  # per supplied word; a 1:1 match is preferred
_SPLIT_WEIGHT = 0.95
PAIRING_MARGIN_WORDS = 2  # pairing may reach this far past the DP window
UNANCHORED_LINE_CAP_SECONDS = 10.0
OPEN_ENDED_LINE_SECONDS = 4.0
_EPS = 1e-9

UNANCHORED_FLAGS = ["low_confidence", "unaligned", "timing_interpolated"]


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
    source_line_index: int | None = None


@dataclass
class Alignment:
    lines: list[AnchoredLine]  # sorted by start
    unmatched: list[WordObs]  # transcript words no line paired, stream order
    hint_offset: float  # seconds added to every .lrc hint (0.0 without hints)


def normalize(token: str) -> str:
    return _PUNCT.sub("", token.lower())


def _ratio(a: str, b: str) -> float:
    from rapidfuzz import fuzz

    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


# --- line order ----------------------------------------------------------------


def _hint_sorted(lines: list[SuppliedLine]) -> list[SuppliedLine]:
    """File order, with hinted lines permuted into hint order among the
    slots hinted lines occupy (stable: file order on equal hints)."""
    slots = [i for i, line in enumerate(lines) if line.hint_seconds is not None]
    by_hint = sorted((lines[i] for i in slots), key=lambda l: l.hint_seconds)
    seq = list(lines)
    for slot, line in zip(slots, by_hint):
        seq[slot] = line
    return seq


# --- window candidates ---------------------------------------------------------


def _candidates(
    seq: list[SuppliedLine],
    words: list[WordObs],
    norm_words: list[str],
    min_anchor_score: float,
) -> list[list[list[tuple[int, float]]]]:
    """cands[i][j] = [(length, score), ...] for windows of line i at word j
    that pass the acceptance threshold and the structural span rules."""
    from rapidfuzz import fuzz

    m = len(words)
    # breaks[k] = 1 when the gap between word k and k+1 is too long.
    breaks_prefix = [0]
    for k in range(m - 1):
        too_long = words[k + 1].start - words[k].end > MAX_INTERNAL_GAP_SECONDS
        breaks_prefix.append(breaks_prefix[-1] + (1 if too_long else 0))
    window_text: dict[tuple[int, int], str] = {}
    cutoff = max(0.0, min_anchor_score) * 100.0

    out: list[list[list[tuple[int, float]]]] = []
    for line in seq:
        tokens = [normalize(t) for t in line.text.split()]
        tokens = [t for t in tokens if t]
        per_start: list[list[tuple[int, float]]] = [[] for _ in range(m)]
        out.append(per_start)
        if not tokens:
            continue
        line_text = " ".join(tokens)
        k = len(tokens)
        for j in range(m):
            for length in range(max(1, k - WINDOW_SHORTER_BY), k + WINDOW_LONGER_BY + 1):
                end = j + length
                if end > m:
                    break
                if words[end - 1].end - words[j].start > MAX_WINDOW_SECONDS:
                    break
                if breaks_prefix[end - 1] - breaks_prefix[j] > 0:
                    break  # longer windows contain the same gap
                key = (j, length)
                text = window_text.get(key)
                if text is None:
                    text = " ".join(norm_words[j:end])
                    window_text[key] = text
                score = fuzz.token_sort_ratio(line_text, text, score_cutoff=cutoff)
                if score <= 0.0 or score / 100.0 < min_anchor_score:
                    continue
                per_start[j].append((length, score / 100.0))
    return out


# --- global monotone DP --------------------------------------------------------


def _hint_cost(hint: float | None, offset: float, start: float) -> float:
    if hint is None:
        return 0.0
    distance = min(abs(start - (hint + offset)), HINT_COST_CAP_SECONDS)
    return HINT_COST_PER_SECOND * distance


def _monotone_dp(
    seq: list[SuppliedLine],
    words: list[WordObs],
    cands: list[list[list[tuple[int, float]]]],
    use_hints: bool,
    offset: float,
) -> tuple[list[tuple[int, int, float] | None], float]:
    """Choose ≤ 1 window per line, windows in line order and disjoint,
    maximising Σ (SKIP_PENALTY + score − hint cost). Returns (per line
    (start, length, score) or None when skipped, the maximised total)."""
    n, m = len(seq), len(words)
    # g[i][j]: best total for lines i.. using words j.. (backward DP, so
    # the forward reconstruction can prefer the earliest window).
    g = [[0.0] * (m + 1) for _ in range(n + 1)]

    def gain(i: int, j: int, score: float) -> float:
        hint = seq[i].hint_seconds if use_hints else None
        return SKIP_PENALTY + score - _hint_cost(hint, offset, words[j].start)

    for i in range(n - 1, -1, -1):
        row, below = g[i], g[i + 1]
        for j in range(m - 1, -1, -1):
            best = row[j + 1] if row[j + 1] > below[j] else below[j]
            for length, score in cands[i][j]:
                value = gain(i, j, score) + below[j + length]
                if value > best:
                    best = value
            row[j] = best

    chosen: list[tuple[int, int, float] | None] = [None] * n
    i = j = 0
    while i < n and j < m:
        target = g[i][j]
        picked = None
        for length, score in cands[i][j]:  # ascending length
            if gain(i, j, score) + g[i + 1][j + length] >= target - _EPS:
                if picked is None or score > picked[1] + _EPS:
                    picked = (length, score)
        if picked is not None:
            chosen[i] = (j, picked[0], picked[1])
            j += picked[0]
            i += 1
        elif g[i][j + 1] >= target - _EPS:
            j += 1  # line i anchors later (or is skipped later)
        else:
            i += 1  # line i is skipped
    return chosen, g[0][0]


def _strong_occurrences(per_start: list[list[tuple[int, float]]]) -> int:
    """How many separate places in the transcript match a line strongly
    (≥ OFFSET_MIN_SCORE); overlapping windows count as one place."""
    count, reach = 0, -1
    for j, cands in enumerate(per_start):
        strong = [length for length, score in cands if score >= OFFSET_MIN_SCORE]
        if not strong:
            continue
        if j >= reach:
            count += 1
        reach = max(reach, j + max(strong))
    return count


def _estimate_offset(
    seq: list[SuppliedLine],
    words: list[WordObs],
    cands: list[list[list[tuple[int, float]]]],
    chosen: list[tuple[int, int, float] | None],
) -> float:
    """median(anchor − hint) over confidently anchored hinted lines that are
    unambiguous — text unique in the file AND one strong match in the
    transcript (a line written once but sung three times says nothing about
    the offset); 0.0 when there are none."""
    keys = [" ".join(normalize(t) for t in l.text.split()) for l in seq]
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    deltas = sorted(
        words[c[0]].start - line.hint_seconds
        for line, key, c, per_start in zip(seq, keys, chosen, cands)
        if c is not None
        and line.hint_seconds is not None
        and counts[key] == 1
        and c[2] >= OFFSET_MIN_SCORE
        and _strong_occurrences(per_start) == 1
    )
    if not deltas:
        return 0.0
    mid = len(deltas) // 2
    median = deltas[mid] if len(deltas) % 2 else (deltas[mid - 1] + deltas[mid]) / 2
    return canonical.round_seconds(median)


# --- word pairing inside a window -----------------------------------------------


def _pair_words(
    supplied: list[str],
    window: list[WordObs],
    edges: tuple[float, float] | None = None,
) -> tuple[list[tuple[str, float, float, float]], bool, set[int]]:
    """Pair supplied words with window words by content (LYR-8).

    Returns (words, interpolated, used) — words as (supplied text, start,
    end, confidence); `interpolated` is True when any word's timing was not
    observed one-to-one; `used` holds the window indices paired."""
    a = [normalize(t) for t in supplied]
    b = [normalize(w.text) for w in window]
    n, m = len(a), len(b)

    def moves(p: int, q: int):
        """Diagonal moves from (p, q) as (dp, dq, gain, ratio), in
        preference order."""
        out = []
        if p < n and q < m:
            r = _ratio(a[p], b[q])
            if r >= WORD_MATCH_RATIO:
                out.append((1, 1, r, r))
        # Merge / split only when the single words don't already match on
        # their own ("oh"+"hold" ≈ "hold" must stay a dropped "oh").
        if (
            p + 1 < n
            and q < m
            and _ratio(a[p], b[q]) < WORD_MATCH_RATIO
            and _ratio(a[p + 1], b[q]) < WORD_MATCH_RATIO
        ):
            r = _ratio(a[p] + a[p + 1], b[q])
            if r >= WORD_MATCH_RATIO:
                out.append((2, 1, 2 * r * _MERGE_WEIGHT, r))
        if (
            p < n
            and q + 1 < m
            and _ratio(a[p], b[q]) < WORD_MATCH_RATIO
            and _ratio(a[p], b[q + 1]) < WORD_MATCH_RATIO
        ):
            r = _ratio(a[p], b[q] + b[q + 1])
            if r >= WORD_MATCH_RATIO:
                out.append((1, 2, r * _SPLIT_WEIGHT, r))
        return out

    h = [[0.0] * (m + 1) for _ in range(n + 1)]
    for p in range(n, -1, -1):
        for q in range(m, -1, -1):
            if p == n and q == m:
                continue
            best = 0.0
            if q < m:
                best = max(best, h[p][q + 1])
            if p < n:
                best = max(best, h[p + 1][q])
            for dp, dq, g, _ in moves(p, q):
                best = max(best, g + h[p + dp][q + dq])
            h[p][q] = best

    # (supplied index) -> (start, end, confidence, shared)
    assigned: dict[int, tuple[float, float, float, bool]] = {}
    used: set[int] = set()
    edge_start, edge_end = edges or (window[0].start, window[-1].end)
    p = q = 0
    while p < n or q < m:
        target = h[p][q]
        step = None
        for dp, dq, g, r in moves(p, q):
            if g + h[p + dp][q + dq] >= target - _EPS:
                step = (dp, dq, r)
                break
        if step is not None:
            dp, dq, r = step
            if dp == 1 and dq == 1:
                assigned[p] = (window[q].start, window[q].end, r, False)
            elif dp == 2:  # two supplied words heard as one: share its span
                w = window[q]
                share = len(a[p]) / max(1, len(a[p]) + len(a[p + 1]))
                mid = w.start + (w.end - w.start) * share
                assigned[p] = (w.start, mid, r, True)
                assigned[p + 1] = (mid, w.end, r, True)
            else:  # one supplied word heard as two
                assigned[p] = (window[q].start, window[q + 1].end, r, False)
            used.update(range(q, q + dq))
            p, q = p + dp, q + dq
        elif q < m and h[p][q + 1] >= target - _EPS:
            q += 1  # inserted (heard, not supplied) word
        else:
            p += 1  # supplied word not heard

    words: list[tuple[str, float, float, float]] = []
    interpolated = any(v[3] for v in assigned.values())
    p = 0
    while p < n:
        if p in assigned:
            s, e, r, _ = assigned[p]
            words.append((supplied[p], s, e, canonical.round_ratio(r)))
            p += 1
            continue
        # Run of unmatched supplied words: interpolate between matched
        # neighbours (window edges when none), split by character share.
        run_end = p
        while run_end < n and run_end not in assigned:
            run_end += 1
        # Leading / trailing runs are bounded by the anchored (DP) window's
        # edges, never stretched over the unpaired margin words.
        if run_end < n:
            right = assigned[run_end][0]
            left = words[-1][2] if words else min(edge_start, right)
            # Observed neighbours may overlap (Whisper word spans can, and
            # so can clamping): never emit end < start — the run collapses
            # to a zero-length span at the left neighbour's end.
            right = max(right, left)
        else:
            left = words[-1][2] if words else edge_start
            right = max(edge_end, left)
        # Unpaired heard words in this gap: the best-matching one's ratio is
        # the word's (low) confidence — the schema's token match score.
        gap_obs = [
            b[k]
            for k in range(m)
            if k not in used and window[k].start >= left - _EPS
            and window[k].end <= right + _EPS
        ]
        weights = [max(1, len(a[k])) for k in range(p, run_end)]
        total = sum(weights)
        t = left
        for k, wgt in zip(range(p, run_end), weights):
            span = (right - left) * wgt / total
            conf = max((_ratio(a[k], o) for o in gap_obs), default=0.0)
            words.append((supplied[k], t, t + span, canonical.round_ratio(conf)))
            t += span
        interpolated = True
        p = run_end
    return words, interpolated, used


def _pairing_range(
    words: list[WordObs],
    chosen: list[tuple[int, int, float] | None],
    idx: int,
    start: int,
    length: int,
    last_used: int,
) -> tuple[int, int]:
    """Word range [lo, hi) the anchored window's words are paired within:
    the window, widened by up to PAIRING_MARGIN_WORDS on each side over
    words no other line can use (after the previous line's last paired
    word, before the next anchored window) and without crossing a gap over
    MAX_INTERNAL_GAP_SECONDS. token_sort_ratio can prefer a window that
    drops a heard edge word ("love you" for "I love you"); content pairing
    recovers it here instead of interpolating the word."""
    next_start = next(
        (c[0] for c in chosen[idx + 1 :] if c is not None), len(words)
    )
    lo, hi = start, start + length
    for _ in range(PAIRING_MARGIN_WORDS):
        k = lo - 1
        if k <= last_used or words[lo].start - words[k].end > MAX_INTERNAL_GAP_SECONDS:
            break
        lo = k
    for _ in range(PAIRING_MARGIN_WORDS):
        if hi >= next_start or words[hi].start - words[hi - 1].end > MAX_INTERNAL_GAP_SECONDS:
            break
        hi += 1
    return lo, hi


# --- unanchored lines ------------------------------------------------------------


def _place_unanchored(
    results: list[AnchoredLine],
    hints: list[float | None],
    clamp,
    duration: float | None,
) -> None:
    """Line timing for unanchored lines, in sequence order (see module
    docstring, step 6). `hints` are offset-corrected."""
    n = len(results)

    def fixed_start(k: int) -> float | None:
        line = results[k]
        if line.anchored:
            return line.start
        if hints[k] is not None:
            return clamp(hints[k])
        return None

    i = 0
    while i < n:
        line = results[i]
        if line.anchored:
            i += 1
            continue
        if hints[i] is not None:
            start = clamp(hints[i])
            nxt = next(
                (s for s in (fixed_start(k) for k in range(i + 1, n))
                 if s is not None and s > start),
                None,
            )
            if nxt is not None:
                end = min(nxt, start + UNANCHORED_LINE_CAP_SECONDS)
            else:
                end = start + OPEN_ENDED_LINE_SECONDS
            line.start, line.end = start, clamp(end)
            i += 1
            continue
        run_end = i
        while (
            run_end < n
            and not results[run_end].anchored
            and hints[run_end] is None
        ):
            run_end += 1
        k = run_end - i
        left = results[i - 1].end if i > 0 else 0.0
        right = fixed_start(run_end) if run_end < n else None
        if right is not None and right > left:
            # Starts evenly spaced over the whole gap; each line lasts its
            # slot, capped, so a long instrumental gap is not claimed as
            # sung and the run is not bunched at the left edge.
            slot = (right - left) / k
            width = min(slot, UNANCHORED_LINE_CAP_SECONDS)
        else:
            width = OPEN_ENDED_LINE_SECONDS
            if duration is not None:
                width = min(width, max(0.0, duration - left) / k)
            slot = width
        for m in range(k):
            results[i + m].start = clamp(left + m * slot)
            results[i + m].end = clamp(left + m * slot + width)
        i = run_end


# --- entry points ----------------------------------------------------------------


def align(
    lines: list[SuppliedLine],
    words: list[WordObs],
    min_anchor_score: float = 0.6,
    duration: float | None = None,
) -> Alignment:
    """Align supplied lines to an observed word stream (module docstring).

    `duration` (seconds) clamps every emitted time to [0, duration]; None
    clamps only at 0 (pure tests that don't care about the track end)."""

    def clamp(t: float) -> float:
        t = max(t, 0.0)
        return min(t, duration) if duration is not None else t

    norm_words = [normalize(w.text) for w in words]

    def solve(seq: list[SuppliedLine]):
        cands = _candidates(seq, words, norm_words, min_anchor_score)
        offset = 0.0
        chosen, total = _monotone_dp(seq, words, cands, use_hints=False, offset=0.0)
        if any(l.hint_seconds is not None for l in seq):
            offset = _estimate_offset(seq, words, cands, chosen)
            chosen, total = _monotone_dp(
                seq, words, cands, use_hints=True, offset=offset
            )
        return seq, chosen, offset, total

    best = solve(list(lines))
    reordered = _hint_sorted(lines)
    if any(a is not b for a, b in zip(reordered, lines)):
        alt = solve(reordered)
        if alt[3] > best[3] + _EPS:  # ties keep file order
            best = alt
    seq, chosen, offset, _ = best

    results: list[AnchoredLine] = []
    used_words: set[int] = set()
    last_used = -1
    for idx, (line, pick) in enumerate(zip(seq, chosen)):
        if pick is None:
            results.append(
                AnchoredLine(
                    text=line.text,
                    start=0.0,
                    end=0.0,
                    anchored=False,
                    confidence=None,
                    flags=list(UNANCHORED_FLAGS),
                    words=[],
                    source_line_index=line.source_line_index,
                )
            )
            continue
        start, length, _ = pick
        lo, hi = _pairing_range(words, chosen, idx, start, length, last_used)
        window = words[lo:hi]
        supplied_words = [t for t in line.text.split() if normalize(t)]
        edges = (words[start].start, words[start + length - 1].end)
        paired, interpolated, used = _pair_words(supplied_words, window, edges)
        used_words.update(lo + k for k in used)
        if used:
            last_used = lo + max(used)
        paired = [(t, clamp(s), clamp(e), c) for t, s, e, c in paired]
        confidence = canonical.round_ratio(sum(p[3] for p in paired) / len(paired))
        results.append(
            AnchoredLine(
                text=line.text,
                start=paired[0][1],
                end=paired[-1][2],
                anchored=True,
                confidence=confidence,
                flags=["timing_interpolated"] if interpolated else [],
                words=paired,
                source_line_index=line.source_line_index,
            )
        )

    hints = [
        l.hint_seconds + offset if l.hint_seconds is not None else None for l in seq
    ]
    _place_unanchored(results, hints, clamp, duration)
    # Schema invariant: lines[] sorted by start_seconds (stable: sequence
    # order on equal starts).
    results.sort(key=lambda a: a.start)

    unmatched = [
        WordObs(w.text, clamp(w.start), clamp(w.end), w.probability)
        for k, w in enumerate(words)
        if k not in used_words
    ]
    return Alignment(lines=results, unmatched=unmatched, hint_offset=offset)


def align_lines(
    lines: list[SuppliedLine],
    words: list[WordObs],
    min_anchor_score: float = 0.6,
    duration: float | None = None,
) -> list[AnchoredLine]:
    """`align(...)` lines only — convenience for callers and tests that
    don't need the unmatched words or the hint offset."""
    return align(lines, words, min_anchor_score, duration).lines
