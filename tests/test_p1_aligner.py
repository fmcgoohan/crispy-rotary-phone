"""Pure aligner tests for the lyrics algorithm-4 rewrite (mrw/align.py):
global monotone line alignment (LYR-1), hints as a cost term with a global
offset estimate (LYR-2), content-based word pairing (LYR-8), unanchored
spreading and clamping (LYR-11, SCH-3), unmatched words (CRIT-1).

Synthetic word streams only — no audio, no Whisper.
"""

from __future__ import annotations

from mrw.align import SuppliedLine, WordObs, align, align_lines


def _say(text: str, start: float, step: float = 0.4, prob: float = 0.9) -> list[WordObs]:
    """One sung phrase: consecutive words `step` seconds apart."""
    out = []
    for k, word in enumerate(text.split()):
        s = round(start + k * step, 3)
        out.append(WordObs(word, s, round(s + step - 0.05, 3), prob))
    return out


def _by_index(lines):
    return {a.source_line_index: a for a in lines}


# --- LYR-1: global monotone alignment ---------------------------------------


V1 = "city lights are bleeding through the window"
C1 = "neon reverie don't wake me up"
C2 = "hold the static in your hands"
V2 = "every signal fading into grey tonight"


def test_misheard_first_chorus_keeps_its_own_occurrence() -> None:
    # The first chorus is slightly misheard, the second is clean and lies
    # well within any lookahead. Greedy argmax anchored chorus-1 lines to
    # chorus 2 and left verse 2 / chorus 2 unaligned.
    words = (
        _say(V1, 16.0)
        + _say("neon river he don't wake me up", 46.0)
        + _say("hold the static in your hand", 50.0)
        + _say(V2, 80.0)
        + _say(C1, 110.0)
        + _say(C2, 114.0)
    )
    lines = [
        SuppliedLine(V1, 0),
        SuppliedLine(C1, 1),
        SuppliedLine(C2, 2),
        SuppliedLine(V2, 3),
        SuppliedLine(C1, 4),
        SuppliedLine(C2, 5),
    ]
    out = _by_index(align_lines(lines, words))
    assert all(a.anchored for a in out.values())
    assert out[0].start == 16.0
    assert out[1].start == 46.0  # its own (misheard) occurrence
    assert out[2].start == 50.0
    assert out[3].start == 80.0
    assert out[4].start == 110.0
    assert out[5].start == 114.0


def test_transcript_word_never_used_by_two_lines() -> None:
    # Two identical supplied lines, one sung occurrence: only one anchors.
    words = _say("give it to me", 40.0)
    out = align_lines(
        [SuppliedLine("give it to me", 0), SuppliedLine("give it to me", 1)], words
    )
    assert [a.anchored for a in sorted(out, key=lambda a: a.source_line_index)] == [
        True,
        False,
    ]


def test_long_internal_gap_rejects_window() -> None:
    # "first half" ... 20 s break ... "second half": no window may span it.
    words = _say("first half of", 10.0) + _say("the line", 30.0)
    result = align([SuppliedLine("first half of the line", 0)], words)
    a = result.lines[0]
    # It anchors on the first phrase alone; the words after the break are
    # never part of its window, so they stay unmatched.
    assert a.anchored
    assert 10.0 <= a.start and a.end < 12.0
    assert [w.text for w in result.unmatched] == ["the", "line"]


# --- LYR-2: hints as a cost term ---------------------------------------------------


HOOK = "give it to me"


def _hook_stream() -> list[WordObs]:
    return (
        _say("we were running through the night", 30.0)
        + _say(HOOK, 40.0)
        + _say(HOOK, 43.0)
        + _say("under the burning sky above", 50.0)
        + _say(HOOK, 60.0)
    )


def _hook_lines(shift: float | None) -> list[SuppliedLine]:
    true = [30.0, 40.0, 43.0, 50.0, 60.0]
    texts = [
        "we were running through the night",
        HOOK,
        HOOK,
        "under the burning sky above",
        HOOK,
    ]
    return [
        SuppliedLine(t, i, None if shift is None else round(s + shift, 3))
        for i, (t, s) in enumerate(zip(texts, true))
    ]


def test_back_to_back_identical_hooks_with_exact_hints() -> None:
    out = _by_index(align_lines(_hook_lines(0.0), _hook_stream()))
    assert [out[i].start for i in range(5)] == [30.0, 40.0, 43.0, 50.0, 60.0]
    assert all(a.anchored for a in out.values())


def test_shifted_hints_give_same_anchors() -> None:
    expected = [30.0, 40.0, 43.0, 50.0, 60.0]
    for shift in (-25.0, 10.0, 0.0, None):
        result = align(_hook_lines(shift), _hook_stream())
        out = _by_index(result.lines)
        assert [out[i].start for i in range(5)] == expected, shift
        if shift is not None:
            assert result.hint_offset == -shift


def test_hint_picks_among_repeats_when_file_has_fewer_lines() -> None:
    # The file carries the hook once; the audio sings it three times. The
    # hint decides which occurrence anchors (unhinted: the earliest).
    stream = _say(HOOK, 10.0) + _say(HOOK, 30.0) + _say(HOOK, 50.0)
    assert align_lines([SuppliedLine(HOOK, 0)], stream)[0].start == 10.0
    hinted = align_lines([SuppliedLine(HOOK, 0, hint_seconds=31.0)], stream)
    assert hinted[0].start == 30.0


def test_unanchored_hinted_line_uses_offset_corrected_hint() -> None:
    # Hints 25 s early; the unfindable line lands where its corrected hint
    # says, not at the stale hint.
    words = (
        _say("we were running through the night", 30.0)
        + _say("under the burning sky above", 50.0)
    )
    lines = [
        SuppliedLine("we were running through the night", 0, 5.0),
        SuppliedLine("zzz qqq xxx", 1, 15.0),
        SuppliedLine("under the burning sky above", 2, 25.0),
    ]
    out = _by_index(align_lines(lines, words))
    assert not out[1].anchored
    assert out[1].start == 40.0


# --- LYR-8: content-based word pairing ------------------------------------------------


def _pairs(supplied: str, heard: list[WordObs]):
    out = align_lines([SuppliedLine(supplied, 0)], heard)
    assert out[0].anchored
    return out[0]


def test_dropped_word_is_interpolated_and_flagged() -> None:
    heard = _stream(("i", 10.0, 10.3), ("really", 10.3, 10.8),
                    ("love", 10.8, 11.2), ("you", 11.2, 11.6))
    a = _pairs("I really really love you", heard)
    texts = [w[0] for w in a.words]
    assert texts == ["I", "really", "really", "love", "you"]
    # 'love' keeps its own observed span — not the dropped word's.
    love = a.words[3]
    assert (love[1], love[2]) == (10.8, 11.2)
    assert "timing_interpolated" in a.flags
    spans = [(w[1], w[2]) for w in a.words]
    assert len(set(spans)) == len(spans)  # no two words share one span
    starts = [w[1] for w in a.words]
    assert starts == sorted(starts)


def test_inserted_word_goes_to_unmatched() -> None:
    heard = _stream(("i", 10.0, 10.3), ("really", 10.3, 10.8),
                    ("love", 10.8, 11.2), ("you", 11.2, 11.6))
    result = align([SuppliedLine("I love you", 0)], heard)
    a = result.lines[0]
    assert [(w[0], w[1]) for w in a.words] == [("I", 10.0), ("love", 10.8), ("you", 11.2)]
    assert a.flags == []  # every supplied word observed
    assert [w.text for w in result.unmatched] == ["really"]


def test_merged_words_share_span_with_flag() -> None:
    heard = _stream(("everybody", 5.0, 6.0), ("dance", 6.0, 6.5), ("now", 6.5, 7.0))
    a = _pairs("every body dance now", heard)
    every, body = a.words[0], a.words[1]
    assert every[1] == 5.0 and body[2] == 6.0
    assert every[2] == body[1]  # split inside the heard word
    assert "timing_interpolated" in a.flags


def test_split_word_takes_both_heard_spans() -> None:
    heard = _stream(("every", 5.0, 5.4), ("body", 5.4, 6.0), ("dance", 6.0, 6.5))
    a = _pairs("everybody dance", heard)
    assert a.words[0][1:3] == (5.0, 6.0)
    assert a.flags == []


def test_leading_dropped_word_does_not_shift_timings() -> None:
    heard = _stream(("hold", 2.0, 2.4), ("on", 2.4, 2.8), ("tonight", 2.8, 3.6))
    a = _pairs("oh hold on tonight", heard)
    assert [(w[0], w[1]) for w in a.words[1:]] == [
        ("hold", 2.0), ("on", 2.4), ("tonight", 2.8)
    ]
    assert "timing_interpolated" in a.flags


def _stream(*items: tuple[str, float, float]) -> list[WordObs]:
    return [WordObs(t, s, e, 0.9) for t, s, e in items]


# --- LYR-11 / SCH-3: unanchored placement and clamping --------------------------------


def test_consecutive_unanchored_lines_are_spread_not_stacked() -> None:
    words = _say("first line here", 10.0) + _say("last line here", 40.0)
    lines = [
        SuppliedLine("first line here", 0),
        SuppliedLine("zzz qqq", 1),
        SuppliedLine("xxx vvv", 2),
        SuppliedLine("kkk jjj", 3),
        SuppliedLine("last line here", 4),
    ]
    out = _by_index(align_lines(lines, words))
    mids = [out[i] for i in (1, 2, 3)]
    assert all(not a.anchored for a in mids)
    spans = [(a.start, a.end) for a in mids]
    assert len(set(spans)) == 3  # distinct
    assert out[0].end <= spans[0][0] < spans[0][1] <= spans[1][0]
    assert spans[1][1] <= spans[2][0] < spans[2][1] <= out[4].start
    assert all(e - s <= 10.0 for s, e in spans)


def test_times_clamped_to_duration() -> None:
    words = _say("only line here", 170.0)
    lines = [
        SuppliedLine("only line here", 0),
        SuppliedLine("trailing outro one", 1),  # unhinted, after last anchor
        SuppliedLine("trailing outro two", 2),
        SuppliedLine("past the end", 3, hint_seconds=195.0),
    ]
    out = align_lines(lines, words, duration=180.0)
    for a in out:
        assert 0.0 <= a.start <= a.end <= 180.0
        for w in a.words:
            assert 0.0 <= w[1] <= w[2] <= 180.0
    trailing = [a for a in out if a.source_line_index in (1, 2)]
    assert trailing[0].end <= trailing[1].start  # spread, not stacked


def test_zero_anchor_convention() -> None:
    # No transcript at all: unhinted lines laid out in order from 0 s,
    # 4 s each (less when the track is too short); hinted lines keep their
    # hints, clamped to the track.
    lines = [SuppliedLine(f"line {k}", k) for k in range(3)]
    result = align(lines, [], duration=30.0)
    out = _by_index(result.lines)
    assert [(out[k].start, out[k].end) for k in range(3)] == [
        (0.0, 4.0), (4.0, 8.0), (8.0, 12.0)
    ]
    # A track too short for 4 s each: the run is spread evenly to the end.
    short = _by_index(align_lines(lines, [], duration=6.0))
    assert [(short[k].start, short[k].end) for k in range(3)] == [
        (0.0, 2.0), (2.0, 4.0), (4.0, 6.0)
    ]
    assert result.hint_offset == 0.0
    assert all("unaligned" in a.flags for a in result.lines)

    # A hinted line past the end clamps to the duration and bounds the
    # unhinted run before it, which is spread evenly up to it.
    hinted = lines + [SuppliedLine("hinted", 3, hint_seconds=20.0)]
    out = _by_index(align_lines(hinted, [], duration=10.0))
    assert (out[3].start, out[3].end) == (10.0, 10.0)
    starts = [out[k].start for k in range(3)]
    assert starts == sorted(starts) and len(set(starts)) == 3


def test_clamps_words_beyond_duration() -> None:
    words = _say("sing it loud", 9.0, step=0.6)  # last word ends 10.15
    out = align_lines([SuppliedLine("sing it loud", 0)], words, duration=10.0)
    assert out[0].end == 10.0
    assert out[0].words[-1][2] == 10.0


# --- CRIT-1: unmatched words ---------------------------------------------------------


def test_unmatched_words_returned_in_stream_order() -> None:
    words = (
        _say("hold on hold on tonight", 10.0)
        + _say("ooh yeah ooh", 20.0)  # ad-lib the file omits
        + _say("hold on until the light", 30.0)
    )
    result = align(
        [SuppliedLine("hold on hold on tonight", 0),
         SuppliedLine("hold on until the light", 1)],
        words,
    )
    assert all(a.anchored for a in result.lines)
    assert [w.text for w in result.unmatched] == ["ooh", "yeah", "ooh"]
    assert result.unmatched[0].start == 20.0


def test_alignment_is_deterministic() -> None:
    lines = _hook_lines(-25.0)
    a = align(lines, _hook_stream(), duration=100.0)
    b = align(lines, _hook_stream(), duration=100.0)
    assert a == b


# --- review B: hints never reorder; spans never invert; long-gap spread ---------


def test_single_wrong_hint_does_not_move_line() -> None:
    # Five lines sung cleanly at 10/20/30/40/50 s; line 2's .lrc timestamp is
    # a typo (300 s). The hint is only a capped cost: the line still anchors
    # at 30 s, and the offset estimate ignores the outlier.
    texts = [
        "city lights are bleeding through",
        "neon reverie dont wake me",
        "hold the static in your hands",
        "every signal fading into grey",
        "burning skyline over water",
    ]
    words = [w for k, t in enumerate(texts) for w in _say(t, 10.0 + 10.0 * k)]
    hints = [10.0, 20.0, 300.0, 40.0, 50.0]
    result = align(
        [SuppliedLine(t, i, h) for i, (t, h) in enumerate(zip(texts, hints))],
        words,
        duration=120.0,
    )
    out = _by_index(result.lines)
    assert all(out[i].anchored for i in range(5))
    assert out[2].start == 30.0
    assert result.hint_offset == 0.0
    assert [a.source_line_index for a in result.lines] == [0, 1, 2, 3, 4]


def test_overlapping_heard_words_never_invert_interpolated_span() -> None:
    # Whisper word spans may overlap: a(1.0-1.6) c(1.5-2.0). The unheard
    # supplied word between them must not end before it starts.
    words = _stream(("a", 1.0, 1.6), ("c", 1.5, 2.0), ("d", 2.0, 2.4))
    a = align_lines([SuppliedLine("a bb c d", 0)], words)[0]
    assert a.anchored and "timing_interpolated" in a.flags
    for _text, start, end, _conf in a.words:
        assert start <= end
    assert [w[0] for w in a.words] == ["a", "bb", "c", "d"]


def test_unanchored_run_spread_over_long_gap() -> None:
    # Anchors end ~12 s and start at 70 s; two unanchored lines between get
    # starts spread over the whole gap (not bunched at its left edge), each
    # capped at 10 s.
    words = _say("first line here", 10.0) + _say("last line here", 70.0)
    lines = [
        SuppliedLine("first line here", 0),
        SuppliedLine("zzz qqq", 1),
        SuppliedLine("xxx vvv", 2),
        SuppliedLine("last line here", 3),
    ]
    out = _by_index(align_lines(lines, words))
    left, right = out[0].end, out[3].start
    slot = (right - left) / 2
    assert out[1].start == left
    assert abs(out[2].start - (left + slot)) < 1e-9
    for i in (1, 2):
        assert abs((out[i].end - out[i].start) - 10.0) < 1e-9
    assert out[2].end <= right
