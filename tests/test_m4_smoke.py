"""M4 smoke test (PLAN §11 / M4 milestone task).

Fast (no Whisper anywhere): pure-aligner anchoring (exact / fuzzy /
unanchored), markup extraction (R-3), flag thresholds, untranscribed-region
computation, and document double-run byte identity with a stubbed engine.

Slow (real Whisper): the tone+click fixture in transcribed mode asserts the
DEGENERATE path (no vocals → empty-ish lines, honest coverage, valid
document); the espeak-ng ground-truth test synthesizes spoken words over a
click bed and asserts they survive separation + transcription with sane
timestamps.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

import mrw.lyrics
from mrw.align import SuppliedLine, WordObs, align_lines
from mrw.cli import app
from mrw.lyrics import Segment, parse_supplied_lyrics
from mrw.stems import STEM_NAMES
from test_m3_smoke import _click_track, _tone, _write_stereo_wav

REPO = Path(__file__).resolve().parents[1]
SCHEMAS = REPO / "schemas"
SAMPLE_RATE = 44100

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)

runner = CliRunner()


# --- helpers -----------------------------------------------------------------


def _stream(*items: tuple[str, float, float]) -> list[WordObs]:
    return [WordObs(text=t, start=s, end=e, probability=0.9) for t, s, e in items]


def _validate(document: dict, schema_name: str) -> None:
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    registry = Registry()
    for f in SCHEMAS.glob("*.schema.json"):
        registry = registry.with_resource(
            f.name, Resource.from_contents(json.loads(f.read_text()))
        )
    schema = json.loads((SCHEMAS / schema_name).read_text())
    errors = list(Draft202012Validator(schema, registry=registry).iter_errors(document))
    assert not errors, "\n".join(e.message for e in errors)


def _fabricate(
    library: Path,
    mix: np.ndarray,
    stems: dict[str, np.ndarray],
    tmp: Path,
    lyrics_text: str | None = None,
) -> str:
    """Ingest (optionally with lyrics), plant synthetic stems, run features."""
    wav = tmp / f"mix_{abs(hash(mix.tobytes())) % 10**8}.wav"
    _write_stereo_wav(wav, mix)
    args = ["ingest", str(wav), "--library", str(library)]
    if lyrics_text is not None:
        lyr = tmp / f"{wav.stem}.lrc"
        lyr.write_text(lyrics_text)
        args += ["--lyrics", str(lyr)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    track_id = next(p.name for p in library.iterdir() if p.is_dir())

    stems_dir = library / track_id / "stems"
    stems_dir.mkdir()
    for name in STEM_NAMES:
        y = stems.get(name, np.zeros_like(mix))
        pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
        sf.write(
            stems_dir / f"{name}.flac",
            np.repeat(pcm[:, None], 2, axis=1),
            SAMPLE_RATE,
            subtype="PCM_16",
        )
    manifest_path = library / track_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["stems"] = {"status": "ok", "retained": True, "config_hash": "0" * 16}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    result = runner.invoke(app, ["features", track_id, "--library", str(library)])
    assert result.exit_code == 0, result.output
    return track_id


def _bursty_vocals(seconds: float = 10.0) -> np.ndarray:
    """Vocal stem with tone bursts at 2-4 s and 6-8 s (two VA regions).

    amp 0.15 → ~-19.5 dBFS RMS: inside vocal-activity detection (enter -35)
    but below the overlapping_vocals threshold (-15) — a normal-loudness
    vocal, so no flags fire from loudness alone."""
    y = np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)
    tone = _tone(2.0, 440.0, amp=0.15)
    y[2 * SAMPLE_RATE : 4 * SAMPLE_RATE] = tone
    y[6 * SAMPLE_RATE : 8 * SAMPLE_RATE] = tone
    return y


def _stub_engine(monkeypatch, segments: list[Segment], language: str = "en") -> None:
    monkeypatch.setattr(
        mrw.lyrics, "_transcribe", lambda path, cfg: (language, segments)
    )


# --- fast: pure aligner -------------------------------------------------------


def test_align_exact_and_fuzzy_anchoring() -> None:
    words = _stream(
        ("hello", 1.0, 1.3), ("world", 1.35, 1.7),
        ("second", 3.0, 3.4), ("line", 3.45, 3.8), ("here", 3.85, 4.2),
    )
    lines = [
        SuppliedLine("Hello, world!", 0),
        SuppliedLine("secund lyne here", 1),  # fuzzy but recognizable
    ]
    out = align_lines(lines, words)
    assert all(a.anchored for a in out)
    assert out[0].start == 1.0 and out[0].end == 1.7
    assert out[0].words[0][0] == "Hello,"  # supplied text is display truth
    assert out[0].confidence > 0.9
    assert out[1].start == 3.0 and out[1].end == 4.2
    assert 0.6 <= out[1].confidence < 1.0


def test_align_unanchored_line_interpolates() -> None:
    words = _stream(
        ("first", 1.0, 1.4), ("line", 1.5, 1.9),
        ("third", 8.0, 8.4), ("line", 8.5, 8.9),
    )
    lines = [
        SuppliedLine("first line", 0),
        SuppliedLine("zzz qqq xxx", 1),  # nothing like it in the stream
        SuppliedLine("third line", 2),
    ]
    out = align_lines(lines, words)
    assert out[0].anchored and out[2].anchored
    middle = out[1]
    assert not middle.anchored
    assert set(middle.flags) == {"low_confidence", "unaligned", "timing_interpolated"}
    assert middle.words == []  # never fake word-level precision
    assert out[0].end <= middle.start < middle.end <= out[2].start


def test_align_unanchored_line_keeps_lrc_hint() -> None:
    words = _stream(("only", 1.0, 1.4), ("line", 1.5, 1.9))
    lines = [SuppliedLine("unfindable words", 0, hint_seconds=42.5)]
    out = align_lines(lines, words)
    assert not out[0].anchored
    assert out[0].start == 42.5


def test_out_of_order_lrc_hints_keep_lines_sorted() -> None:
    # H2 (PR #9 review): a malformed .lrc whose hints run backward must not
    # drag the cursor backward or emit lines[] unsorted by start_seconds.
    words = _stream(
        ("alpha", 1.0, 1.4), ("beta", 1.5, 1.9),
        ("gamma", 5.0, 5.4), ("delta", 5.5, 5.9),
        ("omega", 9.0, 9.4), ("last", 9.5, 9.9),
    )
    lines = [
        SuppliedLine("gamma delta", 0, hint_seconds=5.0),
        SuppliedLine("alpha beta", 1, hint_seconds=1.0),  # hint jumps backward
        SuppliedLine("omega last", 2),  # no hint: cursor must not have regressed
    ]
    out = align_lines(lines, words)
    assert all(a.anchored for a in out)
    starts = [a.start for a in out]
    assert starts == sorted(starts)  # schema invariant
    assert out[-1].start == 9.0  # unhinted line found after the cursor


# --- fast: markup + parsing (R-3) ----------------------------------------------


def test_markup_extraction_and_lrc_hints() -> None:
    raw = "\n".join(
        [
            "[Verse 1]",
            "[00:12.50]City lights are bleeding",
            "",
            "[Chorus]",
            "[00:45.00]Neon reverie",
            "[Weird Header]",
            "plain unstamped line",
        ]
    )
    lines, markup = parse_supplied_lyrics(raw)
    assert [m.text for m in markup] == ["[Verse 1]", "[Chorus]", "[Weird Header]"]
    assert [m.label for m in markup] == ["verse", "chorus", "other"]
    assert markup[0].source_line_index == 0
    assert [l.text for l in lines] == [
        "City lights are bleeding",
        "Neon reverie",
        "plain unstamped line",
    ]
    assert lines[0].hint_seconds == 12.5
    assert lines[1].hint_seconds == 45.0
    assert lines[2].hint_seconds is None


# --- fast: documents with a stubbed engine -------------------------------------


def _seg(text: str, start: float, end: float, words=None, **kw) -> Segment:
    return Segment(
        text=text,
        start=start,
        end=end,
        no_speech_prob=kw.get("no_speech_prob", 0.05),
        compression_ratio=kw.get("compression_ratio", 1.2),
        words=words
        or [
            WordObs(w, start + i * 0.3, start + i * 0.3 + 0.25, kw.get("prob", 0.9))
            for i, w in enumerate(text.split())
        ],
    )


def test_transcribed_document_untranscribed_and_flags(tmp_path, monkeypatch) -> None:
    mix = _click_track(10.0)
    vocals = _bursty_vocals(10.0)
    library = tmp_path / "lib"
    track_id = _fabricate(library, mix + vocals * 0.5, {"vocals": vocals}, tmp_path)

    # Words only in the first burst (2-4 s); second burst (6-8 s) unheard.
    # One hallucination-shaped segment triggers possibly_non_lexical.
    _stub_engine(
        monkeypatch,
        [
            _seg("la la la", 2.2, 3.6),
            _seg("oh oh oh oh", 9.0, 9.8, no_speech_prob=0.9, prob=0.3),
        ],
    )
    result = runner.invoke(app, ["lyrics", track_id, "--library", str(library)])
    assert result.exit_code == 0, result.output
    doc = json.loads((library / track_id / "lyrics.json").read_text())
    _validate(doc, "lyrics.schema.json")

    assert doc["mode"] == "transcribed"
    assert doc["language"] == "en"
    assert doc["engine"] == {"name": "faster-whisper", "model": "small"}
    # Second vocal-activity region has zero word overlap → untranscribed.
    regions = doc["untranscribed_regions"]
    assert len(regions) == 1
    assert abs(regions[0]["start_seconds"] - 6.0) < 0.2
    # Coverage is word-span overlap (schema: "overlapped by at least one
    # word"), not segment-span: 3 words x 0.25 s over 4 s of VA ≈ 0.19.
    assert 0.1 < doc["coverage"]["vocal_activity_covered_ratio"] < 0.3
    flags_by_line = [set(l["flags"]) for l in doc["lines"]]
    assert "possibly_non_lexical" in flags_by_line[1]
    assert "low_confidence" in flags_by_line[1]
    assert doc["coverage"]["lines_flagged_ratio"] == 0.5


def test_aligned_document_with_markup(tmp_path, monkeypatch) -> None:
    mix = _click_track(10.0)
    vocals = _bursty_vocals(10.0)
    library = tmp_path / "lib"
    lyrics_text = "[Verse 1]\n[00:02.20]hello world\n[00:06.30]second line here\n"
    track_id = _fabricate(
        library, mix + vocals * 0.5, {"vocals": vocals}, tmp_path, lyrics_text
    )
    _stub_engine(
        monkeypatch,
        [
            _seg("hello world", 2.2, 3.2),
            _seg("second line here", 6.3, 7.5),
        ],
    )
    result = runner.invoke(app, ["lyrics", track_id, "--library", str(library)])
    assert result.exit_code == 0, result.output
    doc = json.loads((library / track_id / "lyrics.json").read_text())
    _validate(doc, "lyrics.schema.json")

    assert doc["mode"] == "aligned"
    assert doc["supplied_markup"] == [
        {"text": "[Verse 1]", "label": "verse", "source_line_index": 0}
    ]
    assert [l["text"] for l in doc["lines"]] == ["hello world", "second line here"]
    assert doc["lines"][0]["flags"] == []
    assert doc["lines"][0]["words"][0]["text"] == "hello"
    assert abs(doc["lines"][0]["start_seconds"] - 2.2) < 0.05
    assert doc["untranscribed_regions"] == []


def test_lyrics_double_run_byte_identity(tmp_path, monkeypatch) -> None:
    mix = _click_track(8.0)
    vocals = _bursty_vocals(8.0)
    _stub_engine(monkeypatch, [_seg("la la la", 2.2, 3.6)])
    docs = []
    for name in ("lib_a", "lib_b"):
        library = tmp_path / name
        track_id = _fabricate(library, mix + vocals * 0.5, {"vocals": vocals}, tmp_path)
        result = runner.invoke(app, ["lyrics", track_id, "--library", str(library)])
        assert result.exit_code == 0, result.output
        docs.append((library / track_id / "lyrics.json").read_bytes())
    assert docs[0] == docs[1]


def test_lyrics_prerequisites(tmp_path) -> None:
    # No features yet → prerequisite error naming the command to run,
    # before any heavy import (torch/faster_whisper blocked to prove T5).
    sys_modules_guard = {"faster_whisper": None, "torch": None}
    import sys as _sys

    old = {k: _sys.modules.get(k) for k in sys_modules_guard}
    _sys.modules.update(sys_modules_guard)
    try:
        wav = tmp_path / "t.wav"
        _write_stereo_wav(wav, _tone(2.0, 330.0))
        library = tmp_path / "lib"
        result = runner.invoke(app, ["ingest", str(wav), "--library", str(library)])
        assert result.exit_code == 0
        track_id = next(p.name for p in library.iterdir() if p.is_dir())
        result = runner.invoke(app, ["lyrics", track_id, "--library", str(library)])
        assert result.exit_code == 2
        assert "mrw features" in result.output
    finally:
        for k, v in old.items():
            if v is None:
                _sys.modules.pop(k, None)
            else:
                _sys.modules[k] = v


def test_lyrics_engine_failure_records_failed(tmp_path, monkeypatch) -> None:
    # T4 (PR #9 review): mid-stage failure → exit 1, status=failed + error
    # in the manifest, library intact — the M2 stems precedent for lyrics.
    mix = _click_track(8.0)
    vocals = _bursty_vocals(8.0)
    library = tmp_path / "lib"
    track_id = _fabricate(library, mix + vocals * 0.5, {"vocals": vocals}, tmp_path)

    def boom(path, cfg):
        raise RuntimeError("decoder exploded (simulated)")

    monkeypatch.setattr(mrw.lyrics, "_transcribe", boom)
    result = runner.invoke(app, ["lyrics", track_id, "--library", str(library)])
    assert result.exit_code == 1
    manifest = json.loads((library / track_id / "manifest.json").read_text())
    entry = manifest["documents"]["lyrics"]
    assert entry["status"] == "failed"
    assert "decoder exploded" in entry["error"]
    assert not (library / track_id / "lyrics.json").exists()
    # Prior documents untouched.
    assert manifest["documents"]["audio_features"]["status"] == "ok"
    assert (library / track_id / "audio_features.json").is_file()


# --- slow: real Whisper ---------------------------------------------------------


def _run_chain(wav: Path, tmp_path: Path) -> tuple[Path, str]:
    library = tmp_path / "lib"
    cfg = tmp_path / "mrw.toml"
    cfg.write_text('[stems]\ndevice = "cpu"\n')
    for args in (
        ["ingest", str(wav), "--library", str(library)],
        None,
    ):
        if args:
            result = runner.invoke(app, args)
            assert result.exit_code == 0, result.output
    track_id = next(p.name for p in library.iterdir() if p.is_dir())
    for stage in ("stems", "features", "lyrics"):
        result = runner.invoke(
            app, [stage, track_id, "--library", str(library), "--config", str(cfg)]
        )
        assert result.exit_code == 0, f"{stage}: {result.output}"
    return library, track_id


@pytest.mark.slow
def test_transcribed_degenerate_no_vocals(tmp_path) -> None:
    """No vocals → the assertion is HONESTY, not a hallucination count.

    Whisper's hallucination count on near-silence varies by environment
    (different CPU arch → slightly different separation output → different
    decode; observed 2 locally vs 3 on CI, PR #9 review blocker). The
    stable properties are: whatever appears is flagged, every timestamp
    stays inside the timeline domain, and the document validates.
    """
    wav = tmp_path / "clicks.wav"
    _write_stereo_wav(wav, _click_track(15.0))
    library, track_id = _run_chain(wav, tmp_path)
    doc = json.loads((library / track_id / "lyrics.json").read_text())
    _validate(doc, "lyrics.schema.json")
    assert doc["mode"] == "transcribed"
    duration = json.loads(
        (library / track_id / "audio_features.json").read_text()
    )["duration_seconds"]
    for line in doc["lines"]:
        assert line["flags"], "a line hallucinated from silence must be flagged"
        assert 0.0 <= line["start_seconds"] <= line["end_seconds"] <= duration
        for w in line["words"]:
            assert 0.0 <= w["start_seconds"] <= w["end_seconds"] <= duration
    assert 0.0 <= doc["coverage"]["vocal_activity_covered_ratio"] <= 1.0
    assert doc["coverage"]["lines_flagged_ratio"] in (0.0, 1.0)  # none or all


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng not installed")
def test_espeak_ground_truth_transcription(tmp_path) -> None:
    """Synthesized speech over a click bed survives separation+transcription."""
    speech_raw = tmp_path / "speech_raw.wav"
    subprocess.run(
        ["espeak-ng", "-v", "en-us", "-s", "130", "-w", str(speech_raw),
         "the quick brown fox jumps over the lazy dog"],
        check=True, capture_output=True,
    )
    speech44 = tmp_path / "speech44.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(speech_raw),
         "-ar", str(SAMPLE_RATE), "-ac", "2", str(speech44)],
        check=True, capture_output=True,
    )
    speech, _ = sf.read(speech44, dtype="float32", always_2d=True)
    speech_mono = speech.mean(axis=1)

    bed = _click_track(12.0) * 0.25
    start = 3 * SAMPLE_RATE
    n = min(len(speech_mono), len(bed) - start)
    mix = bed.copy()
    mix[start : start + n] += 0.6 * speech_mono[:n]

    wav = tmp_path / "speech_mix.wav"
    _write_stereo_wav(wav, mix)
    library, track_id = _run_chain(wav, tmp_path)
    doc = json.loads((library / track_id / "lyrics.json").read_text())
    _validate(doc, "lyrics.schema.json")

    text = " ".join(l["text"].lower() for l in doc["lines"])
    hits = sum(w in text for w in ("quick", "brown", "fox", "lazy", "dog"))
    assert hits >= 3, f"expected ground-truth words, got: {text!r}"
    speech_end = 3.0 + n / SAMPLE_RATE
    for line in doc["lines"]:
        assert 1.0 <= line["start_seconds"] <= speech_end + 2.0
