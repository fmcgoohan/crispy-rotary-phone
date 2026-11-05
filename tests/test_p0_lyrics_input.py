"""P0-3 / P0-4: LRC variants and lyrics-file decoding."""

from __future__ import annotations

import codecs

import pytest

from mrw import lrc
from mrw.lyrics import parse_supplied_lyrics


def _hints(raw):
    lines, _ = parse_supplied_lyrics(raw)
    return [(l.text, l.hint_seconds, l.source_line_index) for l in lines]


def test_compressed_lrc_line_yields_one_line_per_tag_in_time_order():
    raw = (
        "[00:10.00]Verse line\n"
        "[00:45.00][01:30.00][02:40.00]Neon reverie, don't wake me up\n"
        "[01:00.00]Second verse\n"
    )
    assert _hints(raw) == [
        ("Verse line", 10.0, 0),
        ("Neon reverie, don't wake me up", 45.0, 1),
        ("Second verse", 60.0, 2),
        ("Neon reverie, don't wake me up", 90.0, 1),
        ("Neon reverie, don't wake me up", 160.0, 1),
    ]


@pytest.mark.parametrize("tag,seconds", [
    ("[01:02]", 62.0), ("[01:02.5]", 62.5), ("[01:02.25]", 62.25),
    ("[01:02.250]", 62.25), ("[01:02:25]", 62.25),
])
def test_time_tag_variants(tag, seconds):
    assert _hints(f"{tag}Hello there\n") == [("Hello there", seconds, 0)]


def test_id_tags_are_skipped_and_offset_applied():
    raw = "[ar:Someone]\n[ti:Song]\n[by:me]\n[offset:+500]\n[00:10.00]Line one\n[Chorus]\n"
    lines, markup = parse_supplied_lyrics(raw)
    assert [(l.text, l.hint_seconds) for l in lines] == [("Line one", 9.5)]
    assert [m.text for m in markup] == ["[Chorus]"]  # real markup kept


def test_section_label_with_colon_is_markup_not_id_tag():
    _, markup = parse_supplied_lyrics("[Chorus: Someone]\nla la\n")
    assert [(m.text, m.label) for m in markup] == [("[Chorus: Someone]", "chorus")]


def test_plain_text_order_untouched():
    assert [t for t, _, _ in _hints("b line\na line\n")] == ["b line", "a line"]


@pytest.mark.parametrize("data", [
    codecs.BOM_UTF8 + "[00:01.00]Café ’n’ crème\n".encode("utf-8"),
    codecs.BOM_UTF16_LE + "[00:01.00]Café ’n’ crème\n".encode("utf-16-le"),
    codecs.BOM_UTF16_BE + "[00:01.00]Café ’n’ crème\n".encode("utf-16-be"),
    "[00:01.00]Café ’n’ crème\n".encode("cp1252"),
    "[00:01.00]Café ’n’ crème\n".encode("utf-8"),
])
def test_decode_text_handles_bom_utf16_and_cp1252(data):
    text = lrc.decode_text(data)
    assert text == "[00:01.00]Café ’n’ crème\n"
    assert _hints(text) == [("Café ’n’ crème", 1.0, 0)]


def test_decode_text_never_fails():
    assert lrc.decode_text(bytes(range(256)))  # latin-1 fallback


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None,
                    reason="ffmpeg/ffprobe not available")
def test_ingest_sees_timestamps_behind_a_bom(tmp_path):
    import json

    from typer.testing import CliRunner

    from mrw.cli import app
    from test_m3_smoke import _tone, _write_stereo_wav

    wav = tmp_path / "t.wav"
    _write_stereo_wav(wav, _tone(1.0, 440.0))
    lyr = tmp_path / "t.lrc"
    lyr.write_bytes(codecs.BOM_UTF8 + b"[00:00.50]Hello\n[ar:Someone]\n")
    library = tmp_path / "lib"
    result = CliRunner().invoke(
        app, ["ingest", str(wav), "--lyrics", str(lyr), "--library", str(library)])
    assert result.exit_code == 0, result.output
    track = next(p for p in library.iterdir() if p.is_dir())
    li = json.loads((track / "source.json").read_text())["lyrics_input"]
    assert li["has_timestamps"] is True


def test_whisper_model_resolves_from_cache_only(tmp_path, monkeypatch):
    # P0-5 (LYR-3): an uncached model fails with a fetch hint and makes no
    # network connection at all.
    import socket

    from mrw.lyrics import LyricsError, resolve_whisper_model

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE",
                        str(tmp_path / "empty-cache"))

    def no_network(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    with pytest.raises(LyricsError, match="mrw models fetch"):
        resolve_whisper_model("small")
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    assert resolve_whisper_model(str(model_dir)) == str(model_dir)
