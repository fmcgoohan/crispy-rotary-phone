"""P1 `mrw ingest --update` (ORCH-6/ARCH-9/DET-4/VID-10/LYR-13).

Fast: generated tones and a generated 2 s video clip; no neural stack.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mrw.cli import app
from test_m2_field_fixes import _ingest, _make_wav
from test_p1_inputs import _tiny_video

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)

runner = CliRunner()


def _manifest(library: Path, track_id: str) -> dict:
    return json.loads((library / track_id / "manifest.json").read_text())


def _source(library: Path, track_id: str) -> dict:
    return json.loads((library / track_id / "source.json").read_text())


def _snapshot(track_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(track_dir.iterdir()) if p.is_file()}


def _audio_track(tmp_path: Path) -> tuple[Path, Path, str]:
    wav = tmp_path / "song.wav"
    _make_wav(wav)
    library = tmp_path / "lib"
    return wav, library, _ingest(wav, library)


def _mark_downstream_ok(library: Path, track_id: str) -> None:
    mpath = library / track_id / "manifest.json"
    m = json.loads(mpath.read_text())
    for name in ("audio_features", "lyrics", "structure"):
        m["documents"][name] = {"status": "ok", "path": f"{name}.json",
                                "content_sha256": "1" * 64}
    m["stems"] = {"status": "ok", "retained": True, "content_sha256": "2" * 64}
    mpath.write_text(json.dumps(m))


def test_plain_reingest_names_every_ignored_option(tmp_path):
    mp4 = tmp_path / "clip.mp4"
    _tiny_video(mp4, "red")
    library = tmp_path / "lib"
    track_id = _ingest(mp4, library)
    before = _snapshot(library / track_id)

    lrc = tmp_path / "words.lrc"
    lrc.write_text("[00:01.00]hello there\n")
    cfg = tmp_path / "mrw.toml"
    cfg.write_text("[ingest]\ncopy_video = true\n")
    result = runner.invoke(app, [
        "ingest", str(mp4), "--library", str(library), "--config", str(cfg),
        "--lyrics", str(lrc), "--title", "New Name"])
    assert result.exit_code == 0, result.output
    assert "no-op" in result.output
    assert f"ignored --lyrics {lrc} (no lyrics are registered)" in result.output
    assert "ignored --title 'New Name'" in result.output
    assert "ignored ingest.copy_video = true" in result.output
    assert result.output.count("--update to apply") == 3
    assert _snapshot(library / track_id) == before  # truly a no-op


def test_identical_reingest_stays_a_silent_noop(tmp_path):
    wav, library, track_id = _audio_track(tmp_path)
    result = runner.invoke(app, ["ingest", str(wav), "--library", str(library)])
    assert result.exit_code == 0
    assert "no-op" in result.output and "warning" not in result.output


def test_update_registers_lyrics_and_resets_only_lyrics_and_structure(tmp_path):
    wav, library, track_id = _audio_track(tmp_path)
    _mark_downstream_ok(library, track_id)
    before = _manifest(library, track_id)

    txt = tmp_path / "words.txt"
    txt.write_text("hello there\ngeneral words\n")
    lib = ["--library", str(library)]
    result = runner.invoke(app, ["ingest", str(wav), *lib, "--lyrics", str(txt), "--update"])
    assert result.exit_code == 0, result.output
    assert "registered lyrics words.txt (txt)" in result.output

    track_dir = library / track_id
    source = _source(library, track_id)
    assert source["lyrics_input"]["path"] == "lyrics_input.txt"
    assert source["lyrics_input"]["sha256"] == hashlib.sha256(txt.read_bytes()).hexdigest()
    m = _manifest(library, track_id)
    assert m["documents"]["source"]["content_sha256"] == hashlib.sha256(
        (track_dir / "source.json").read_bytes()).hexdigest()
    assert m["documents"]["lyrics"] == {"status": "pending"}
    assert m["documents"]["structure"] == {"status": "pending"}
    assert m["documents"]["audio_features"] == before["documents"]["audio_features"]
    assert m["stems"] == before["stems"]

    # Same lyrics again: nothing to update.
    result = runner.invoke(app, ["ingest", str(wav), *lib, "--lyrics", str(txt), "--update"])
    assert "nothing to update" in result.output

    # Switching to an .lrc replaces the registered file (old one removed).
    lrc = tmp_path / "words.lrc"
    lrc.write_text("[00:01.00]hello there\n")
    result = runner.invoke(app, ["ingest", str(wav), *lib, "--lyrics", str(lrc), "--update"])
    assert result.exit_code == 0, result.output
    assert _source(library, track_id)["lyrics_input"]["format"] == "lrc"
    assert (track_dir / "lyrics_input.lrc").is_file()
    assert not (track_dir / "lyrics_input.txt").exists()


def test_update_title_touches_only_the_title(tmp_path):
    wav, library, track_id = _audio_track(tmp_path)
    source_before = (library / track_id / "source.json").read_bytes()
    before = _manifest(library, track_id)
    result = runner.invoke(app, ["ingest", str(wav), "--library", str(library),
                                 "--title", "Better Title", "--update"])
    assert result.exit_code == 0, result.output
    after = _manifest(library, track_id)
    assert after["title"] == "Better Title"
    assert after["documents"] == before["documents"]
    assert (library / track_id / "source.json").read_bytes() == source_before


def test_update_regenerates_missing_flac(tmp_path):
    wav, library, track_id = _audio_track(tmp_path)
    flac = library / track_id / "source_audio.flac"
    original = flac.read_bytes()
    flac.unlink()
    result = runner.invoke(app, ["ingest", str(wav), "--library", str(library)])
    assert "missing source_audio.flac" in result.output and not flac.exists()
    result = runner.invoke(app, ["ingest", str(wav), "--library", str(library), "--update"])
    assert result.exit_code == 0, result.output
    assert "regenerated source_audio.flac" in result.output
    assert flac.read_bytes() == original


def test_moved_video_relinked_and_copied_with_update(tmp_path):
    mp4 = tmp_path / "clip.mp4"
    _tiny_video(mp4, "red")
    library = tmp_path / "lib"
    track_id = _ingest(mp4, library)
    moved = tmp_path / "elsewhere" / "clip.mp4"
    moved.parent.mkdir()
    mp4.rename(moved)
    lib = ["--library", str(library)]

    result = runner.invoke(app, ["video", track_id, *lib])
    assert result.exit_code == 2 and "--update" in result.output

    # The remedy the message names: relink from the new location, keeping a
    # verified copy in the library.
    cfg = tmp_path / "mrw.toml"
    cfg.write_text("[ingest]\ncopy_video = true\n")
    result = runner.invoke(
        app, ["ingest", str(moved), *lib, "--config", str(cfg), "--update"])
    assert result.exit_code == 0, result.output
    assert "copied the video into the library" in result.output
    copy = library / track_id / "source_video.mp4"
    assert copy.read_bytes() == moved.read_bytes()
    source = _source(library, track_id)
    assert source["file"]["original_path"] == str(moved)
    out = runner.invoke(app, ["status", *lib, "--config", str(cfg)]).output
    assert "source=ok" in out  # ingest config now recorded as current

    moved.unlink()  # the library copy alone is enough now
    result = runner.invoke(app, ["video", track_id, *lib])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("name", ["words.srt", "words.docx", "words"])
def test_unsupported_lyrics_extension_is_bad_invocation(tmp_path, name):
    wav = tmp_path / "song.wav"
    _make_wav(wav)
    lyrics = tmp_path / name
    lyrics.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n")
    library = tmp_path / "lib"
    result = runner.invoke(
        app, ["ingest", str(wav), "--library", str(library), "--lyrics", str(lyrics)])
    assert result.exit_code == 2
    assert "must be a .lrc or .txt" in result.output
    assert not library.exists()  # refused before any work


# --- crash safety of --update (review A-4) ---------------------------------------


def test_failed_video_copy_leaves_registered_lyrics_untouched(tmp_path, monkeypatch):
    import mrw.ingest
    from mrw.ingest import IngestError

    mp4 = tmp_path / "clip.mp4"
    _tiny_video(mp4, "red")
    library = tmp_path / "lib"
    txt = tmp_path / "words.txt"
    txt.write_text("hello there\n")
    lib = ["--library", str(library)]
    result = runner.invoke(app, ["ingest", str(mp4), *lib, "--lyrics", str(txt)])
    assert result.exit_code == 0, result.output
    track_id = next(p.name for p in library.iterdir() if p.is_dir())
    track_dir = library / track_id
    m = json.loads((track_dir / "manifest.json").read_text())
    m["documents"]["lyrics"] = {"status": "ok", "path": "lyrics.json",
                                "content_sha256": "1" * 64}
    (track_dir / "manifest.json").write_text(json.dumps(m))
    before = _snapshot(track_dir)

    def enospc(media_path, track_dir, expected_sha):
        raise IngestError("No space left on device")

    monkeypatch.setattr(mrw.ingest, "copy_video_atomic", enospc)
    lrc = tmp_path / "words.lrc"
    lrc.write_text("[00:01.00]hello there\n")
    cfg = tmp_path / "mrw.toml"
    cfg.write_text("[ingest]\ncopy_video = true\n")
    result = runner.invoke(app, ["ingest", str(mp4), *lib, "--config", str(cfg),
                                 "--lyrics", str(lrc), "--update"])
    assert result.exit_code == 1, result.output
    # Nothing registered changed: the copy runs before any lyrics work.
    assert _snapshot(track_dir) == before


def test_interrupt_before_lyrics_move_is_detected_and_repaired(tmp_path, monkeypatch):
    import mrw.ingest
    from test_m3_smoke import _click_track, _fabricate_track

    mix = _click_track(3.0)
    library = tmp_path / "lib"
    track_id = _fabricate_track(library, mix, {"vocals": mix}, tmp_path)
    lib = ["--library", str(library)]
    assert runner.invoke(app, ["features", track_id, *lib]).exit_code == 0
    track_dir = library / track_id
    media = Path(_source(library, track_id)["file"]["original_path"])
    txt = tmp_path / "words.txt"
    txt.write_text("hello there\n")
    assert runner.invoke(
        app, ["ingest", str(media), *lib, "--lyrics", str(txt), "--update"]).exit_code == 0

    real_replace = mrw.ingest.os.replace

    def interrupted(src, dst):  # source.json rewritten; lyrics not moved yet
        if Path(src).name == ".lyrics_input.tmp":
            raise KeyboardInterrupt
        return real_replace(src, dst)

    monkeypatch.setattr(mrw.ingest.os, "replace", interrupted)
    lrc = tmp_path / "words.lrc"
    lrc.write_text("[00:01.00]hello there\n")
    args = ["ingest", str(media), *lib, "--lyrics", str(lrc), "--update"]
    assert runner.invoke(app, args).exit_code == 130
    monkeypatch.undo()
    assert not (track_dir / ".lyrics_input.tmp").exists()
    assert _source(library, track_id)["lyrics_input"]["path"] == "lyrics_input.lrc"
    assert not (track_dir / "lyrics_input.lrc").exists()
    assert _manifest(library, track_id)["documents"]["lyrics"] == {"status": "pending"}

    # The lyrics stage refuses the mismatch before any heavy work...
    result = runner.invoke(app, ["lyrics", track_id, *lib])
    assert result.exit_code == 2, result.output
    assert "does not match the lyrics registered" in result.output
    assert _manifest(library, track_id)["documents"]["lyrics"] == {"status": "pending"}

    # ...and re-running the same --update repairs it.
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "registered lyrics words.lrc (lrc)" in result.output
    assert (track_dir / "lyrics_input.lrc").read_bytes() == lrc.read_bytes()
    assert not (track_dir / "lyrics_input.txt").exists()
    m = _manifest(library, track_id)
    assert m["documents"]["source"]["content_sha256"] == hashlib.sha256(
        (track_dir / "source.json").read_bytes()).hexdigest()
