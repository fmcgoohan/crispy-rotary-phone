"""OQ-7 resolution (lyrics 1.3.0): lyrics.json records whether the Whisper
model was pinned or a local directory, never a machine-specific absolute
path, and a local model's weights changing in place marks lyrics stale.

Stubbed engine (no Whisper); needs ffmpeg for ingest + features.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mrw import inputs
from mrw.cli import app
from mrw.whisper_models import (
    WHISPER_MODELS,
    local_model_fingerprint,
    model_provenance,
)
from test_m3_smoke import _click_track
from test_m4_smoke import _bursty_vocals, _fabricate, _seg, _stub_engine, _validate

pytestmark = pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None,
    reason="ffmpeg/ffprobe not available",
)

runner = CliRunner()


def _model_dir(parent: Path, name: str = "my-whisper") -> Path:
    d = parent / name
    (d / "sub").mkdir(parents=True)
    (d / "model.bin").write_bytes(b"\x00weights-v1")
    (d / "config.json").write_text('{"a": 1}')
    (d / "sub" / "vocabulary.txt").write_text("hello\nworld\n")
    return d


# --- fingerprint unit tests ---------------------------------------------------


def test_fingerprint_is_content_based_and_path_independent(tmp_path) -> None:
    a = _model_dir(tmp_path / "one")
    b = _model_dir(tmp_path / "two" / "deeper", name="other-name")
    fp = local_model_fingerprint(a)
    assert len(fp) == 64 and all(c in "0123456789abcdef" for c in fp)
    assert local_model_fingerprint(b) == fp  # same files, different location

    # Dot-entries carry no weights and are ignored (.DS_Store, .cache).
    (a / ".DS_Store").write_bytes(b"junk")
    (a / ".cache").mkdir()
    (a / ".cache" / "x").write_text("y")
    assert local_model_fingerprint(a) == fp

    # Any content change, rename or added file changes it.
    (a / "model.bin").write_bytes(b"\x00weights-v2")
    fp2 = local_model_fingerprint(a)
    assert fp2 != fp
    (a / "sub" / "vocabulary.txt").rename(a / "sub" / "vocab.txt")
    fp3 = local_model_fingerprint(a)
    assert fp3 not in (fp, fp2)
    (a / "extra.json").write_text("{}")
    assert local_model_fingerprint(a) not in (fp, fp2, fp3)


def test_fingerprint_follows_file_symlinks(tmp_path) -> None:
    """A Hugging Face cache snapshot is a tree of symlinks into blobs/."""
    real = _model_dir(tmp_path / "real")
    snap = tmp_path / "snapshot"
    (snap / "sub").mkdir(parents=True)
    for rel in ("model.bin", "config.json", "sub/vocabulary.txt"):
        os.symlink(real / rel, snap / rel)
    assert local_model_fingerprint(snap) == local_model_fingerprint(real)


def test_model_provenance_pinned_and_local(tmp_path) -> None:
    pinned = model_provenance("small")
    assert pinned.source == "pinned"
    assert pinned.model == "small"
    assert pinned.revision == WHISPER_MODELS["small"][1]
    assert pinned.fingerprint is None

    d = _model_dir(tmp_path)
    local = model_provenance(str(d))
    assert local.source == "local"
    assert local.model == "my-whisper"
    assert local.revision is None
    assert local.fingerprint == local_model_fingerprint(d)


def test_lyrics_inputs_hash_unchanged_for_pinned_models() -> None:
    """The fingerprint joins inputs_sha256 only for a local model, so every
    existing pinned-model entry keeps its recorded value."""
    from test_p1_inputs import _manifest_model

    m = _manifest_model(content_sha256="d" * 64)
    src = {"lyrics_input": {"sha256": "9" * 64}}
    base = inputs.current_inputs("lyrics", m, src)
    assert inputs.current_inputs("lyrics", m, src, model_fingerprint=None) == base
    with_fp = inputs.current_inputs("lyrics", m, src, model_fingerprint="f" * 64)
    assert with_fp != base
    assert inputs.current_inputs(
        "lyrics", m, src, model_fingerprint="e" * 64) != with_fp


# --- documents with a stubbed engine -----------------------------------------


def _segments():
    return [_seg("la la la", 2.2, 3.6)]


def test_local_model_recorded_without_absolute_path(tmp_path, monkeypatch) -> None:
    mix = _click_track(10.0)
    vocals = _bursty_vocals(10.0)
    library = tmp_path / "lib"
    track_id = _fabricate(library, mix + vocals * 0.5, {"vocals": vocals}, tmp_path)

    model_dir = _model_dir(tmp_path / "models")
    cfg = tmp_path / "local.toml"
    cfg.write_text(f'[lyrics]\nmodel = "{model_dir}"\n')
    lib = ["--library", str(library), "--config", str(cfg)]

    _stub_engine(monkeypatch, _segments())
    result = runner.invoke(app, ["lyrics", track_id, *lib])
    assert result.exit_code == 0, result.output
    path = library / track_id / "lyrics.json"
    raw = path.read_text()
    doc = json.loads(raw)
    _validate(doc, "lyrics.schema.json")
    fp = local_model_fingerprint(model_dir)
    assert doc["engine"] == {
        "name": "faster-whisper",
        "model": "my-whisper",
        "language_source": "detected_vocal_window",
        "model_source": "local",
        "model_fingerprint": fp,
    }
    # No absolute path anywhere in the document (D1/D2): not the model
    # directory, not any parent of it, not the library.
    assert str(model_dir) not in raw
    assert str(tmp_path) not in raw
    assert '"/' not in raw

    # Unchanged weights: a no-op.
    out = runner.invoke(app, ["lyrics", track_id, *lib]).output
    assert "no-op" in out or "up to date" in out, out
    status = runner.invoke(app, ["status", *lib]).output
    assert "lyrics=ok" in status, status

    # Weights replaced in place (same path, same config_hash): lyrics is
    # stale in status and the stage re-runs, recording the new fingerprint.
    (model_dir / "model.bin").write_bytes(b"\x00weights-v2")
    status = runner.invoke(app, ["status", *lib]).output
    assert "lyrics=stale" in status, status
    result = runner.invoke(app, ["lyrics", track_id, *lib])
    assert result.exit_code == 0, result.output
    assert "no-op" not in result.output and "up to date" not in result.output
    doc2 = json.loads(path.read_text())
    assert doc2["engine"]["model_fingerprint"] == local_model_fingerprint(model_dir)
    assert doc2["engine"]["model_fingerprint"] != fp


def test_local_model_document_identical_across_locations(tmp_path, monkeypatch) -> None:
    """Same weights in two different directories (two machines' layouts)
    with the same basename give byte-identical lyrics.json."""
    mix = _click_track(10.0)
    vocals = _bursty_vocals(10.0)
    _stub_engine(monkeypatch, _segments())
    outputs = []
    for where in ("machine-a", "machine-b/elsewhere"):
        root = tmp_path / where
        root.mkdir(parents=True)
        library = root / "lib"
        track_id = _fabricate(library, mix + vocals * 0.5, {"vocals": vocals}, root)
        model_dir = _model_dir(root / "models")
        cfg = root / "local.toml"
        cfg.write_text(f'[lyrics]\nmodel = "{model_dir}"\n')
        result = runner.invoke(
            app, ["lyrics", track_id, "--library", str(library), "--config", str(cfg)]
        )
        assert result.exit_code == 0, result.output
        outputs.append((library / track_id / "lyrics.json").read_bytes())
    assert outputs[0] == outputs[1]


def test_unmatched_words_sorted_on_rounded_values_with_total_key():
    import random

    from mrw.align import WordObs
    from mrw.lyrics import _words_from_obs, sort_unmatched_words

    # 1.0004 and 1.0001 both publish as 1.0: raw-time sorting would order
    # them by sub-ms jitter; the published order must depend only on
    # published values (start, end, text).
    raw = [WordObs("b", 1.0004, 1.2), WordObs("a", 1.0001, 1.2),
           WordObs("c", 0.5, 0.7), WordObs("a", 1.0002, 1.1)]
    expected = [("c", 0.5, 0.7), ("a", 1.0, 1.1), ("a", 1.0, 1.2), ("b", 1.0, 1.2)]
    for seed in range(5):
        shuffled = raw[:]
        random.Random(seed).shuffle(shuffled)
        out = sort_unmatched_words(_words_from_obs(shuffled))
        assert [(w.text, w.start_seconds, w.end_seconds) for w in out] == expected
