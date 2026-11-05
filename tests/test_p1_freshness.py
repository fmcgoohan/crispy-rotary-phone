"""P1-1 (DET-1): new upstream output marks dependent documents stale."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mrw import config, versions
from mrw.cli import app
from mrw.library import mark_dependents_stale
from mrw.models import Manifest
from test_m3_smoke import _click_track, _fabricate_track

pytestmark = pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None,
    reason="ffmpeg/ffprobe not available",
)

runner = CliRunner()


def _ok_lyrics(library: Path, track_id: str) -> None:
    mpath = library / track_id / "manifest.json"
    m = json.loads(mpath.read_text())
    m["documents"]["lyrics"] = {
        "status": "ok", "path": "lyrics.json",
        "config_hash": config.stage_hash(config.Config().lyrics),
        "algorithm_version": versions.current("lyrics")}
    mpath.write_text(json.dumps(m))


def _doc_status(library, track_id, name):
    return json.loads((library / track_id / "manifest.json").read_text())[
        "documents"][name]["status"]


def test_changed_features_mark_lyrics_stale(tmp_path):
    mix = _click_track(6.0)
    library = tmp_path / "lib"
    track_id = _fabricate_track(library, mix, {"drums": mix, "vocals": mix}, tmp_path)
    lib = ["--library", str(library)]
    assert runner.invoke(app, ["features", track_id, *lib]).exit_code == 0
    _ok_lyrics(library, track_id)

    # Same config, same output: a no-op leaves lyrics alone.
    assert "no-op" in runner.invoke(app, ["features", track_id, *lib]).output
    assert _doc_status(library, track_id, "lyrics") == "ok"

    # New vocal-activity settings change audio_features.json → lyrics, which
    # was computed from the old regions, goes stale (and shows so).
    cfg = tmp_path / "mrw.toml"
    cfg.write_text("[features]\nvocal_min_gap_seconds = 1.0\nvocal_enter_db = -20.0\n")
    result = runner.invoke(app, ["features", track_id, *lib, "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert _doc_status(library, track_id, "lyrics") == "stale"
    out = runner.invoke(app, ["status", *lib, "--config", str(cfg)]).output
    assert "lyrics=stale" in out


def test_mark_dependents_stale_is_transitive_and_ok_only():
    pending = {"status": "pending"}
    ok = {"status": "ok", "path": "x.json"}
    m = Manifest.model_validate({
        "track_id": "0" * 16, "title": "t", "has_video": False,
        "created_at": "2026-01-01T00:00:00Z",
        "documents": {"source": ok, "audio_features": ok, "lyrics": ok,
                      "video": {"status": "not_applicable"}, "structure": ok}})
    assert mark_dependents_stale(m, "stems") == ["audio_features", "lyrics", "structure"]
    assert m.documents.audio_features.status == "stale"
    assert m.documents.audio_features.path == "x.json"  # entry otherwise kept
    assert m.documents.video.status == "not_applicable"
    assert m.documents.source.status == "ok"
    assert mark_dependents_stale(m, "stems") == []  # idempotent
