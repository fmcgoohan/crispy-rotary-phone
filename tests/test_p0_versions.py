"""P0-0 / P0-6: code changes mark documents stale (mrw/versions.py), and
`mrw status` derives stale for every stage, lyrics and video included."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mrw import config, versions
from mrw.cli import app
from test_m3_smoke import _click_track, _fabricate_track

pytestmark = pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None,
    reason="ffmpeg/ffprobe not available",
)

runner = CliRunner()


def _status(library: Path, *extra: str) -> str:
    result = runner.invoke(app, ["status", "--library", str(library), *extra])
    assert result.exit_code == 0, result.output
    return result.output


def test_algorithm_version_bump_marks_stale_and_reruns(tmp_path, monkeypatch):
    mix = _click_track(6.0)
    library = tmp_path / "lib"
    track_id = _fabricate_track(library, mix, {"drums": mix}, tmp_path)

    result = runner.invoke(app, ["features", track_id, "--library", str(library)])
    assert result.exit_code == 0, result.output
    entry = json.loads((library / track_id / "manifest.json").read_text())[
        "documents"]["audio_features"]
    assert entry["algorithm_version"] == versions.current("audio_features")
    assert "features=ok" in _status(library)

    bumped = dict(versions.ALGORITHM_VERSIONS)
    bumped["audio_features"] += 1
    monkeypatch.setattr(versions, "ALGORITHM_VERSIONS", bumped)
    assert "features=stale" in _status(library)

    result = runner.invoke(app, ["features", track_id, "--library", str(library)])
    assert result.exit_code == 0, result.output
    assert "no-op" not in result.output
    assert "features=ok" in _status(library)


def test_entry_without_version_counts_as_version_1():
    assert versions.is_current("lyrics", None) == (versions.current("lyrics") == 1)
    assert versions.is_current("structure", None)  # unversioned stage


def test_status_shows_lyrics_and_video_stale_on_config_change(tmp_path):
    mix = _click_track(4.0)
    library = tmp_path / "lib"
    track_id = _fabricate_track(library, mix, {"drums": mix}, tmp_path)
    mpath = library / track_id / "manifest.json"
    manifest = json.loads(mpath.read_text())
    defaults = config.Config()
    manifest["documents"]["lyrics"] = {
        "status": "ok", "path": "lyrics.json",
        "config_hash": config.stage_hash(defaults.lyrics),
        "algorithm_version": versions.current("lyrics")}
    manifest["documents"]["video"] = {
        "status": "ok", "path": "video.json",
        "config_hash": config.stage_hash(defaults.video),
        "algorithm_version": versions.current("video")}
    mpath.write_text(json.dumps(manifest))

    out = _status(library)
    assert "lyrics=ok" in out and "video=ok" in out

    cfg = tmp_path / "changed.toml"
    cfg.write_text("[lyrics]\nmin_anchor_score = 0.7\n[video]\ndetector_threshold = 30.0\n")
    out = _status(library, "--config", str(cfg))
    assert "lyrics=stale" in out and "video=stale" in out
