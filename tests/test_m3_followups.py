"""Review 006 follow-ups (F-1 phase-fit window config, F-2 lazy torch, and
the config-evolution/staleness behavior from note 2).

Test economy per 006 note 2: the staleness test uses the new
phase_fit_window_seconds field as its knob, proving F-1's config_hash
membership and the derived stale display in one test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from mrw.cli import app
from test_m3_smoke import _click_track, _fabricate_track, _tone, _write_stereo_wav

pytestmark = pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None,
    reason="ffmpeg/ffprobe not available",
)

runner = CliRunner()


def test_phase_window_in_config_hash_and_stale_display(tmp_path: Path) -> None:
    # F-1 + note 2: changing phase_fit_window_seconds must shift the
    # features config_hash (stale display derives from it; re-run is not a
    # no-op), because the window is an output-affecting tuning knob.
    mix = _click_track(8.0)
    library = tmp_path / "lib"
    track_id = _fabricate_track(library, mix, {"drums": mix}, tmp_path)

    result = runner.invoke(app, ["features", track_id, "--library", str(library)])
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["status", "--library", str(library)])
    assert "features=ok" in result.output

    cfg = tmp_path / "mrw.toml"
    cfg.write_text("[features]\nphase_fit_window_seconds = 0.05\n")

    # Derived stale display: manifest untouched, status compares hashes.
    result = runner.invoke(
        app, ["status", "--library", str(library), "--config", str(cfg)]
    )
    assert "features=stale" in result.output
    manifest = json.loads((library / track_id / "manifest.json").read_text())
    assert manifest["documents"]["audio_features"]["status"] == "ok"  # not rewritten

    # Re-run under the new config regenerates (not a no-op) and goes ok.
    result = runner.invoke(
        app, ["features", track_id, "--library", str(library), "--config", str(cfg)]
    )
    assert result.exit_code == 0, result.output
    assert "no-op" not in result.output
    result = runner.invoke(
        app, ["status", "--library", str(library), "--config", str(cfg)]
    )
    assert "features=stale" not in result.output
    assert "features=ok" in result.output


def test_missing_source_reported_before_torch_needed(
    tmp_path: Path, monkeypatch
) -> None:
    # F-2 / T5: with torch unimportable, a missing source file must still be
    # reported as exactly that — prerequisite validation precedes heavy
    # imports (explicit device, so 'auto' torch probing is not involved).
    wav = tmp_path / "t.wav"
    _write_stereo_wav(wav, _tone(2.0, 330.0))
    library = tmp_path / "lib"
    result = runner.invoke(app, ["ingest", str(wav), "--library", str(library)])
    assert result.exit_code == 0, result.output
    track_id = next(p.name for p in library.iterdir() if p.is_dir())
    (library / track_id / "source_audio.flac").unlink()

    monkeypatch.setitem(sys.modules, "torch", None)  # import torch → ImportError
    cfg = tmp_path / "mrw.toml"
    cfg.write_text('[stems]\ndevice = "cpu"\n')
    result = runner.invoke(
        app, ["stems", track_id, "--library", str(library), "--config", str(cfg)]
    )
    assert result.exit_code == 1
    assert "source_audio.flac" in result.output
    assert "torch" not in result.output
