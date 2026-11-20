"""P1 operational robustness (ORCH-4 rest, ORCH-7/SCH-4/VID-7, SEC-9):
per-stage run locks, the append-only API ledger, interruption recording,
and the atomic copy_video.

Fast: generated tones/clips, stubbed engines, zero API calls.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mrw.cli import app
from mrw.library import Library
from test_m2_field_fixes import _ingest, _make_wav

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)

runner = CliRunner()


def _manifest(library: Path, track_id: str) -> dict:
    return json.loads((library / track_id / "manifest.json").read_text())


# --- per-stage run lock ---------------------------------------------------------


@pytest.mark.parametrize("stage", ["stems", "features", "lyrics", "video"])
def test_stage_already_running_is_refused(tmp_path, stage):
    wav = tmp_path / "t.wav"
    _make_wav(wav)
    library = tmp_path / "lib"
    track_id = _ingest(wav, library)
    before = (library / track_id / "manifest.json").read_bytes()
    with Library(library).stage_lock(track_id, stage):
        result = runner.invoke(app, [stage, track_id, "--library", str(library)])
    assert result.exit_code == 2, result.output
    assert f"stage {stage} already running for {track_id}" in result.output
    assert (library / track_id / "manifest.json").read_bytes() == before


def test_stage_lock_is_per_stage_and_released(tmp_path):
    wav = tmp_path / "t.wav"
    _make_wav(wav)
    library = tmp_path / "lib"
    track_id = _ingest(wav, library)
    lib = Library(library)
    with lib.stage_lock(track_id, "video"):
        with lib.stage_lock(track_id, "stems"):  # other stages unaffected
            pass
    with lib.stage_lock(track_id, "video"):  # released on exit
        pass
    assert (library / track_id / ".video.lock").is_file()
