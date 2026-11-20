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


# --- append-only API ledger -----------------------------------------------------


class _Backend:
    """Stub caption backend: pays for every call; optionally raises `boom`
    after paying (a failure or an interruption mid-captioning)."""

    name = "anthropic"

    def __init__(self, boom: BaseException | None = None):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.boom = boom

    def caption(self, frame_jpegs, frame_sha256s):
        from mrw.models import ShotCaption

        self.calls += 1
        self.input_tokens += 1000
        self.output_tokens += 100
        if self.boom is not None:
            raise self.boom
        return ShotCaption(text="paid caption", tags=["paid"])


def _use_backend(monkeypatch, backend: _Backend) -> None:
    import mrw.captions
    from mrw.captions import CaptionCache

    monkeypatch.setattr(
        mrw.captions, "make_backend", lambda vcfg, cache_dir: (backend, CaptionCache(cache_dir))
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-used")


def _video_track(tmp_path: Path) -> tuple[Path, str]:
    from test_p1_inputs import _tiny_video

    mp4 = tmp_path / "clip.mp4"
    _tiny_video(mp4, "red")
    library = tmp_path / "lib"
    return library, _ingest(mp4, library)


def _costs_line(library: Path) -> str:
    result = runner.invoke(app, ["costs", "--library", str(library)])
    assert result.exit_code == 0, result.output
    return result.output


def test_ledger_survives_fail_retry_and_null_rerun(tmp_path, monkeypatch):
    library, track_id = _video_track(tmp_path)
    paid = tmp_path / "paid.toml"
    paid.write_text('[video]\ncaption_backend = "anthropic"\n')
    args = ["video", track_id, "--library", str(library)]

    _use_backend(monkeypatch, _Backend(boom=RuntimeError("API exploded")))
    assert runner.invoke(app, [*args, "--config", str(paid)]).exit_code == 1
    _use_backend(monkeypatch, _Backend())
    assert runner.invoke(app, [*args, "--config", str(paid)]).exit_code == 0
    # Back to the null backend: the entry's api_usage disappears...
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    m = _manifest(library, track_id)
    assert "api_usage" not in m["documents"]["video"]["run"]
    # ...but the ledger keeps both paid runs, in order, never rewritten.
    ledger = m["api_ledger"]
    assert [(e["stage"], e["status"], e["calls"]) for e in ledger] == [
        ("video", "failed", 1), ("video", "ok", 1)]
    assert "2 API-spending run(s): 2 calls" in _costs_line(library)


def test_ledger_seeded_from_pre_1_5_manifest(tmp_path):
    # A manifest last written before api_ledger existed: its recorded spend
    # must survive the first write by this version.
    wav = tmp_path / "t.wav"
    _make_wav(wav)
    library = tmp_path / "lib"
    track_id = _ingest(wav, library)
    mpath = library / track_id / "manifest.json"
    m = _manifest(library, track_id)
    usage = {"calls": 3, "input_tokens": 3000, "output_tokens": 300,
             "usd": 0.0045, "estimated_usd": 0.006}
    m["documents"]["video"] = {"status": "failed", "error": "x", "run": {
        "started_at": "2026-01-01T00:00:00Z", "duration_seconds": 1.0,
        "tool_version": "mrw 0.1.0", "api_usage": usage}}
    mpath.write_text(json.dumps(m))
    assert "1 API-spending run(s): 3 calls" in _costs_line(library)  # fallback

    # Any later write replaces the video entry (here: wholesale).
    from mrw.models import DocumentEntry

    Library(library).update_manifest(
        track_id,
        lambda mm: setattr(mm.documents, "video", DocumentEntry(status="pending")),
    )
    m = _manifest(library, track_id)
    assert m["api_ledger"] == [{"stage": "video", "started_at": "2026-01-01T00:00:00Z",
                                "status": "failed", **usage}]
    assert "1 API-spending run(s): 3 calls" in _costs_line(library)


def test_ledger_absent_without_spend(tmp_path):
    wav = tmp_path / "t.wav"
    _make_wav(wav)
    library = tmp_path / "lib"
    track_id = _ingest(wav, library)
    Library(library).update_manifest(track_id, lambda m: None)
    assert "api_ledger" not in _manifest(library, track_id)  # absent, not []


# --- interruption ----------------------------------------------------------------


def test_interrupt_mid_captioning_records_failed_and_spend(tmp_path, monkeypatch):
    library, track_id = _video_track(tmp_path)
    paid = tmp_path / "paid.toml"
    paid.write_text('[video]\ncaption_backend = "anthropic"\n')
    _use_backend(monkeypatch, _Backend(boom=KeyboardInterrupt()))
    result = runner.invoke(
        app, ["video", track_id, "--library", str(library), "--config", str(paid)])
    assert result.exit_code == 130, result.output
    assert "interrupted" in result.output
    m = _manifest(library, track_id)
    entry = m["documents"]["video"]
    assert entry["status"] == "failed"
    assert entry["error"] == "interrupted (KeyboardInterrupt)"
    assert entry["run"]["api_usage"]["calls"] == 1  # paid call kept
    assert m["api_ledger"][0]["status"] == "failed"
    assert not (library / track_id / ".frames.tmp").exists()


def _features_track(tmp_path: Path) -> tuple[Path, str]:
    from test_m3_smoke import _click_track, _fabricate_track

    mix = _click_track(3.0)
    library = tmp_path / "lib"
    return library, _fabricate_track(library, mix, {"drums": mix}, tmp_path)


def test_sigterm_during_stage_is_recorded_and_exits_130(tmp_path, monkeypatch):
    import os
    import signal

    import mrw.features

    library, track_id = _features_track(tmp_path)

    def killed(path):
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("SIGTERM was not turned into KeyboardInterrupt")

    monkeypatch.setattr(mrw.features, "_load", killed)
    before = signal.getsignal(signal.SIGTERM)
    result = runner.invoke(app, ["features", track_id, "--library", str(library)])
    assert result.exit_code == 130, result.output
    assert "error: interrupted" in result.output
    entry = _manifest(library, track_id)["documents"]["audio_features"]
    assert entry["status"] == "failed"
    assert entry["error"] == "interrupted (KeyboardInterrupt)"
    assert signal.getsignal(signal.SIGTERM) is before  # handler restored


def test_interrupted_stems_records_failed_and_cleans_tmp(tmp_path, monkeypatch):
    import mrw.stems

    wav = tmp_path / "t.wav"
    _make_wav(wav)
    library = tmp_path / "lib"
    track_id = _ingest(wav, library)

    def interrupted(flac, out_dir, model, device, cpu_threads=1):
        (out_dir / "vocals.flac").write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(mrw.stems, "_separate", interrupted)
    cfg = tmp_path / "mrw.toml"
    cfg.write_text('[stems]\ndevice = "cpu"\n')
    result = runner.invoke(
        app, ["stems", track_id, "--library", str(library), "--config", str(cfg)])
    assert result.exit_code == 130, result.output
    stems = _manifest(library, track_id)["stems"]
    assert stems["status"] == "failed"
    assert stems["error"] == "interrupted (KeyboardInterrupt)"
    assert not (library / track_id / ".stems.tmp").exists()
    assert not (library / track_id / "stems").exists()


# --- atomic copy_video -------------------------------------------------------------


def test_interrupted_video_copy_leaves_nothing_behind(tmp_path, monkeypatch):
    import mrw.ingest
    from test_p1_inputs import _tiny_video

    mp4 = tmp_path / "clip.mp4"
    _tiny_video(mp4, "red")
    library = tmp_path / "lib"
    cfg = tmp_path / "mrw.toml"
    cfg.write_text("[ingest]\ncopy_video = true\n")

    real_copy = mrw.ingest.shutil.copyfile

    def partial_copy(src, dst):
        Path(dst).write_bytes(Path(src).read_bytes()[:1000])
        raise KeyboardInterrupt

    monkeypatch.setattr(mrw.ingest.shutil, "copyfile", partial_copy)
    result = runner.invoke(
        app, ["ingest", str(mp4), "--library", str(library), "--config", str(cfg)])
    assert result.exit_code == 130, result.output
    track_dir = next(library.iterdir())
    assert not list(track_dir.glob("source_video*"))
    assert not (track_dir / ".source_video.tmp").exists()

    monkeypatch.setattr(mrw.ingest.shutil, "copyfile", real_copy)
    result = runner.invoke(
        app, ["ingest", str(mp4), "--library", str(library), "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    copy = track_dir / "source_video.mp4"
    assert copy.read_bytes() == mp4.read_bytes()


def test_truncated_copy_is_ignored_for_the_original(tmp_path):
    from mrw.video import _resolve_video_path
    from test_p1_inputs import _tiny_video

    mp4 = tmp_path / "clip.mp4"
    _tiny_video(mp4, "red")
    library = tmp_path / "lib"
    track_id = _ingest(mp4, library)
    track_dir = library / track_id
    (track_dir / "source_video.mp4").write_bytes(mp4.read_bytes()[:500])
    source = json.loads((track_dir / "source.json").read_text())
    assert _resolve_video_path(track_dir, source) == mp4

    # A same-size copy with other bytes is picked, then refused by identity.
    data = bytearray(mp4.read_bytes())
    data[-1] ^= 0xFF
    (track_dir / "source_video.mp4").write_bytes(bytes(data))
    result = runner.invoke(app, ["video", track_id, "--library", str(library)])
    assert result.exit_code == 2
    assert "not the ingested video" in result.output
