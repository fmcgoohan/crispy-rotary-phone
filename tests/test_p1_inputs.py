"""P1 input tracking (DET-1 fuller form, DET-3/ORCH-2/VID-5): stems
content_sha256, per-document inputs_sha256, and the video identity check.

Fast: synthetic stems, generated tones and a generated 2 s video clip.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

import mrw.stems
from mrw import inputs
from mrw.cli import app
from mrw.models import Manifest
from mrw.stems import STEM_NAMES
from test_m2_field_fixes import _fake_stems, _ingest, _make_wav
from test_m3_smoke import _click_track, _fabricate_track

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)

runner = CliRunner()


def _manifest(library: Path, track_id: str) -> dict:
    return json.loads((library / track_id / "manifest.json").read_text())


def _write_manifest(library: Path, track_id: str, m: dict) -> None:
    (library / track_id / "manifest.json").write_text(json.dumps(m))


def _fake_separate(monkeypatch) -> None:
    monkeypatch.setattr(
        mrw.stems, "_separate",
        lambda flac, out_dir, model, device, cpu_threads=1: _fake_stems(out_dir),
    )


@pytest.mark.parametrize("retain", [True, False])
def test_stems_content_sha256_recorded(tmp_path, monkeypatch, retain):
    wav = tmp_path / "t.wav"
    _make_wav(wav)
    library = tmp_path / "lib"
    track_id = _ingest(wav, library)
    _fake_separate(monkeypatch)
    cfg = tmp_path / "mrw.toml"
    cfg.write_text(f'[stems]\ndevice = "cpu"\nretain = {str(retain).lower()}\n')
    result = runner.invoke(
        app, ["stems", track_id, "--library", str(library), "--config", str(cfg)])
    assert result.exit_code == 0, result.output

    # Expected: sha256 over the per-file hex digests in STEM_NAMES order.
    ref = tmp_path / "ref"
    ref.mkdir()
    _fake_stems(ref)
    digests = [hashlib.sha256((ref / f"{n}.flac").read_bytes()).hexdigest()
               for n in STEM_NAMES]
    expected = hashlib.sha256("".join(digests).encode()).hexdigest()
    stems = _manifest(library, track_id)["stems"]
    assert stems["content_sha256"] == expected  # kept even when not retained
    assert stems["retained"] is retain


def test_features_records_inputs_and_reruns_when_stems_change(tmp_path):
    mix = _click_track(4.0)
    library = tmp_path / "lib"
    track_id = _fabricate_track(library, mix, {"drums": mix}, tmp_path)
    lib = ["--library", str(library)]
    m = _manifest(library, track_id)
    m["stems"]["content_sha256"] = "a" * 64
    _write_manifest(library, track_id, m)

    assert runner.invoke(app, ["features", track_id, *lib]).exit_code == 0
    entry = _manifest(library, track_id)["documents"]["audio_features"]
    source = json.loads((library / track_id / "source.json").read_text())
    assert entry["inputs_sha256"] == inputs.current_inputs(
        "audio_features", Manifest.model_validate(_manifest(library, track_id)), source)
    assert len(entry["inputs_sha256"]) == 16
    assert "no-op" in runner.invoke(app, ["features", track_id, *lib]).output

    # Stems changed behind features' back (e.g. the stale flip was lost):
    # status derives stale and the stage re-runs instead of a no-op.
    m = _manifest(library, track_id)
    m["stems"]["content_sha256"] = "b" * 64
    _write_manifest(library, track_id, m)
    assert "features=stale" in runner.invoke(app, ["status", *lib]).output
    result = runner.invoke(app, ["features", track_id, *lib])
    assert result.exit_code == 0 and "no-op" not in result.output
    assert "features=ok" in runner.invoke(app, ["status", *lib]).output


def test_absent_inputs_sha256_counts_as_matching(tmp_path):
    # Entries recorded before manifest 1.5.0 carry no inputs_sha256: they
    # must stay up to date (no mass re-run on upgrade).
    mix = _click_track(4.0)
    library = tmp_path / "lib"
    track_id = _fabricate_track(library, mix, {"drums": mix}, tmp_path)
    lib = ["--library", str(library)]
    assert runner.invoke(app, ["features", track_id, *lib]).exit_code == 0
    m = _manifest(library, track_id)
    del m["documents"]["audio_features"]["inputs_sha256"]
    m["stems"]["content_sha256"] = "c" * 64
    _write_manifest(library, track_id, m)
    assert "features=ok" in runner.invoke(app, ["status", *lib]).output
    assert "no-op" in runner.invoke(app, ["features", track_id, *lib]).output


def _manifest_model(**stems) -> Manifest:
    ok = {"status": "ok", "path": "x.json", "content_sha256": "1" * 64}
    return Manifest.model_validate({
        "track_id": "0" * 16, "title": "t", "has_video": True,
        "created_at": "2026-01-01T00:00:00Z",
        "documents": {"source": ok, "audio_features": ok, "lyrics": ok,
                      "video": ok, "structure": ok},
        "stems": {"status": "ok", "retained": True, **stems}})


def test_current_inputs_cover_exactly_the_consumed_identities():
    source = {
        "file": {"original_path": "/a/x.mp4", "sha256": "f" * 64},
        "normalized_audio": {"audio_stream_sha256": "e" * 64},
    }
    m = _manifest_model(content_sha256="d" * 64)
    base = {s: inputs.current_inputs(s, m, source)
            for s in ("audio_features", "lyrics", "video")}

    # Supplied lyrics change only lyrics' inputs.
    with_lyrics = {**source, "lyrics_input": {"sha256": "9" * 64}}
    assert inputs.current_inputs("lyrics", m, with_lyrics) != base["lyrics"]
    assert inputs.current_inputs("audio_features", m, with_lyrics) == base["audio_features"]

    # New stems change features and lyrics, not video.
    m2 = _manifest_model(content_sha256="8" * 64)
    assert inputs.current_inputs("audio_features", m2, source) != base["audio_features"]
    assert inputs.current_inputs("lyrics", m2, source) != base["lyrics"]
    assert inputs.current_inputs("video", m2, source) == base["video"]

    # New features content changes lyrics only.
    m3 = _manifest_model(content_sha256="d" * 64)
    m3.documents.audio_features.content_sha256 = "7" * 64
    assert inputs.current_inputs("lyrics", m3, source) != base["lyrics"]
    assert inputs.current_inputs("audio_features", m3, source) == base["audio_features"]

    # Path-only changes (relink) never change any stage's inputs.
    moved = {**source, "file": {"original_path": "/b/y.mp4", "sha256": "f" * 64}}
    for stage, value in base.items():
        assert inputs.current_inputs(stage, m, moved) == value


def _tiny_video(path: Path, color: str) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=44100",
         "-f", "lavfi", "-i", f"color=c={color}:size=160x120:rate=12:duration=2",
         "-map", "1:v", "-map", "0:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "alac", str(path)],
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode()[:400]


def test_replaced_original_is_refused_before_analysis(tmp_path):
    mp4 = tmp_path / "clip.mp4"
    _tiny_video(mp4, "red")
    library = tmp_path / "lib"
    track_id = _ingest(mp4, library)
    before = (library / track_id / "manifest.json").read_bytes()

    _tiny_video(mp4, "blue")  # a different cut saved at the same path
    result = runner.invoke(app, ["video", track_id, "--library", str(library)])
    assert result.exit_code == 2, result.output
    assert "not the ingested video" in result.output
    assert str(mp4) in result.output
    source = json.loads((library / track_id / "source.json").read_text())
    assert source["file"]["sha256"] in result.output  # expected hash named
    assert not (library / track_id / "video.json").exists()
    assert not (library / track_id / "frames").exists()
    assert (library / track_id / "manifest.json").read_bytes() == before
