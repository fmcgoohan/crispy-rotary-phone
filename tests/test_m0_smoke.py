"""M0 smoke test (PLAN.md §11):

`mrw ingest fixture.wav` twice → track dir exists; second run reports no-op;
source.json byte-identical across runs (and across separate library roots);
documents validate against the hand-drafted schemas.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mrw.cli import app

REPO = Path(__file__).resolve().parents[1]
SCHEMAS = REPO / "schemas"

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)

runner = CliRunner()


@pytest.fixture(scope="module")
def fixture_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """2-second 440 Hz stereo tone, 44.1 kHz 16-bit."""
    path = tmp_path_factory.mktemp("fixtures") / "tone.wav"
    sr, seconds = 44100, 2.0
    frames = bytearray()
    for i in range(int(sr * seconds)):
        v = int(32767 * 0.3 * math.sin(2 * math.pi * 440 * i / sr))
        frames += struct.pack("<hh", v, v)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))
    return path


def _ingest(wav: Path, library: Path) -> Path:
    result = runner.invoke(app, ["ingest", str(wav), "--library", str(library)])
    assert result.exit_code == 0, result.output
    track_dirs = [p for p in library.iterdir() if p.is_dir()]
    assert len(track_dirs) == 1
    return track_dirs[0]


def _validate(document: dict, schema_name: str) -> None:
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    registry = Registry()
    for f in SCHEMAS.glob("*.schema.json"):
        registry = registry.with_resource(
            f.name, Resource.from_contents(json.loads(f.read_text()))
        )
    schema = json.loads((SCHEMAS / schema_name).read_text())
    errors = list(
        Draft202012Validator(schema, registry=registry).iter_errors(document)
    )
    assert not errors, "\n".join(e.message for e in errors)


def test_ingest_creates_valid_documents(fixture_wav: Path, tmp_path: Path) -> None:
    track_dir = _ingest(fixture_wav, tmp_path / "lib")

    source = json.loads((track_dir / "source.json").read_text())
    manifest = json.loads((track_dir / "manifest.json").read_text())
    _validate(source, "source.schema.json")
    _validate(manifest, "manifest.schema.json")

    assert source["track_id"] == track_dir.name
    assert abs(source["normalized_audio"]["duration_seconds"] - 2.0) < 0.01
    assert (track_dir / "source_audio.flac").is_file()
    assert manifest["has_video"] is False
    assert manifest["documents"]["source"]["status"] == "ok"
    assert manifest["documents"]["video"]["status"] == "not_applicable"
    assert manifest["documents"]["audio_features"]["status"] == "pending"


def test_second_ingest_is_noop(fixture_wav: Path, tmp_path: Path) -> None:
    library = tmp_path / "lib"
    track_dir = _ingest(fixture_wav, library)
    before = (track_dir / "source.json").read_bytes()

    result = runner.invoke(app, ["ingest", str(fixture_wav), "--library", str(library)])
    assert result.exit_code == 0, result.output
    assert "no-op" in result.output
    assert (track_dir / "source.json").read_bytes() == before


def test_source_document_is_deterministic(fixture_wav: Path, tmp_path: Path) -> None:
    dir_a = _ingest(fixture_wav, tmp_path / "lib_a")
    dir_b = _ingest(fixture_wav, tmp_path / "lib_b")
    assert dir_a.name == dir_b.name  # same track_id
    assert (dir_a / "source.json").read_bytes() == (dir_b / "source.json").read_bytes()


def test_missing_file_is_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["ingest", str(tmp_path / "nope.wav"), "--library", str(tmp_path / "lib")]
    )
    assert result.exit_code == 2
