"""M1 smoke test (PLAN.md §11):

Ingest (a) a generated tone WAV, (b) a generated test-pattern MP4 with a
sine track → correct `has_video`, correct durations (±10 ms), FLAC decodes
to the expected sample count. Plus: .txt lyrics registration, the
ingest.copy_video escape hatch, and the no-audio-stream failure path.

The MP4 fixture uses ALAC audio (lossless, MP4-legal): AAC's ~2100 priming
samples would put the decoded duration ~19 ms past the container duration,
outside the ±10 ms budget, without testing anything about our code.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mrw.cli import app

REPO = Path(__file__).resolve().parents[1]
SCHEMAS = REPO / "schemas"
SAMPLE_RATE = 44100

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)

runner = CliRunner()


def _ffmpeg(*args: str) -> None:
    proc = subprocess.run(["ffmpeg", "-y", "-v", "error", *args], capture_output=True)
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg fixture generation failed: {proc.stderr.decode()[:200]}")


@pytest.fixture(scope="module")
def fixture_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """2-second 440 Hz stereo tone, 44.1 kHz 16-bit."""
    path = tmp_path_factory.mktemp("fixtures") / "tone.wav"
    frames = bytearray()
    for i in range(SAMPLE_RATE * 2):
        v = int(32767 * 0.3 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE))
        frames += struct.pack("<hh", v, v)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(frames))
    return path


@pytest.fixture(scope="module")
def fixture_mp4(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """3-second 320x240 24 fps test pattern with a 440 Hz ALAC sine track."""
    path = tmp_path_factory.mktemp("fixtures") / "pattern.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=24",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3:sample_rate=44100",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "alac", "-shortest", str(path),
    )
    return path


@pytest.fixture(scope="module")
def fixture_video_only(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """1-second test pattern with no audio stream at all."""
    path = tmp_path_factory.mktemp("fixtures") / "silent.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=24",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(path),
    )
    return path


def _ingest(media: Path, library: Path, *extra: str) -> Path:
    result = runner.invoke(app, ["ingest", str(media), "--library", str(library), *extra])
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
    errors = list(Draft202012Validator(schema, registry=registry).iter_errors(document))
    assert not errors, "\n".join(e.message for e in errors)


def _decoded_sample_count(flac: Path) -> int:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(flac), "-f", "s16le", "-"],
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    return len(proc.stdout) // (2 * 2)  # 16-bit stereo


def test_video_ingest_probe_fields(fixture_mp4: Path, tmp_path: Path) -> None:
    track_dir = _ingest(fixture_mp4, tmp_path / "lib")
    source = json.loads((track_dir / "source.json").read_text())
    manifest = json.loads((track_dir / "manifest.json").read_text())
    _validate(source, "source.schema.json")
    _validate(manifest, "manifest.schema.json")

    assert manifest["has_video"] is True
    assert manifest["documents"]["video"]["status"] == "pending"

    video = source["media"]["video"]
    assert video["codec"] == "h264"
    assert (video["width"], video["height"]) == (320, 240)
    assert video["fps"] == 24.0
    assert video["frame_count"] == 72

    assert abs(source["normalized_audio"]["duration_seconds"] - 3.0) <= 0.010
    assert abs(source["media"]["duration_seconds"] - 3.0) <= 0.010


def test_wav_duration_within_10ms(fixture_wav: Path, tmp_path: Path) -> None:
    track_dir = _ingest(fixture_wav, tmp_path / "lib")
    source = json.loads((track_dir / "source.json").read_text())
    assert abs(source["normalized_audio"]["duration_seconds"] - 2.0) <= 0.010


def test_flac_decodes_to_expected_sample_count(
    fixture_wav: Path, fixture_mp4: Path, tmp_path: Path
) -> None:
    for name, media, seconds in [("wav", fixture_wav, 2.0), ("mp4", fixture_mp4, 3.0)]:
        track_dir = _ingest(media, tmp_path / f"lib_{name}")
        assert _decoded_sample_count(track_dir / "source_audio.flac") == int(
            seconds * SAMPLE_RATE
        )


def test_txt_lyrics_registration(fixture_wav: Path, tmp_path: Path) -> None:
    lyrics = tmp_path / "words.txt"
    lyrics.write_text("First line of words\nSecond line\n\nThird after a blank\n")
    track_dir = _ingest(fixture_wav, tmp_path / "lib", "--lyrics", str(lyrics))
    source = json.loads((track_dir / "source.json").read_text())
    _validate(source, "source.schema.json")

    li = source["lyrics_input"]
    assert li["format"] == "txt"
    assert li["line_count"] == 3  # blank lines don't count
    assert li["has_timestamps"] is False
    assert (track_dir / "lyrics_input.txt").is_file()


def test_copy_video_config(fixture_mp4: Path, tmp_path: Path) -> None:
    # Default: original video is referenced, not copied.
    track_dir = _ingest(fixture_mp4, tmp_path / "lib_default")
    assert not (track_dir / "source_video.mp4").exists()

    # ingest.copy_video = true: self-contained copy lands in the track dir.
    cfg = tmp_path / "mrw.toml"
    cfg.write_text("[ingest]\ncopy_video = true\n")
    track_dir = _ingest(fixture_mp4, tmp_path / "lib_copy", "--config", str(cfg))
    copied = track_dir / "source_video.mp4"
    assert copied.is_file()
    assert copied.read_bytes() == fixture_mp4.read_bytes()


def test_no_audio_stream_fails_cleanly(
    fixture_video_only: Path, tmp_path: Path
) -> None:
    library = tmp_path / "lib"
    result = runner.invoke(
        app, ["ingest", str(fixture_video_only), "--library", str(library)]
    )
    assert result.exit_code == 1
    assert "no audio stream" in result.output
    # Fail-loudly without corrupting the library: no half-registered track.
    assert not library.exists() or not any(library.iterdir())
