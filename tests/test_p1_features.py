"""Feature-measure fixes (analysis report FEAT-1/3/8/13, DET-11, SCH-9).

- Onsets are level-gated: floor frames and (stems) frames more than
  |onset_gate_relative_db| below the mix drop their onsets; a channel at the
  floor throughout takes the R-9 sentinel (FEAT-1); silence decisions read
  the rounded rms series (FEAT-13).
- Beat grid trimmed (no beats in beatless pads) and tempo.confidence measures
  onset support (FEAT-3).
- A digitally silent mix fails the stage (DET-11/SCH-9); sub-400 ms input
  is measured as one loudness block (FEAT-8).

Generated fixtures only (T3).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from mrw import config, features
from mrw.cli import app
from test_m3_smoke import (
    SAMPLE_RATE,
    _click_track,
    _fabricate_track,
    _tone,
    _validate,
)

pytestmark = pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None,
    reason="ffmpeg/ffprobe not available",
)

runner = CliRunner()
WINDOW = config.FeaturesConfig().phase_fit_window_seconds


def _q16(y: np.ndarray) -> np.ndarray:
    """What a PCM_16 FLAC round-trip would hand the stage."""
    return (np.round(np.clip(y, -1, 1) * 32767) / 32767).astype(np.float32)


def _dither(seconds: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lsb = rng.integers(-1, 2, int(SAMPLE_RATE * seconds))  # ±1 LSB
    return (lsb / 32767).astype(np.float32)


def _pink(seconds: float, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(SAMPLE_RATE * seconds)
    spec = np.fft.rfft(rng.standard_normal(n))
    k = np.arange(len(spec), dtype=np.float64)
    k[0] = 1.0
    y = np.fft.irfft(spec / np.sqrt(k), n)
    return (0.1 * y / np.abs(y).max()).astype(np.float32)


def _drone(seconds: float) -> np.ndarray:
    return _tone(seconds, 110.0, 0.2) + _tone(seconds, 165.0, 0.15) + _tone(seconds, 220.5, 0.1)


def _pad(seconds: float) -> np.ndarray:
    """Slowly swelling chord: sustained, beatless."""
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    y = sum(
        0.08 * np.sin(2 * np.pi * f * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.2 * t + i))
        for i, f in enumerate((220.0, 277.2, 329.6, 440.0))
    )
    return y.astype(np.float32)


def _clicks(seconds: float, period: float, amp: float, first: float = 0.25) -> np.ndarray:
    n = int(SAMPLE_RATE * seconds)
    y = np.zeros(n, dtype=np.float32)
    click_len = SAMPLE_RATE // 100
    t = np.arange(click_len) / SAMPLE_RATE
    burst = np.sin(2 * np.pi * 1500 * t) * np.linspace(1.0, 0.0, click_len)
    k = 0
    while (start := int(round((first + k * period) * SAMPLE_RATE))) + click_len <= n:
        y[start : start + click_len] += amp * burst
        k += 1
    return y


# --- (a) onset gating -----------------------------------------------------------


@pytest.fixture(scope="module")
def gated_doc(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("gate")
    mix = _click_track(8.0)
    stems = {
        "drums": mix,  # the real part: equal to the mix
        "vocals": _dither(8.0),  # dither only: rms at the floor throughout
        "bass": mix * 10 ** (-55 / 20),  # bleed only, 55 dB under the mix
        # other: bit-exact silence (the original R-9 case)
    }
    library = tmp / "lib"
    track_id = _fabricate_track(library, mix, stems, tmp)
    result = runner.invoke(app, ["features", track_id, "--library", str(library)])
    assert result.exit_code == 0, result.output
    return json.loads((library / track_id / "audio_features.json").read_text())


def test_dither_only_stem_takes_r9_sentinel(gated_doc: dict) -> None:
    _validate(gated_doc)
    vocals = gated_doc["stems"]["vocals"]
    assert all(v == -80.0 for v in vocals["rms_db"]["values"])
    assert all(v == 0.0 for v in vocals["spectral_centroid_hz"]["values"])
    assert vocals["onsets"]["times"] == [] and vocals["onsets"]["strengths"] == []
    assert vocals["onsets"]["strength_reference"] == 1.0


def test_bleed_only_stem_is_gated_relative_to_mix(gated_doc: dict) -> None:
    bass = gated_doc["stems"]["bass"]
    # Above the floor in the click frames, so only the relative gate removes it.
    assert max(bass["rms_db"]["values"]) > -80.0
    assert bass["onsets"]["times"] == []
    assert bass["onsets"]["strength_reference"] == 1.0


def test_drums_stem_keeps_its_onsets(gated_doc: dict) -> None:
    drums = gated_doc["stems"]["drums"]["onsets"]
    mix = gated_doc["mix"]["onsets"]
    assert len(drums["times"]) >= 14  # 16 clicks in 8 s
    assert drums["times"] == mix["times"]  # stem == mix: nothing gated
    assert drums["strengths"] == mix["strengths"]


def test_gate_embedded_next_to_onsets(gated_doc: dict) -> None:
    assert gated_doc["schema_version"] == "1.1.0"
    assert gated_doc["mix"]["onsets"]["gate"] == {"floor_db": -80.0}  # floor rule only
    for name in ("vocals", "drums", "bass", "other"):
        assert gated_doc["stems"][name]["onsets"]["gate"] == {
            "floor_db": -80.0,
            "relative_db": -30.0,
        }


def test_relative_gate_is_config_driven(tmp_path: Path) -> None:
    # A gate below the bleed level lets the bleed through, the embedded gate
    # records the configured value, and the config_hash moves (mass-stale is
    # deliberate per review 006).
    mix = _click_track(8.0)
    library = tmp_path / "lib"
    track_id = _fabricate_track(library, mix, {"bass": mix * 10 ** (-55 / 20)}, tmp_path)
    cfg = tmp_path / "mrw.toml"
    cfg.write_text("[features]\nonset_gate_relative_db = -70.0\n")
    result = runner.invoke(
        app, ["features", track_id, "--library", str(library), "--config", str(cfg)]
    )
    assert result.exit_code == 0, result.output
    doc = json.loads((library / track_id / "audio_features.json").read_text())
    _validate(doc)
    bass = doc["stems"]["bass"]["onsets"]
    assert bass["gate"]["relative_db"] == -70.0
    assert len(bass["times"]) >= 14
    assert config.stage_hash(config.load(cfg).features) != config.stage_hash(
        config.FeaturesConfig()
    )


def test_positive_relative_gate_rejected(tmp_path: Path) -> None:
    cfg = tmp_path / "mrw.toml"
    cfg.write_text("[features]\nonset_gate_relative_db = 5.0\n")
    with pytest.raises(Exception):
        config.load(cfg)


def test_silence_decisions_use_rounded_rms() -> None:
    # FEAT-13: a sine whose RMS is -79.998 dBFS publishes -80.0 everywhere,
    # so it must be 'silent throughout' for centroid and onsets alike.
    amp = np.sqrt(2) * 10 ** (-79.998 / 20)
    y = _tone(2.0, 1000.0, amp)
    channel, rms = features._channel_features(y)
    assert all(v == -80.0 for v in channel.rms_db.values)
    assert list(rms) == channel.rms_db.values
    assert all(v == 0.0 for v in channel.spectral_centroid_hz.values)
    assert channel.onsets.times == [] and channel.onsets.strength_reference == 1.0


# --- (b) beat grid: trim + onset-support confidence ---------------------------------


def test_click_track_confidence_high_and_bpm_unchanged() -> None:
    tempo, beats = features._beats_and_tempo(_q16(_click_track(20.0)), WINDOW)
    assert tempo.bpm_global == 120.0
    assert tempo.confidence >= 0.8


def test_straight_eighths_are_not_read_as_beatless() -> None:
    # The half beat is not an off-beat probe point (BEAT_OFF_PROBES).
    y = _clicks(20.0, 0.5, 0.6) + _clicks(20.0, 0.25, 0.3)
    tempo, _ = features._beats_and_tempo(_q16(y), WINDOW)
    assert tempo.bpm_global == 120.0
    assert tempo.confidence >= 0.8


def _subdivided(bpm: float, per_beat: int, seconds: float = 20.0) -> np.ndarray:
    """Beat clicks (0.6) plus hats (0.3) dividing each beat into per_beat."""
    period = 60.0 / bpm
    return _clicks(seconds, period, 0.6) + _clicks(seconds, period / per_beat, 0.3)


@pytest.mark.parametrize(
    ("bpm", "per_beat"),
    [(100, 3), (120, 3), (150, 3), (120, 4), (140, 4), (120, 5), (120, 6)],
    ids=["trip100", "trip120", "trip150", "16th120", "16th140", "quint120", "sext120"],
)
def test_subdivided_rhythms_are_not_read_as_beatless(bpm: float, per_beat: int) -> None:
    # Review D-1: the first draft probed 1/3 and 2/3 of a beat, exactly where
    # triplet hats sit, and scored correctly-tracked shuffles ~0.07. The
    # k/7 probes miss every common subdivision.
    tempo, _ = features._beats_and_tempo(_q16(_subdivided(bpm, per_beat)), WINDOW)
    assert abs(tempo.bpm_global - bpm) <= 1.0  # tracked at the true tempo
    assert tempo.confidence >= 0.5


def test_off_beat_probes_avoid_common_subdivisions() -> None:
    w = features.BEAT_SALIENCE_WINDOW_SECONDS
    for per_beat in (2, 3, 4, 5, 6):
        grid = [k / per_beat for k in range(per_beat + 1)]
        # At 120 BPM (0.5 s beat) some probe clears every subdivision hit by
        # more than the ±30 ms window.
        assert any(
            min(abs(x - g) for g in grid) * 0.5 > w for x in features.BEAT_OFF_PROBES
        ), per_beat


@pytest.mark.parametrize("make", [_drone, _pink], ids=["drone", "pink_noise"])
def test_beatless_mix_confidence_low(make) -> None:
    tempo, _ = features._beats_and_tempo(_q16(make(20.0)), WINDOW)
    assert tempo.confidence < 0.5


def test_no_beats_inside_beatless_pads() -> None:
    # 12 s pad | 16 s of 120 BPM clicks over the pad | 12 s pad.
    y = np.concatenate([_pad(12.0), _click_track(16.0) + _pad(16.0), _pad(12.0)])
    tempo, beats = features._beats_and_tempo(_q16(y), WINDOW)
    times = np.array(beats.times)
    assert len(times) >= 25
    assert times.min() >= 12.0 - 0.05 and times.max() <= 28.0 + 0.05
    assert tempo.confidence >= 0.8


# --- (c) degenerate inputs ------------------------------------------------------------


def test_silent_mix_fails_stage_and_keeps_prior_document(tmp_path: Path) -> None:
    silence = np.zeros(4 * SAMPLE_RATE, dtype=np.float32)
    library = tmp_path / "lib"
    track_id = _fabricate_track(library, silence, {}, tmp_path)
    # A prior document from an older config: the failed re-run must not touch it.
    prior = library / track_id / "audio_features.json"
    prior.write_text('{"prior": true}\n')
    mpath = library / track_id / "manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["documents"]["audio_features"] = {
        "status": "ok", "path": "audio_features.json", "config_hash": "f" * 16}
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")

    result = runner.invoke(app, ["features", track_id, "--library", str(library)])
    assert result.exit_code == 1, result.output
    assert "no beat evidence in mix" in result.output
    assert "Traceback" not in result.output
    entry = json.loads(mpath.read_text())["documents"]["audio_features"]
    assert entry["status"] == "failed"
    assert "no beat evidence in mix" in entry["error"]
    assert prior.read_text() == '{"prior": true}\n'


def test_sub_block_input_does_not_crash(tmp_path: Path) -> None:
    mix = _click_track(0.3)  # one click at 0.25 s; under pyloudnorm's 400 ms block
    library = tmp_path / "lib"
    track_id = _fabricate_track(library, mix, {"drums": mix}, tmp_path)
    result = runner.invoke(app, ["features", track_id, "--library", str(library)])
    assert result.exit_code == 0, result.output
    doc = json.loads((library / track_id / "audio_features.json").read_text())
    _validate(doc)
    assert doc["duration_seconds"] == 0.3
    assert doc["tempo"]["bpm_global"] > 0
    assert -70.0 < doc["loudness"]["integrated_lufs"] < 0.0


def test_single_block_loudness_matches_sine_level() -> None:
    # A 0.3 s 1 kHz sine of peak 0.1 in both channels: K-weighting is ~0 dB
    # at 1 kHz, so LUFS ≈ -0.691 + 20·log10(0.1) = -20.69.
    y = _tone(0.3, 1000.0, 0.1)
    lufs = features._integrated_lufs(np.stack([y, y], axis=1))
    assert abs(lufs - (-20.69)) < 1.0
