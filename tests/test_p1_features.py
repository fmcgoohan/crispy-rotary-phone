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
