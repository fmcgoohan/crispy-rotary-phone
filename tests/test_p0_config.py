"""P0-7 / P0-8: config fails closed; the serializer refuses non-finite
floats and never emits -0.0."""

from __future__ import annotations

import math

import pytest
from typer.testing import CliRunner

from mrw import canonical, config
from mrw.cli import app

runner = CliRunner()


@pytest.mark.parametrize("toml,needle", [
    ('[stems]\ndevices = "cpu"\n', "stems.devices"),          # typo
    ('[stems]\ndevice = "gpu"\n', "stems.device"),            # bad value
    ('[lyrics]\nno_speech_threshold = 0.4\n', "no_speech"),  # renamed key
    ('[video]\ncaption_budget_usd_per_run = inf\n', "caption_budget"),
    ('[ingest]\nsample_rate = 48000\n', "ingest.sample_rate"),
    ('[nonsense]\nx = 1\n', "nonsense"),
    ('[stems\n', "cannot read config"),                      # TOML syntax
])
def test_bad_config_exits_2_with_the_key_named(tmp_path, toml, needle):
    cfg = tmp_path / "mrw.toml"
    cfg.write_text(toml)
    with pytest.raises(config.ConfigError, match=needle):
        config.load(cfg)
    result = runner.invoke(app, ["status", "--library", str(tmp_path / "lib"),
                                 "--config", str(cfg)])
    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_strictness_does_not_move_stage_hashes():
    # Golden values from before the change: extra="forbid" / Literal types
    # must not alter model_dump, or every library would mass-stale.
    defaults = config.Config()
    assert [config.stage_hash(getattr(defaults, s))
            for s in ("ingest", "stems", "features", "lyrics", "video")] == [
        "3bafde60e0bcb7eb", "2f212e9aa4a05d70", "39dc2973b83251c3",
        "762e9f5f09f43085", "c3041f27e221e1c0"]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_dumps_rejects_non_finite(value):
    with pytest.raises(ValueError):
        canonical.dumps({"x": value})


def test_rounders_normalise_negative_zero():
    for fn in (canonical.round_seconds, canonical.round_db, canonical.round_hz,
               canonical.round_ratio, canonical.round_bpm):
        assert canonical.dumps(fn(-0.0001)) == "0.0\n"
