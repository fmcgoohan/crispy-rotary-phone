"""Shared test fixtures.

Config isolation: `mrw` commands invoked without --config read ./mrw.toml
from the CWD — which in tests is the repo root, whose mrw.toml is the
operator's live config and changes with field experiments. Discovered on
PR #10 when an uncommitted `[lyrics] language = "en"` pin flipped a
determinism test's language_source. Every test runs chdir'd into its own
tmp dir so no-config invocations see built-in defaults, never operator
state.

The session-scoped fixture covers module- and session-scoped fixtures too
(they are instantiated before any function-scoped autouse fixture runs).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_from_repo_config(tmp_path_factory, monkeypatch):
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


@pytest.fixture(scope="session", autouse=True)
def _isolate_session_from_repo_config(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path_factory.mktemp("session-cwd"))
        yield
