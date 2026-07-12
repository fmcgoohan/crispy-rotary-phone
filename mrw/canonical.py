"""Canonical JSON serialization — the only code path that writes documents.

Determinism lives or dies here: fixed key order (pydantic field order),
UTF-8 without ASCII escaping, 2-space indent, trailing newline, and explicit
float rounding per the schema precision contract (schemas/README.md).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

# Precision contract: seconds 3 dp, dB 2 dp, Hz 1 dp, ratios/confidences 3 dp.
def round_seconds(x: float) -> float:
    return round(float(x), 3)


def round_db(x: float) -> float:
    return round(float(x), 2)


def round_hz(x: float) -> float:
    return round(float(x), 1)


def round_ratio(x: float) -> float:
    return round(float(x), 3)


def round_bpm(x: float) -> float:
    return round(float(x), 1)


def dumps(data: Any) -> str:
    """Serialize to the canonical text form (insertion order, trailing newline)."""
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def write(path: Path, data: Any) -> str:
    """Atomically write canonical JSON; returns the text written.

    Writes to a temp file in the same directory and renames into place, so a
    failed stage never leaves a half-written document.
    """
    path = Path(path)
    text = dumps(data)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return text
