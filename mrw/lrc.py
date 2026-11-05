"""Supplied-lyrics text handling shared by ingest and the lyrics stage:
byte decoding and LRC time/ID tags.

One implementation, so ingest's `has_timestamps` and the lyrics parser can
never disagree about what a time tag is.
"""

from __future__ import annotations

import codecs
import re

# [mm:ss], [mm:ss.x], [mm:ss.xx], [mm:ss.xxx] and the [mm:ss:xx] variant.
TIME_TAG = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\]")
# The run of time tags that starts a line; compressed LRC repeats a line by
# stacking several: [00:45.00][01:30.00]Chorus line.
LEADING_TIME_TAGS = re.compile(r"^(?:\[\d{1,3}:\d{2}(?:[.:]\d{1,3})?\])+")
# Standard LRC ID tags (metadata, not lyrics or section markup). An explicit
# list: a generic [word: …] pattern would also swallow "[Chorus: Artist]".
ID_TAG = re.compile(
    r"^\[(?:ar|al|ti|au|by|length|offset|re|ve|tool|la|lang|#)\s*:[^\]]*\]$",
    re.IGNORECASE,
)
OFFSET_TAG = re.compile(r"^\[offset\s*:\s*([+-]?\d+)\s*\]$", re.IGNORECASE)

_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)


def decode_text(data: bytes) -> str:
    """Decode a lyrics file without silently mangling it.

    A byte-order mark selects UTF-8/16/32 (and is removed — a kept U+FEFF
    survives str.strip() and hides the first line's time tag). Otherwise
    strict UTF-8, then cp1252 (the usual Windows editor encoding), then
    latin-1, which cannot fail. Never errors="replace".
    """
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            try:
                return data[len(bom):].decode(encoding)
            except UnicodeDecodeError:
                break  # truncated/corrupt: fall through to the byte-safe chain
    for encoding in ("utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def tag_seconds(match: re.Match) -> float:
    """Seconds encoded by one TIME_TAG match (unrounded)."""
    minutes, seconds = int(match.group(1)), int(match.group(2))
    frac = (match.group(3) or "0").ljust(3, "0")[:3]
    return minutes * 60 + seconds + int(frac) / 1000


def leading_time_tags(line: str) -> list[float]:
    """Seconds of every time tag in the run that starts `line` (stripped)."""
    run = LEADING_TIME_TAGS.match(line)
    if not run:
        return []
    return [tag_seconds(m) for m in TIME_TAG.finditer(run.group(0))]


def offset_seconds(lines: list[str]) -> float:
    """The file's [offset:±ms] adjustment, as seconds to SUBTRACT from every
    time tag (LRC: a positive offset makes lyrics appear sooner)."""
    for line in lines:
        m = OFFSET_TAG.match(line.strip())
        if m:
            return int(m.group(1)) / 1000
    return 0.0
