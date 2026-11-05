"""Per-stage algorithm versions (the code half of a document's identity).

A stage's config_hash covers its config subset; this constant covers its
code. Bump a stage's version in the same PR as any change that can alter
that stage's output under an unchanged config (a bug fix, a changed
structural constant, a parsing change). A document recorded under an older
version is then treated as stale — the stage re-runs instead of reporting
"up to date", and `mrw status` shows `stale`.

Kept apart from stage_hash on purpose: folding it into the config hash would
make every bump re-separate every track's stems, and it keeps the
config-evolution policy (review 006) and the code-evolution policy separate.
Entries written before this field existed carry no version and count as 1.

Ingest ("source") is deliberately unversioned for now: ingest never
re-runs on an existing track (re-ingest is a no-op), so a version could
only ever display a stale that nothing can clear. Known consequence of the
lyrics-decoding fix (lyrics 2): source.json lyrics_input.line_count /
has_timestamps for a BOM or non-UTF-8 lyrics file differ between tracks
ingested before and after it. The lyrics stage re-decodes the file itself
and reads neither field. Version source once ingest can update a track.
"""

from __future__ import annotations

from typing import Optional

ALGORITHM_VERSIONS: dict[str, int] = {
    "stems": 1,
    "audio_features": 1,
    # 2: LRC parsing (one line per time tag, ID tags skipped, [offset:],
    #    mm:ss:xx) and BOM/UTF-16/cp1252-aware decoding (LYR-5/7, SEC-7).
    "lyrics": 2,
    # 2: palette clustered from per-frame downscaled pixels (was averages
    #    of whole image rows — VID-1); rotation-aware frame geometry
    #    (portrait phone video decoded scrambled — VID-4).
    "video": 2,
}


def current(stage: str) -> int:
    return ALGORITHM_VERSIONS[stage]


def is_current(stage: str, stored: Optional[int]) -> bool:
    """True when a recorded entry was produced by this stage's current code."""
    if stage not in ALGORITHM_VERSIONS:
        return True
    return (stored or 1) == ALGORITHM_VERSIONS[stage]
