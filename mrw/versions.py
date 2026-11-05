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
"""

from __future__ import annotations

from typing import Optional

ALGORITHM_VERSIONS: dict[str, int] = {
    "stems": 1,
    "audio_features": 1,
    "lyrics": 1,
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
