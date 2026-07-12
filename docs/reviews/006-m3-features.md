# Review 006 — M3: audio feature extraction (milestone verification)

**Scope:** PR #6 as merged (`f71c96f`), `mrw/features.py` read in full, fast
suite executed independently.
**Verdict:** **M3 approved.** Findings F-1/F-2 are small follow-ups for the
next PR; neither blocks the post-merge ear-check.

## Independent verification

- M3 test file fully green in an external environment (Linux, py3.12) —
  including the click-track ground truth (BPM exact, beats within one hop),
  the R-2 backfill and R-9 silent-channel convention tests, document
  double-run byte identity, and the prerequisite failure path.
- `round_bpm` and the params rounders added in review rounds 2–3 verified in
  `canonical.py`; per-document schema versions in the manifest verified.
- The ±30 ms phase-fit sampling window matches the documented rationale and
  is implemented as a symmetric ±3-frame local max.
- Verification footnote for the record: an initial cross-file test failure
  in the external environment was traced to a half-installed torch in that
  sandbox (disk-full during dependency install), not to the repo.
  Diagnosing it surfaced F-2.

## Findings

### F-1 — Phase-fit window is a hardcoded constant — [minor]

±30 ms (±3 frames) lives only in `features.py`. The staleness model keys on
`config_hash`, so changing this constant later would alter `downbeat_offset`
in new runs while every existing document stays `ok`. It has already been
tuned once during development — it is a tuning knob. Promote it to
`FeaturesConfig` (e.g. `phase_fit_window_seconds: 0.03`) so it participates
in the features `config_hash`. No schema change — consumers use
`downbeat_offset`, they don't reinterpret the estimator. While there, audit
`features.py` once for other output-affecting tunables that should be config
members versus structural constants that ride `tool_version`.

### F-2 — run_stems imports torch before validating prerequisites — [minor]

Since PR #5, device resolution (`import torch`) runs before the source-file
existence check, so on a machine where torch is missing or broken, a trivial
problem ("source_audio.flac not found") is reported as "No module named
'torch'". Reorder: validate track/manifest/source file first, import torch
last. Side benefits: faster failures, and the fast test suite passes again
on a machine without the neural dependencies installed — a property the
suite had at M2 and PR #5 unknowingly removed.

### C-1 — Checklist addition

Add under Tests: "**T5** The fast suite (`-m 'not slow'`) must pass in an
environment without the neural dependencies installed; heavy imports happen
lazily and after cheap prerequisite validation." This encodes F-2's lesson
and protects contributors and future CI matrices.

## Additional notes

1. **Config-evolution policy:** adding `phase_fit_window_seconds` to
   `FeaturesConfig` changes the canonical features config subset, so the
   features `config_hash` shifts for every existing track even though
   outputs are identical at the default — the same thing PR #5's
   `cpu_threads` addition already did to the stems hash. This is deliberate:
   subset hashes cover the full canonical config including defaults, so
   adding a field invalidates the stage. State that policy in one sentence
   in CLAUDE.md, and note the expected mass-stale in the PR description so
   it isn't mistaken for a bug.
2. **Test economy:** use the new `phase_fit_window_seconds` field as the
   config knob in the staleness test — one test then proves both F-1's hash
   membership and the stale-display derivation.

## M3 milestone record (automated PR review summary, per P1)

PR #6 went through four automated review rounds with strictly converging
severity, all findings fixed in-PR:

Round 1 (1 major, 1 minor): shared `SCHEMA_VERSION` constant would have let
a future `source.json` bump silently re-version the frozen
`audio_features.json` contract → split into per-document constants;
config-derived `vocal_activity.params` written unrounded → routed through
the precision-contract rounders. Round 2 (1 minor, 1 nit): the DSP
pipeline's thread/BLAS jitter policy was implemented but unstated →
documented (rounding IS the policy; PLAN §7 layer 1); two bare `round()`
calls → `canonical.round_bpm` (new) / `round_ratio`. Round 3 (2 minor,
2 nit): BPM missing from the stated precision contract → added to
schemas/README.md + CLAUDE.md; hysteresis ran on raw config/series while
params recorded rounded values → regions now computed from the document's
own rounded envelope and thresholds, making them exactly derivable from
published values; duplicate vocals-RMS compute removed, mix-envelope
duplicate documented as deliberate; dead test loop removed. Round 4: clean
except one nit (BPM missing from `canonical.py`'s inline precision comment)
— deferred to the F-1/F-2/C-1 follow-up PR. Operational note: the round-4
run initially failed at the 50-turn cap (milestone-sized diff + four rounds
of history); a re-run of the same commit completed within the cap.
Turn-cap policy question left open pending operator decision.
