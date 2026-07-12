# Review 002 — Schema revisions (follow-up to 001)

**Scope:** revised `schemas/*`, `README.md`, worked examples under `examples/`.
**Verdict:** **Approved to proceed to M0.** R-1 through R-6 are applied
correctly; the two items below are one-line schema-description additions that
can land in the same commit as the M0 skeleton. No re-review needed.

## Revision verification

- **R-1** ✅ `strength_reference` (p98, raw units) + clamped strengths;
  recoverability documented; drums example shows a clamped 1.0 accent.
- **R-2** ✅ hold-last centroid with leading-silence backfill — the backfill
  detail goes beyond what was asked and is correct.
- **R-3** ✅ `supplied_markup` block (excluded from lines/coverage, normalized
  labels, `.lrc` hints) + `supplied_markup` in the structure evidence enum,
  documented as dominating label assignment.
- **R-4** ✅ `loudness` and `engine` now required.
- **R-5** ✅ params-embedding policy in the README; `vocal_activity.params`
  embedded.
- **R-6** ✅ `per_section[].section_index`.
- **R-7 / R-8** — not verifiable from this drop (PLAN/config and the repo
  README weren't included). Confirm `ingest.copy_video` landed in the config
  design and that the do-not-publish note covers `frames/` and `stems/` as
  well as lyric text.

Worked examples were checked numerically: beat times follow
0.612 + 0.4992·i; all section boundaries land on downbeats with correct bar
indices; cut offsets are consistent with the grid; shot frame indices are
consistent with 23.976 fps. Treat these examples as normative going forward.

## New findings

### R-9 — Define the fully-silent-channel convention — **required, one commit**

An instrumental track's vocals stem never crosses the silence floor. As
revised, two fields have no defined value there:

1. `spectral_centroid_hz`: hold-last/backfill has no valid frame to hold.
   **Define:** a channel with no valid frames writes `0.0` for the entire
   series, and one sentence in the field description says so ("a channel that
   is silent throughout writes all zeros; check `rms_db` before using
   centroid").
2. `onsets.strength_reference`: p98 of an all-zero envelope is 0, violating
   `exclusiveMinimum: 0`. **Define:** when `times` is empty, write
   `strength_reference: 1.0` (documented sentinel) — or relax the schema to
   allow omission when `times` is empty. Either is fine; pick one and state it.

This will otherwise surface as a schema-validation failure in M3 on the first
instrumental fixture.

### N-1 — Bar index before the first downbeat — nit

The intro section starts at beat 0 with `start_bar_index: 0`, but the bar
formula is only defined for `i >= downbeat_offset`. The example does the right
thing; the schema doesn't say it. Add to `beats` (or `start_bar_index`): "bar
index clamps to 0 for beats before the first downbeat."

### N-2 — Use a real pinned model id in `examples/video.json` — nit

`caption_backend.model: "claude-sonnet-5"` is not a real model string.
Examples get copy-pasted into configs; use an actual pinned id (e.g.
`claude-sonnet-4-6`).

---

**Next step:** fold R-9 (+ nits) into the M0 commit and start building. The
schema review loop is closed; future reviews move to implementation PRs.

---

## Author response (commit follows this review)

- **R-7 / R-8 confirmation:** both landed in commit `97e5357`.
  `ingest.copy_video` (default `false`) is in the config design — PLAN.md
  §4 stage 1 bullet and OPEN_QUESTIONS OQ-2 resolution. The do-not-publish
  caveat covering `frames/` and `stems/` alongside lyric text is in
  OPEN_QUESTIONS OQ-16 (resolution paragraph); it will also appear in the
  repo README when one is written at M0.
- **R-9 applied** (option: sentinel, not omission): a fully-silent channel
  writes an all-zeros `spectral_centroid_hz` series, and an empty-onsets
  channel writes `strength_reference: 1.0` — both stated in the field
  descriptions in `audio_features.schema.json`.
- **N-1 applied:** bar-index clamp-to-0 before the first downbeat is now
  stated in the `beats` description.
- **N-2 applied with a correction:** `claude-sonnet-5` is in fact a valid
  current model ID (Claude Sonnet 5), per Anthropic's current model catalog —
  but the spirit of the nit is right, and the example now uses
  `claude-opus-4-8`, the canonical recommended vision-capable ID.
- Since M0 hasn't started, these changes ship as their own commit rather
  than folded into the M0 skeleton commit.
