# Review 005 — PR #4: M2 stem separation (automated review record)

**Scope:** PR #4 (`m2-stems`, merged as `b1edf8e`) — Demucs htdemucs wrapper,
retention semantics, `mrw models fetch`, manifest schema 1.1.0, M2 smoke
tests. Reviewed by the automated PR reviewer over three rounds; this file is
the milestone record of its findings and the process notes.

## Findings (all fixed in-PR)

- **Round 1 [major] H2** — silent/DC-only input drove `ref.std()` to 0 and
  the demucs.separate normalization produced NaN in every stem, undetected.
  Fixed: (near-)silent input separates unnormalized, plus an `isfinite`
  guard that fails the stage rather than writing garbage; slow test added
  (3 s silence → finite, near-silent stems).
- **Round 1 [major]** — `mrw stems` triggers the first-use weight download,
  but CLAUDE.md's offline-by-default exception list named only the caption
  backend. Fixed: CLAUDE.md now lists both permitted network exceptions
  (captioning; first-use model-weight downloads per PLAN §10).
- **Round 2 [major] H2** — `retain: true → false` transition left the
  previously-retained `stems/` on disk while the manifest recorded
  `retained: false`. Fixed: the retain=false path removes a pre-existing
  `stems/`; transition test added.
- **Round 2 [minor] D5** — `torch.set_num_threads(1)` is process-global and
  unrestored. Documented in code: combined-stage runs must set their own
  threading policy after separation.
- **Round 3** — clean; 18 tests passed on CI including all slow tests.

## Post-merge notes (P1)

1. **Reviewer-infrastructure changes cannot ride in reviewed PRs.**
   claude-code-action validates that the workflow file on a PR is
   byte-identical to main's version (anti-tampering) and skips itself
   otherwise. The M2 caching change had to land on main first (`754bb40`);
   the same applies to any future prompt/permission tuning of
   `claude-review.yml`.
2. **Turn cap raised 25 → 50** (`6a5adae`). The M2 review hit
   `error_max_turns` at 25 without posting anything — milestone-sized diffs
   plus `uv sync` + slow pytest don't fit in 25 turns. `timeout-minutes: 20`
   remains the cost backstop; the prompt now instructs the reviewer to post
   findings before exhausting its budget.
3. **CI cost measurement (M2 review decision):** cached review-job time is
   ~7m25s (test portion ~2 min; the review conversation dominates) — under
   the ~10-minute threshold, so the slow Demucs tests stay in the review
   workflow. The CPU double-run is the neural-path determinism evidence.
   Revisit when M3+ grows the suite.
