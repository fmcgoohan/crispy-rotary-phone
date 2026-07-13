# Review 007 — M4: lyrics (milestone verification)

**Scope:** PR #9 as merged (`efb195f`), `mrw/align.py` read in full,
`mrw/lyrics.py` prerequisite/flag paths read, fast suite executed
independently.
**Verdict:** **M4 approved.** F-1 is a one-paragraph documentation item;
F-2 is a P1 record-keeping obligation on Claude Code.

## Independent verification

- Fast suite fully green in an external torchless environment (34 passed) —
  including every aligner unit test, which is the aligner-isolation
  requirement paying off exactly as intended: the alignment logic is now
  provable without Whisper, torch, or audio.
- Slow-path verification (Whisper transcription, espeak ground truth) rests
  on CI, whose environment installs espeak-ng and caches the Whisper model;
  the external sandbox cannot reach the model host. Division of evidence
  noted and accepted.
- `run_lyrics` prerequisite ordering is exemplary T5: cheap validation with
  fix-command error messages, no-op decision, and only then heavy imports.
- `align.py` verified: cursor monotonicity (fixing PR #9 review finding H2,
  cited in a code comment — a convention worth keeping), timeline-wins
  stable sort matching the schema invariant, and interpolation that floors
  at 0.0 and bounds its guesses, so the internal −1.0 placeholder cannot
  reach a document.
- Verification footnote: an initial 4-test failure in the external
  environment was a missing rapidfuzz caused by the verifier's own
  `--no-deps` install shortcut. The repo was green throughout.

## Findings

### F-1 — Record the tunables classification for align.py — [doc only]

Per the PR #7 tunables convention: `min_anchor_score` is correctly a
`LyricsConfig` member. `LOOKAHEAD_WORDS` (400), `HINT_BEFORE/AFTER_SECONDS`
(5/20), and the interpolation bounds (+4 s guess, 10 s cap) remain module
constants — defensible as structural search bounds riding `tool_version`,
but the classification should be *stated* (one comment block in align.py or
a line in this file's follow-ups), not implicit. Changing any of them later
without a config field would silently alter outputs under an unchanged
config hash; the stated classification is what makes that a conscious
decision.

### F-2 — Unexplained revert/re-land on main — [P1, action required]

Commits `2db7fc8` → `bc6957e` (revert) → `ee93a47` (re-land) put a
byte-identical CI change through a revert cycle with no rationale recorded
in any message. Only the Actions logs and the operating session know what
actually failed. Add a post-merge note under this review stating the cause
(suspected: interaction between a mid-review workflow change on main and
PR #9's in-flight review, or a transient CI failure misattributed to the
change) and, if the cause was the mid-review interaction, add the norm to
CHECKLIST P-section: reviewer-infrastructure changes land on main only
between PR review cycles.

Also fold the PR #9 review-round summary into this record per P1.

## Post-merge validation (with Harvey)

Studio tracks: expect mostly clean transcribed documents; judge text
quality by skim. Live track: judged on HONESTY — flags, low coverage, and
untranscribed_regions where the crowd wins; confidently hallucinated text
is the finding of the milestone. Report `mode`, `lines_flagged_ratio`, and
`vocal_activity_covered_ratio` per track.

---

**Next:** post-merge validation, then M5 (video analysis) planning.

## Post-merge note: F-2 cause (P1)

The reviewer's suspected causes are both wrong; the actual cause was an
operator-session error, recorded here verbatim from the session: the M4
implementation was left **uncommitted in the working tree** while switching
to main to land the CI-only change, and `git add -A` on main swept the
entire M4 tree (plus a stray editor `.swp`) into `2db7fc8`. That landed M4
on main unreviewed and left the PR branch with an empty diff. `bc6957e`
reverted the accident; `ee93a47` re-landed only the CI pieces (the revert
rationale was recorded in ee93a47's message, but not back-referenced from
this record until now — hence F-2). M4 then landed through PR #9 as
intended. The norm this actually teaches (added to CHECKLIST P-3 in the
field-fixes PR): commit the feature branch before any checkout of main;
a main-bound infra commit must contain only its stated files, verified
against `git status` before `add`.

## M4 milestone record (automated PR review summary, per P1)

PR #9 went through three automated review rounds. Round 1 (1 blocker,
3 major, 1 minor): the reviewer independently reproduced a "passes
locally, fails on CI" divergence — the degenerate no-vocals test asserted
an environment-dependent hallucination count (2 on Apple Silicon vs 3 on
CI's x86 separation output), and one hallucinated timestamp ran past the
clip's end. Fixes: the test now asserts stable honesty properties (every
hallucinated line flagged, timestamps in-domain, valid document), and the
stage clamps all segment/word times to [0, duration] — timeline validity
is the stage's job, content honesty the flags'. Majors: Whisper thread
pinning (cpu_threads=1, the stems precedent), hop_seconds read from the
document's own embedded series rather than a parallel constant, and a
missing T4 mid-stage-failure test (added). Minor: rounding through
canonical helpers. Round 2 (1 minor): out-of-order `.lrc` hints could
drag the alignment cursor backward and emit lines[] unsorted — fixed with
cursor monotonicity + a timeline-wins stable sort, tested. Round 3: clean
— but silent: the run succeeded with no new findings and did not post the
"no findings" summary comment it is prompted to leave. Operational note:
a green review run with no new comments is a clean pass; the missing
courtesy summary is a known reviewer quirk.

## Post-merge field findings (2026-07-13, first real-track runs)

Recorded per operator report; items 1–3 and 6 fixed in the m4-field-fixes
PR (lyrics schema 1.1.0, additive); 4 is an investigation with a config
knob landed; 5 is calibration data only.

1. **Language detection on silent stem heads** — detection ran on a
   near-silent head and locked onto Welsh (a known Whisper noise-
   attractor), hallucinating repetition loops timestamped inside regions
   audio_features declares instrumental. Fix: with no pin, detect on a
   window assembled from the earliest vocal_activity regions (features is
   already a prerequisite); provenance recorded via additive
   `engine.language_source: pinned | detected_vocal_window`.
2. **`outside_vocal_activity` line flag** — lines whose span overlaps no
   vocal-activity region are flagged (additive enum member), never dropped.
3. **`untranscribed_regions` sharpened to uncovered spans** — word-covered
   intervals subtracted from vocal-activity regions, spans ≥
   `lyrics.uncovered_min_seconds` (default 1.0) emitted. Field evidence: an
   18 s missed opening verse was invisible because its merged region was
   partially covered. Same shape; bump-level argued in the PR (additive
   minor — no external consumers exist yet).
4. **en-pin verse drop (open investigation)** — with `language = en`,
   42–60 s of clean lead vocal produced no words, though the unpinned run
   transcribed it. Cannot be reproduced synthetically; the suspected gate
   (`no_speech_threshold`) is now a `LyricsConfig` member
   (`decode_no_speech_threshold`, default 0.6 = engine default) so decode-
   setting experiments are config-hash events. Needs the field track to
   close.
5. **`possibly_non_lexical` calibration (no tuning yet)** — fired on 41/47
   substantially-correct lines and scored the Welsh hallucination run
   similarly: zero discrimination on sung material. Mechanism: chorus
   repetition legitimately inflates compression ratio. Candidate designs
   pending multi-track data: per-window compression instead of per-segment,
   or 2-of-3 signal logic (no_speech / compression / dictionary hit rate).
6. **Tests** — uncovered-span computation unit-tested pure (partial
   coverage → localized spans); synthetic out-of-region line asserts the
   new flag; language-window assembly unit-tested; espeak fixture gained a
   10 s silent lead-in asserting English detection with
   `detected_vocal_window` provenance.
