# Review 004 — PR #1: review automation

**Scope:** PR #1 (`claude-review.yml`, `CLAUDE.md`, `docs/reviews/CHECKLIST.md`)
plus the 003 doc fixes that landed directly on main (verified: F-1 in PLAN §7,
F-2 as README "Do not publish library contents").
**Verdict:** **Approve.** A-1 and A-3 are two-line YAML edits — make them on
the branch before merge since this PR defines the reviewer. A-2 is a
verify-on-first-run item.

## Confirmed right

- Safe trigger model: plain `pull_request` (never `pull_request_target`), no
  `issue_comment`/mention entry point, no `allowed_non_write_users`.
- Minimal permissions (`contents: read`), tightly scoped `--allowedTools`,
  `--max-turns 25`, `timeout-minutes: 20`, explicit no-approve/no-push/no-edit
  limits in the prompt.
- ffmpeg + uv installed before the action so T1/T2 checks can actually run.
- CHECKLIST.md is a faithful distillation of reviews 001–003 with citable
  item IDs; CLAUDE.md captures the hard constraints including the F-1 scoping.

## Findings

### A-1 — Fork PRs will fail red instead of skipping — [minor]

GitHub does not expose repository secrets to workflows triggered by fork
PRs, so `ANTHROPIC_API_KEY` will be empty and the job will fail, leaving a
confusing red X on any outside contribution. Add a same-repo guard on the
job:

    if: github.event.pull_request.head.repo.full_name == github.repository

Outside contributions then simply get no automated review (fine — they get
manual review), with no failed check.

### A-2 — Summary comment permission — [minor, verify on first run]

`gh pr comment` posts via the issue-comments API; with only
`pull-requests: write` some setups 403 on it while inline review comments
succeed. On the first live run, check the summary comment posted; if it
403s, add `issues: write` to the job permissions. Not adding it
preemptively is defensible (least privilege) — just don't be surprised.

### A-3 — Stacked runs on rapid pushes — [nit]

Add a concurrency group so a new push cancels the in-flight review instead
of paying for both:

    concurrency:
      group: pr-review-${{ github.event.pull_request.number }}
      cancel-in-progress: true

## Threat-model note (for the record)

For same-repo PRs the API key is present in the runner while tests execute
PR code — acceptable because same-repo PRs require write access. Fork PRs
never see the secret. Revisit only if collaborators are ever added.

---

**Next:** merge after A-1/A-3, then M1 as a PR — the first one the automated
reviewer handles. Compare its findings against CHECKLIST expectations; tune
the prompt if it's noisy. External review moves to milestone boundaries.

## Post-merge note (trigger model, 2026-07-12)

`/install-github-app` added two workflows to main outside PR review: a
duplicate generic auto-reviewer (`claude-code-review.yml`, removed in
`ae55911`) and an `@claude` mention workflow (`claude.yml`) — the
issue_comment/mention entry point this review's "confirmed right" section
records as absent. Decision: deleted. The trigger model stands as reviewed:
`pull_request` automation only, no mention entry point, write-access users
only.
