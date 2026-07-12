# CLAUDE.md

mrw — the Mac-side analysis workbench for a music-reinterpretation project.
Python CLI that turns a local song (audio or music video) + optional lyrics
into deterministic, versioned analysis documents in a local library.
Architecture and milestones: `PLAN.md`. Document contracts: `schemas/`.

## Hard constraints (never violate)

- **Source-agnostic ingest.** Input is a local media file. No downloading,
  no network acquisition of media of any kind, ever.
- **Offline by default.** Two permitted network exceptions: (1) the
  pluggable vision-captioning backend (Anthropic API) behind the
  `CaptionBackend` interface, defaulting to a no-op, API key from
  `ANTHROPIC_API_KEY`; (2) first-use model-weight downloads (Demucs, later
  Whisper) from their official distribution — pre-fetch with
  `mrw models fetch` so batch runs never surprise-download (PLAN §10).
  Never any network acquisition of media or analysis content.
- **Determinism.** Same input + same config ⇒ byte-identical analysis
  documents. `manifest.json` is the one file allowed to differ between runs
  — ALL volatile data (timestamps, durations, host, device, tool versions)
  goes there and nowhere else. `source.json` byte-identity is scoped to
  re-runs at the same path (PLAN §7); `track_id` is over file bytes alone.
- **Canonical serializer only.** Every document write goes through
  `mrw/canonical.py` — never `json.dump` directly. Precision contract:
  seconds 3 dp, dB 2 dp, Hz 1 dp, BPM 1 dp, ratios/confidences 3 dp.
- **Platform.** macOS on Apple Silicon, Python 3.11+, environment managed
  with `uv`. Prefer mature, well-maintained libraries.
- **No copyrighted media** in the repo or tests — fixtures are generated
  (tones, click tracks, synthetic video). Library contents are private
  (see README "Do not publish").

## Commands

- `uv sync` — install dependencies
- `uv run pytest -q` — run tests (need `ffmpeg`/`ffprobe` on PATH)
- `uv run mrw ingest <file> [--lyrics f] [--title t] [--library dir]`
- `uv run mrw status` / `uv run mrw export-schemas`

## Conventions

- Hand-drafted schemas in `schemas/` are the reviewed contract; keep the
  pydantic models in `mrw/models.py` in field-order lockstep with them.
  Serialization is absent-not-null (`exclude_none=True`); enums are
  lowercase strings; uncertainty surfaces as `flags`/`confidence` fields,
  never silently.
- Reviews live in `docs/reviews/NNN-*.md` with numbered finding IDs
  (e.g. R-1, F-2). Disputes are recorded under the finding ID; decisions
  that change a recommendation get a resolution note in
  `OPEN_QUESTIONS.md`. PR-level standards: `docs/reviews/CHECKLIST.md`.
- Milestones follow PLAN §11; each milestone PR includes its smoke test,
  and determinism claims are tested by double-run byte comparison.
- Exit codes: 0 success, 1 stage failure, 2 bad invocation.
