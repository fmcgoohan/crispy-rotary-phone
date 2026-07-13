# PR Review Checklist

Distilled from reviews 001–003. Applied by the automated PR reviewer on
every PR; the external reviewer handles milestone/design-level review only.
Findings reference the item they violate (e.g. "D3").

## Determinism (D)

- **D1** Analysis documents are pure functions of (input file bytes at a
  given path, per-stage config). Same input + config ⇒ byte-identical
  output. `track_id` is over file bytes alone (PLAN §7 scoping).
- **D2** All volatile data (timestamps, durations, hostnames, device, tool
  versions) lives in `manifest.json` only — never in analysis documents.
- **D3** Every new float field obeys the precision contract: seconds 3 dp,
  dB 2 dp, Hz 1 dp, ratios/confidences 3 dp.
- **D4** Every document write goes through `mrw/canonical.py` (atomic
  temp-and-rename, fixed key order). No direct `json.dump` of documents.
- **D5** Any new nondeterminism source (neural model, external API, RNG,
  thread-order float jitter) ships with an explicit policy: seed it, cache
  it (deterministic-given-cache), or round it away — stated in the doc/schema.

## Schema discipline (S)

- **S1** Hand-drafted schemas in `schemas/` are the contract; pydantic
  models in `mrw/models.py` match them field-for-field, order included.
- **S2** Additive schema change ⇒ minor version bump; breaking ⇒ major.
  `schema_version` present in every document.
- **S3** Fixed, schema-declared keys only (no dynamic maps); absent means
  absent (never `null`); enums are lowercase strings and consumers must
  decode unknown values leniently.
- **S4** Params are embedded next to the data wherever a consumer might
  reinterpret or compare values across tracks (thresholds, windows,
  normalization references); everywhere else the manifest `config_hash`
  suffices.
- **S5** Worked examples in `schemas/examples/` updated alongside schema
  changes and still validating (internally consistent numbers included).

## Honesty machinery (H)

- **H1** Uncertainty surfaces as `flags` / `confidence` / coverage fields in
  the documents — never silent inclusion of dubious data, never silent
  omission.
- **H2** Degenerate cases have defined, documented conventions: silent
  channels, missing streams, empty event lists, zero-length regions.
- **H3** Skips are recorded and distinguishable (`not_applicable` vs
  `pending` vs `failed` in the manifest); no runtime fallback masking — a
  failed prerequisite refuses downstream stages rather than improvising.

## Tests (T)

- **T1** A milestone PR lands with its PLAN §11 smoke test implemented and
  passing — not promised for later.
- **T2** Determinism claims are tested by double-run byte comparison (two
  library roots or two invocations), not asserted.
- **T3** Fixtures are generated in-test (tones, click tracks, synthetic
  video) — never copyrighted media, never binary blobs checked into the repo.
- **T4** Failure paths are exercised (bad input → correct exit code, intact
  library, `failed` status recorded).
- **T5** The fast suite (`-m 'not slow'`) must pass in an environment
  without the neural dependencies installed; heavy imports happen lazily
  and after cheap prerequisite validation.

## Two-consumer check (C)

- **C1** Swift `Codable`: stable snake_case keys, homogeneous arrays, no
  null-vs-absent ambiguity, no type changes to existing fields.
- **C2** LLM reader: self-describing structure, units stated inline,
  provenance/method fields explicit (it should be safe to hand the document
  to a model with no other context).

## Process (P)

- **P1** Findings are numbered with severity; disputes are recorded under
  the finding ID in the relevant `docs/reviews/NNN-*.md`.
- **P2** Decisions that change a recommendation get a resolution note in
  `OPEN_QUESTIONS.md`.
- **P3** A main-bound infrastructure commit contains only its stated files:
  commit the feature branch before any checkout of main, and verify
  `git status` before `add` (the 2db7fc8 lesson — review 007 F-2).
