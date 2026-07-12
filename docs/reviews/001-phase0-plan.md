# Review 001 — Phase 0 plan & schemas

**Scope:** `PLAN.md`, `OPEN_QUESTIONS.md`, `schemas/*` (all read in full).
**Verdict:** Architecture and plan **approved**. Schema revisions **R-1 through R-5 requested before M0** freezes the pydantic models; R-6 through R-8 recommended, author's discretion on timing. All sixteen open questions are dispositioned below; none reopens the architecture.

---

## Confirmed right — do not "fix" during revision

- `manifest.json` as the only mutable file, with **per-stage `config_hash`** so unrelated config edits don't stale everything. Keep exactly as designed.
- `untranscribed_regions` + per-line flags + coverage ratios. This is the most important feature of `lyrics.json`; the LLM consumer depends on it.
- Caption **absent vs `backend: "null"`** distinction, and cache-keyed caption determinism.
- `canonical.py` as the single JSON write path, with rounding as the stated precision contract.
- The madmom **CC BY-NC-SA** catch and the `downbeat_method` honesty field. Ship `assumed_4_4_phase_fit`.
- Fixture-only test media; no copyrighted bytes in the repo.
- Separate documents per stage. The rationale in PLAN §5 is correct on all four counts.

---

## Findings

### R-1 — Onset `strengths` normalized to channel max (audio_features) — **revise**

Normalizing to this channel's max makes every strength hostage to the single
loudest onset (one snare accent compresses the whole track toward 0) and makes
values non-comparable across tracks — which matters because scenes/mappings
are meant to be **reusable across tracks**; "trigger at drum strength > 0.6"
should mean roughly the same thing everywhere.

**Requested change:** normalize by a robust reference — p98 of the channel's
onset-strength envelope — and clamp to [0, 1]. Record the reference value
(raw envelope units) in the onsets block (e.g. `strength_reference`) so raw
values are recoverable. Deterministic either way; strictly better behaved.

### R-2 — Centroid sentinel `0.0` contradicts the interpolation convention (common + audio_features) — **revise**

`timeseries` promises "sample at arbitrary t by linear interpolation," but
`spectral_centroid_hz` writes `0.0` for silent frames. Interpolating across a
silence boundary then yields a bogus sweep toward 0 Hz — a mapping engine
driving "brightness" gets a flicker artifact at every silence edge, exactly
where visuals are most noticeable.

**Requested change:** hold the last valid centroid through silent frames (no
sentinel; series is everywhere-interpolable). Silence itself is already
knowable from `rms_db`, which is the right signal for "is anything playing."
Note the hold-last behavior in the field description. (Alternative — declaring
per-series sentinels and forbidding interpolation across them — complicates
the one convention that is currently beautifully simple. Don't.)

### R-3 — Supplied-lyrics section markup is unhandled, and it's free structure evidence (lyrics + structure) — **revise**

Real-world lyric `.txt` files very often contain `[Verse 1]` / `[Chorus]`
header lines. As drafted: (a) the aligner will treat them as sung lines, so
they all surface as `unaligned` and pollute `coverage`; (b) gold-standard
structure evidence is discarded — the `evidence` enum cannot express it.

**Requested change:** the lyrics stage detects bracketed/header lines,
excludes them from `lines[]` and from coverage math, and records them with
their positions. Add `supplied_markup` to `structure.sections[].evidence`
(additive enum change) and let it dominate label assignment when present.
Zero runtime cost; material quality win on exactly the inputs users will
actually supply.

### R-4 — Required-field oversights — **revise**

- `audio_features`: `loudness` is always computable but absent from
  `required`. Add it.
- `lyrics`: `engine` is always known but absent from `required`. Add it.

If either omission was deliberate, say why in the schema description instead.

### R-5 — Parameter self-description is inconsistent across documents — **revise (policy + one block)**

`video.json` embeds its `detector` params; `structure.json` embeds
`on_beat_window_seconds`; `audio_features.json` embeds nothing — the
vocal-activity hysteresis thresholds (which OQ-5 explicitly expects to change)
are invisible to a consumer comparing two tracks' regions.

**Requested change:** state the policy once in the schema README — *embed a
small params block wherever a consumer might need to reinterpret or compare
the field; otherwise `config_hash` suffices* — and, under that policy, add the
enter/exit/min-region/min-gap values to `vocal_activity`.

### R-6 — `editing.per_section` is a position-parallel array (structure) — minor

Position-parallelism is the one place the schemas violate their own
self-description ethos; a consumer that filters or re-sorts sections silently
breaks the linkage. Add `section_index` to each entry. Cheap now, annoying
later.

### R-7 — Promote the OQ-2 mitigation into config now — minor

Accept path-staleness for video re-analysis, but add `ingest.copy_video`
(default `false`) to the config schema in this phase so the escape hatch is
designed rather than retrofitted. Fail-loudly already covers the error path.

### R-8 — Extend the OQ-16 copyright note beyond lyric text — minor

`frames/` (stills from a copyrighted video) and `stems/` (derivative audio)
carry the same do-not-publish caveat as lyric snapshots. One line in the
eventual README covering all three.

---

## Open-question dispositions

- **OQ-1** (track_id = original file bytes): **agree**. The
  `audio_stream_sha256` dedup hint is the right mitigation.
- **OQ-2** (copy audio, reference video): **agree**, with R-7.
- **OQ-3** (retain stems by default): **agree**.
- **OQ-4** (4/4 phase-fit downbeats, madmom rejected on license): **agree**.
  Verify the Beat This weight license before any future adoption.
- **OQ-5** (absolute dBFS thresholds): **agree for v1**, calibrate in M3;
  embed the values per R-5. Revisit relative thresholds only if calibration
  shows separator output levels vary more than expected.
- **OQ-6** (rapidfuzz Whisper-anchoring over MFA): **agree**.
- **OQ-7** (faster-whisper `small` default): **agree**. During M4, run one
  informal small-vs-medium comparison on a real track to sanity-check the
  default; not a CI gate.
- **OQ-8** (PySceneDetect v1, TransNetV2 upgrade path): **agree**.
- **OQ-9** (palette hex only): **agree** — one representation, conversion is
  the consumer's one-liner.
- **OQ-10** (captions deterministic-given-cache): **agree**; the contract is
  honest and auditable. M7's mocked-HTTP double-run is the right check.
- **OQ-11** (closed section-label enum): **agree**; keep `section_a..d` as the
  honest fallback and require lenient unknown-label decoding in the Swift
  guidelines.
- **OQ-12** (±70 ms reported window + per-cut offsets): **agree** — the
  per-cut list makes the window advisory, which is the correct design.
- **OQ-13** (document identity both devices; stem-file identity CPU only):
  **agree**. Pragmatic and clearly recorded in the manifest.
- **OQ-14** (1 ms beat-time precision): **confirmed sufficient** — a 120 fps
  frame is 8.3 ms; 1 ms is far below any consumer's resolution.
- **OQ-15** (TOML config, `./library` default): **agree**.
- **OQ-16** (copyrighted text snapshots): **agree**, extended per R-8.

## One observation, no action

`audio_features.json` at ten dense series (~1.5–2 MB) is fine: the
separate-documents decision already keeps it out of LLM context, and a one-time
`JSONDecoder` load is acceptable on iPad. If it ever hurts, the escape hatch
is a binary sidecar for `values` arrays — a minor schema bump, not a redesign.
Do not build it now.

---

**Next step:** apply R-1..R-5 (and R-6..R-8 as desired) to the schemas and the
relevant PLAN sections, then proceed to M0. No re-review needed unless a
finding is disputed — dispute by adding a response under the finding ID.
