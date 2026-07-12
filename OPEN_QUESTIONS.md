# Open Questions — Phase 0

Every judgment call a reviewer might dispute, with the answer currently baked
into `PLAN.md` and the schemas. Changing an answer here changes the plan, not
code — nothing is implemented yet.

**Review status:** all sixteen questions were dispositioned (all agreed) in
`docs/reviews/001-phase0-plan.md`; resolutions are noted inline below where
the review changed the design (OQ-2, OQ-5, OQ-16).

---

**OQ-1 — What bytes define `track_id`?**
Options: hash of the original file bytes; hash of the decoded audio stream;
UUID assigned at ingest.
**Recommendation: hash of the original file bytes** (first 16 hex chars of
SHA-256). It is trivially reproducible, needs no decode policy, and makes
"same file ⇒ same track" exact. Cost: a remux or re-encode of the same song
creates a new track_id. Mitigation already planned: we also store the decoded
PCM hash (`audio_stream_sha256`) so ingest can *warn* "this audio already
exists as track X" without complicating identity.

**OQ-2 — Copy media into the library, or reference the original path?**
**Recommendation: copy the normalized audio (FLAC) in; reference the original
by path but never re-read it after ingest.** The library stays self-contained
and analysis stays reproducible even if the source file moves — at ~25 MB per
track. The original *video* is the exception: copying it would double library
size, so video re-analysis requires the original path to still be valid. Flag
for review: is "video re-runs may break if the file moves" acceptable, or
should video inputs be copied too (config flag)?
**Resolution (review 001, R-7):** agreed; path-staleness is acceptable, and
`ingest.copy_video` (default `false`) is added to the config design now as
the escape hatch rather than retrofitted later.

**OQ-3 — Are stems retained on disk?**
**Recommendation: retain by default (16-bit FLAC, ~40–60 MB/track),
`stems.retain: false` to discard after dependent stages run.** Stems are the
most expensive artifact to recompute and the most likely to be wanted later
(feature re-runs, future iPad vocal-stem effects). Personal-library scale
makes the storage trivial.

**OQ-4 — Downbeats: honest tracker or 4/4 estimate?**
madmom's DBN downbeat tracker is the quality choice, but its trained models
are **CC BY-NC-SA** (poison for any future commercial product) and the
package fights Python 3.11. **Recommendation: ship v1 with the
`assumed_4_4_phase_fit` estimate, clearly labeled in the schema
(`downbeat_method`), and keep `"model"` reserved for a clean-licensed tracker
later** (e.g. Beat This, MIT-ish — needs a license/maturity check). Most of
the target material is 4/4 pop; the schema already tells consumers when bar
indices are approximate.

**OQ-5 — Vocal-activity thresholds.**
Enter −35 dBFS / exit −45 dBFS hysteresis, min region 300 ms, min gap 200 ms
— plausible but untested defaults. **Recommendation: accept as config
defaults and calibrate during M3 against a handful of real tracks**; the
schema is agnostic to the values. Reviewer question: should thresholds be
relative to the stem's own loudness rather than absolute dBFS? (Probably yes
eventually; absolute is simpler and stems from the same separator are fairly
consistent.)
**Resolution (review 001, R-5):** agreed for v1; the threshold values are now
embedded in the document as `vocal_activity.params` so regions are
interpretable across tracks. Revisit relative thresholds only if M3
calibration shows separator output levels vary more than expected.

**OQ-6 — Lyric alignment method.**
Options: fuzzy-anchor supplied lyrics to Whisper's word stream (chosen);
Montreal Forced Aligner; trust `.lrc` timestamps outright.
**Recommendation: Whisper-anchor with rapidfuzz.** MFA is a heavy non-Python
toolchain tuned for speech, weak on singing without custom acoustic models;
`.lrc` timestamps are line-granular and frequently offset (they become
search-window hints only). Known weakness: passages Whisper can't transcribe
can't anchor the supplied text either — those lines surface as `unaligned`
with interpolated timing rather than fake precision.

**OQ-7 — Whisper engine: faster-whisper (CPU) vs mlx-whisper (Apple GPU/ANE)?**
**Recommendation: faster-whisper, model `small`, int8.** Mature word
timestamps, deterministic decode, and transcription is not the pipeline
bottleneck (Demucs is), so mlx-whisper's 2–4× speedup buys little. The engine
sits behind one function; swapping later is cheap. Related default to
confirm: is `small` the right size, or should `medium` be the default for
sung vocals at ~2–3× the runtime? (Recommend: `small` default, `medium` as
documented config for hard material.)

**OQ-8 — Shot detector: PySceneDetect ContentDetector vs TransNetV2?**
**Recommendation: PySceneDetect for v1.** Deterministic, dependency-light,
strong on the hard cuts that dominate music videos. TransNetV2 is the upgrade
path for dissolve-heavy videos — before adopting it, verify the weight
license (repo is MIT; weights need confirmation). The `detector` block in
`video.json` already names the detector, so documents stay self-describing
across a future swap.

**OQ-9 — Palette extraction: color space and sampling.**
**Recommendation: k-means (k = 5, fixed seed) in CIELab over the 1–2
representative frames, downscaled.** Lab distance approximates perceptual
difference, which is what "dominant color" means. Disputed alternatives:
sampling many frames per shot (better for shots with lighting changes, ~10×
cost) and median-cut (faster, worse). Reviewer question: should swatches also
carry Lab or HSL values for the mapping engine, or is hex + proportion enough?
(Recommend: hex only; conversion is trivial and one representation avoids
drift.)

**OQ-10 — Captions vs determinism.**
API captions are not strictly deterministic, conflicting with the
byte-identical rule. **Recommendation: define caption determinism as
"deterministic given cache"** — first call is cached by
(frame_sha256, prompt_version, model); re-runs replay the cache byte-for-byte;
changing prompt_version or model is a deliberate cache miss. The alternative
(dropping captions from the determinism contract entirely) makes `video.json`
unauditable. Accepting this means the determinism CI check must run with a
warm or mocked cache.

**OQ-11 — Section label vocabulary.**
The schema fixes an enum: `intro, verse, chorus, bridge, outro, instrumental,
section_a..d`. **Recommendation: keep the closed enum** — both consumers
(Swift `Codable`, LLM prompting) benefit from a fixed vocabulary, and
`section_a..d` is the honest fallback when lyric evidence is absent. Costs:
no `pre_chorus`/`drop`/`refrain`; genre-specific structure gets flattened.
Adding labels later is a minor (additive) schema bump if consumers decode
unknown labels leniently — worth stating in the Swift decoding guidelines.

**OQ-12 — "On-beat" cut window: ±70 ms?**
Perceptual audiovisual synchrony tolerance is roughly 50–100 ms, so 70 ms is
a defensible middle. **Recommendation: keep ±70 ms as the *reported* default
and let consumers reinterpret** — the per-cut `offset_seconds` list exists
precisely so the mapping engine can apply its own definition. Alternative
(beat-relative window, e.g. ±15% of the beat period) adapts to tempo but is
harder to explain; noted for review.

**OQ-13 — How hard is the determinism guarantee on MPS?**
CPU: byte-identical everything, guaranteed. MPS: kernels are not all
deterministic; float jitter in Demucs/feature inputs is absorbed by the
schema-level rounding for *documents*, but stem *files* may differ at the
sample level between runs. **Recommendation: contract = document byte-identity
on both devices; stem-file byte-identity on CPU only; manifest records the
device.** The alternative (CPU-only by default for guaranteed identity)
costs 4–6× on the slowest stage and isn't worth it for a creative tool.

**OQ-14 — Float precision contract.**
Seconds 3 dp (1 ms), dB 2 dp, Hz 1 dp, ratios/confidences 3 dp — chosen as
"more precise than any consumer needs, coarse enough to absorb float
nondeterminism." Reviewer check: is 1 ms fine enough for beat times feeding a
renderer? (At 120 BPM a beat is ~500 ms; 1 ms is 0.2% of a beat — yes.)

**OQ-15 — Library root and config format.**
Not specified by the spec. **Recommendation:** library root defaults to
`./library` relative to the working directory, overridable by
`--library` / `MRW_LIBRARY`; config is a single TOML file (`mrw.toml`,
`--config` to point elsewhere) whose parsed, canonicalized JSON is what gets
hashed per stage. TOML over YAML for stdlib parsing (`tomllib`) and less
footgun surface.

**OQ-16 — Where do lyrics *snapshots* of copyrighted text live?**
Supplied lyric files are copied into the track dir and embedded in
`lyrics.json`. Fine for a private local library; would be a redistribution
problem if libraries are ever shared/synced. **Recommendation: accept for
v1; note in the eventual README that track directories contain copyrighted
text/transcriptions and must not be published.** No schema change needed now.
**Resolution (review 001, R-8):** agreed and extended — the same
do-not-publish caveat covers `frames/` (stills from a copyrighted video) and
`stems/` (derivative audio), not just lyric text. One README line covers all
three.
