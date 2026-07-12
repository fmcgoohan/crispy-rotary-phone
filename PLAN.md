# Music Reinterpretation Workbench — Analysis Pipeline Plan

**Phase 0 design document. No implementation exists yet.** This plan, the draft
schemas in `schemas/`, and `OPEN_QUESTIONS.md` are the review artifacts.

Working name for the tool/CLI: **`mrw`**.

---

## 1. Purpose and scope

`mrw` is the Mac-side analysis workbench for a future iPad app that
reinterprets songs as generative visuals. Given a local media file (video or
audio) and an optional lyrics file, it produces a set of versioned,
deterministic **analysis documents** (JSON) in a local library, one directory
per track. It is CLI-first and batch-oriented.

Out of scope for this repo: the storyboard/scene format, the audio→visual
mapping engine, and the Metal renderer. Those are *consumers* of our documents,
and two consumer profiles shape every schema decision:

1. **A Swift app decoding with `Codable`** — so: stable key names, homogeneous
   arrays, string enums, no `null`-vs-absent ambiguity, no dynamic keys.
2. **An LLM reading the documents to propose visual interpretations** — so:
   self-describing structure, human-readable labels, units stated inline,
   confidence and provenance made explicit.

## 2. Design principles

- **Documents are pure functions of (input bytes, config).** Same input + same
  config ⇒ byte-identical documents. All run metadata (timestamps, host,
  durations, tool versions) lives in exactly one place — `manifest.json` — so
  the analysis documents themselves never contain volatile data.
- **One timeline.** Every timestamp in every document is seconds (float) from
  t = 0 = first sample of the normalized audio decode. Beat and bar indices are
  *derivable* from the beat grid in `audio_features.json`; no document invents
  its own clock.
- **Everything is cheaply samplable at arbitrary `t`.** Each temporal
  representation is one of exactly three shapes, all O(1) or O(log n) to query:
  - *uniform-hop series* (envelopes): `index = (t − start) / hop`, linear
    interpolation between neighbors;
  - *sorted event list* (beats, onsets, cuts): binary search;
  - *sorted, non-overlapping interval list* (sections, shots, vocal-activity
    regions): binary search on start times.
  No nested or relative timing anywhere.
- **Stages are independent and resumable.** Each pipeline stage is a separate
  CLI subcommand that reads prerequisites from the library and writes one
  document. Re-running captioning never re-runs Demucs.
- **Uncertainty is data, not a log line.** Low confidence, hallucination
  suspicion, and estimation shortcuts are recorded as structured `flags` and
  `confidence` fields in the documents, never silently included or dropped.
- **Offline by default.** The only network touchpoint is the pluggable vision
  captioning backend, which defaults to a no-op.

## 3. Architecture overview

```
                    ┌──────────────────────────────────────────────┐
 media file ──────▶│ 1. ingest      ffprobe, hash, normalize audio │──▶ source.json, source_audio.flac
 lyrics file ─────▶│                copy lyrics, register track    │    manifest.json (created)
                    └──────────────────────────────────────────────┘
                                        │
              ┌─────────────────────────┼──────────────────────────┐
              ▼                         ▼                          ▼
     ┌────────────────┐      ┌──────────────────┐       ┌──────────────────┐
     │ 2. stems       │      │ 5. video         │       │ (audio-only:     │
     │ Demucs htdemucs│      │ shots, palettes, │       │  video stage     │
     │ → stems/*.flac │      │ motion, captions │       │  skipped, noted  │
     └────────────────┘      │ → video.json     │       │  in manifest)    │
              │              └──────────────────┘       └──────────────────┘
              ▼                         │
     ┌────────────────┐                 │
     │ 3. features    │                 │
     │ librosa et al. │                 │
     │ → audio_       │                 │
     │   features.json│                 │
     └────────────────┘                 │
              │                         │
              ▼                         │
     ┌────────────────┐                 │
     │ 4. lyrics      │                 │
     │ align or       │                 │
     │ transcribe     │                 │
     │ → lyrics.json  │                 │
     └────────────────┘                 │
              │                         │
              └───────────┬─────────────┘
                          ▼
                 ┌────────────────┐
                 │ 6. structure   │
                 │ sections +     │
                 │ cut/beat align │
                 │ → structure.json│
                 └────────────────┘
```

Stage dependencies: `features` needs `stems` (per-stem features);
`lyrics` needs `stems` (vocal stem) and `features` (vocal-activity regions to
sanity-check coverage); `structure` needs `features`, `lyrics` (if any), and
`video.json` (if video). `video` needs only `ingest`.

### Module breakdown

```
mrw/
  cli.py             # typer app: ingest / analyze / status / show
  config.py          # frozen dataclass config, per-stage subsets, config hashing
  library.py         # track dirs, manifest read/write, prerequisite checks
  canonical.py       # canonical JSON serializer (key order, float rounding, newline)
  hashing.py         # file & stream & frame hashing (sha256)
  timeseries.py      # uniform-hop series build/round/sample helpers
  ingest.py          # stage 1
  stems.py           # stage 2 (Demucs wrapper)
  features.py        # stage 3 (tempo/beats, onsets, RMS, centroid, LUFS, vocal activity)
  lyrics.py          # stage 4 (align supplied lyrics OR transcribe vocal stem)
  video/
    shots.py         # PySceneDetect wrapper, representative-frame extraction
    palette.py       # k-means palettes in CIELab
    motion.py        # optical-flow motion energy
    captions.py      # CaptionBackend protocol: NullBackend, AnthropicBackend
  structure.py       # stage 6 (sections, cut/beat alignment)
```

`canonical.py` is deliberately its own module: determinism lives or dies in
serialization, so there is exactly one code path that writes JSON.

## 4. Stage-by-stage plan

### Stage 1 — Ingest & split

- **Probe** with `ffprobe -print_format json`: container, duration, audio
  stream (codec, sample rate, channels), video stream if present (codec,
  dimensions, fps, frame count), embedded tags (title/artist, advisory only).
- **`track_id`** = first 16 hex chars of SHA-256 of the *original file bytes*.
  Full hash stored in `source.json`. We additionally store a hash of the
  decoded audio stream (`ffmpeg … -f s16le - | sha256`) so a future dedup check
  can recognize the same audio in a remuxed container, but the file hash is the
  identity (simple, stable, no decode ambiguity). See OQ-1.
- **Normalize audio**: decode to 44.1 kHz stereo and store as
  `source_audio.flac` (16-bit) inside the track directory, so the library is
  self-contained even if the original file moves. The original is referenced by
  path in `source.json`, never re-read after ingest. See OQ-2.
- **Original video is referenced, not copied**, by default; config
  `ingest.copy_video: true` copies it into the track dir for a fully
  self-contained library at the cost of size (review 001, R-7). Without the
  copy, video re-analysis fails loudly if the original path has gone stale.
- **Lyrics file** (optional): copied verbatim into the track dir as
  `lyrics_input.lrc` / `lyrics_input.txt`, hashed, and registered in
  `source.json`. `.lrc` timestamps are treated as *hints*, not truth (they are
  frequently offset); alignment still runs against audio.
- **Frame-access strategy** (video inputs), designed so we never do random
  full-res decode in hot loops — at most ~2 sequential decodes plus a handful
  of precise seeks:
  1. *Shot detection pass*: one sequential decode streamed through
     PySceneDetect (it downscales internally).
  2. *Motion pass*: one sequential decode via ffmpeg to grayscale 320-px-wide
    frames at 12 fps, piped as raw frames into numpy for optical flow.
  3. *Stills*: representative frames extracted with precise ffmpeg seeks
     (`-ss` fast-seek + accurate decode to the exact timestamp), max 1280 px
     wide, JPEG q90, stored under `frames/`.
- Audio-only input: manifest records `has_video: false`; the `video` and the
  video-dependent parts of `structure` are skipped *and recorded as skipped*
  (status `not_applicable`), not silently absent.

### Stage 2 — Stem separation

- **Demucs `htdemucs`** → `vocals / drums / bass / other`, written as 16-bit
  FLAC under `stems/`.
- **Retention: keep stems on disk by default, configurable off**
  (`stems.retain: true`). Why: stems are the single most expensive artifact to
  recompute (~1–2 min on MPS, ~5–10 min on CPU per track), and at ~40–60 MB
  per track as FLAC they are cheap to store at personal-library scale
  (100 tracks ≈ 5 GB). Downstream stages (features, lyrics) re-read them, and
  the iPad app may eventually want the vocal stem for effects. With
  `retain: false`, stems are written to a temp dir, consumed by dependent
  stages in the same `analyze` run, and deleted; the manifest records that
  stems are not resident so a later re-run knows to regenerate. See OQ-3.
- **MPS**: Demucs runs on `mps` when available (`--device mps` equivalent),
  CPU fallback otherwise with ~4–6× wall-clock penalty. Determinism caveat in §7.

### Stage 3 — Audio features

Interpretation note (recorded decision): tempo/beat grid is computed **once,
on the full mix** — per-stem tempo is not meaningful. Onsets, RMS, and
spectral centroid are computed per mix *and* per stem; vocal activity from the
vocal stem only.

- **Hop: 10 ms fixed time grid (100 Hz), i.e. hop = 441 samples at 44.1 kHz.**
  Justification: (a) it is a round, sample-rate-independent number the renderer
  can index with trivial arithmetic (`i = round(t * 100)`); (b) 100 Hz
  comfortably captures loudness envelopes (perceptual loudness integration is
  ~10× slower than that) while keeping a 4-minute track at ~24 k points per
  series (~150–200 KB as rounded JSON — acceptable, and gzip-friendly);
  (c) 5 ms would double size for no perceptual gain, 25 ms starts to smear
  drum transients in the envelope (though onsets, not the envelope, are the
  transient signal anyway). Analysis FFTs use window 2048 / hop 441.
- **Tempo & beat grid**: `librosa.beat.beat_track` on the mix (onset-strength
  based), emitting sorted beat times plus a global BPM and a confidence score.
  **Downbeats**: v1 estimates them by assuming 4/4 and choosing the beat phase
  (0–3) that maximizes mean onset strength — recorded with
  `"method": "assumed_4_4_phase_fit"` so consumers know it is an estimate.
  The honest downbeat tracker (madmom DBN) has a model-license problem
  (CC BY-NC-SA) and Python 3.11 packaging friction — see OQ-4.
- **Onsets**: `librosa.onset.onset_detect` per stem (drums are the load-bearing
  one for visuals) and for the mix; sorted times, plus per-onset strength
  normalized by the channel's p98 onset-envelope value (robust reference,
  recorded in the document as `strength_reference`) and clamped to [0, 1], so
  "drum strength > 0.6" means roughly the same thing across tracks and one
  outlier accent can't compress the rest toward 0 (review 001, R-1).
- **RMS/loudness envelope**: RMS in dBFS at the 10 ms hop, per mix and per
  stem, as uniform-hop series. Plus one scalar **integrated LUFS**
  (EBU R128 via `pyloudnorm`) for the mix, so the mapping engine can normalize
  across tracks.
- **Spectral centroid**: Hz at the same hop, per mix and per stem. Silent
  frames hold the last valid centroid (backfilled at the start) instead of a
  0.0 sentinel, so the series interpolates cleanly everywhere; "is anything
  playing" is `rms_db`'s job (review 001, R-2).
- **Vocal-activity regions**: hysteresis thresholding on the vocal stem's RMS
  envelope (enter at −35 dBFS, exit at −45 dBFS, min region 300 ms, min gap
  200 ms — all config), emitting sorted intervals with mean level. Threshold
  values are a tuning point — see OQ-5.

### Stage 4 — Lyrics

Two paths, one output document:

- **Lyrics file supplied → align.** Transcribe the vocal stem with Whisper
  (word timestamps) anyway, then anchor the *supplied* lyric lines to the
  transcript words via fuzzy sequence alignment (`rapidfuzz`, Needleman-Wunsch
  style over normalized tokens). Supplied text is the display truth; Whisper
  provides the timing. `.lrc` line timestamps, when present, constrain the
  alignment search window. **Section-header lines** (`[Verse 1]`, `[Chorus]`
  and the like, common in real lyric files) are detected, excluded from
  `lines[]` and from coverage math (they are not sung), and recorded as
  `supplied_markup` — gold-standard structure evidence consumed by stage 6
  (review 001, R-3). Alternative considered: Montreal Forced Aligner
  (proper forced alignment, better timing on clean speech) — rejected for v1
  as a heavy non-Python toolchain that underperforms on sung vocals without a
  custom acoustic model. See OQ-6.
- **No lyrics file → transcribe.** Whisper on the **isolated vocal stem**
  (dramatically better than the mix), word-level timestamps, language
  auto-detected then pinned into the document.
- **Engine: `faster-whisper` (CTranslate2), model `small` by default,
  int8 on CPU.** Chosen for maturity, word-timestamp quality, deterministic
  greedy decoding (`temperature 0`, `beam_size 1`... beam 5 is also
  deterministic; exact decode params are config), and speed — a 4-minute vocal
  stem transcribes in well under a minute on M-series CPU. **MPS: not
  applicable to CTranslate2**; the Apple-Silicon-native alternative is
  `mlx-whisper` (GPU/ANE, roughly 2–4× faster) — considered and kept as a
  possible backend swap, but it is younger and its word-timestamp plumbing has
  less mileage. Since transcription is not the bottleneck (Demucs is), we take
  the mature choice. See OQ-7.
- **Expected failure modes, and how they surface** (never silently included):
  - *Heavy vocal effects* (distortion, autotune, vocoder) → mistranscription;
    surfaces as low `avg_logprob` → segment flag `low_confidence`.
  - *Dense harmonies / doubled vocals* → merged or hallucinated words; same
    flag path, plus `overlapping_vocals` heuristic flag when the vocal stem
    shows sustained high energy across the segment.
  - *Non-lexical vocals* (scatting, "oh-oh-oh", vocal chops) → Whisper
    hallucinates real words or repeats phrases; flagged via high
    `no_speech_prob`, pathological `compression_ratio` (repetition), or
    dictionary-hit-rate heuristics → `possibly_non_lexical`.
  - *Melisma* (one syllable over many notes) → stretched/odd word timestamps;
    flagged `long_word_duration` when a word exceeds ~2.5 s.
  - *Coverage gaps*: vocal-activity regions (stage 3) with no transcript
    overlap are emitted explicitly as `untranscribed_regions`, so a consumer
    knows "there is singing here we could not read" — the single most
    important honesty feature of this document.
- Every word carries `confidence` (from Whisper word probability, or alignment
  match score on the align path); every line carries `flags: [...]`; the
  document carries coverage stats.

### Stage 5 — Video analysis (video inputs only)

- **Shot detection: PySceneDetect `ContentDetector`** (HSV content delta,
  default threshold 27, min shot length 0.4 s). Mature, deterministic, fast.
  Alternative considered: **TransNetV2** (neural, better on dissolves and
  gradual transitions) — kept as an upgrade path; music videos are heavy on
  hard cuts where ContentDetector is already strong, and TransNetV2 adds a
  TensorFlow/torch weight dependency for marginal v1 gain. See OQ-8.
- **Per shot**:
  - `start_seconds` / `end_seconds` (+ frame indices for reference);
  - **representative frames**: midpoint for shots < 4 s; at 1/3 and 2/3 for
    longer shots (1–2 frames per spec). Stored under `frames/`, each with its
    SHA-256 (which keys the caption cache);
  - **palette**: k-means, k = 5, run in **CIELab** on pixels pooled from the
    representative frame(s) (downscaled to ≤ 256 px, k-means++ with fixed
    seed, `n_init = 1`), output as sRGB hex + proportion, sorted by
    proportion descending. Lab because perceptual distance is what "dominant
    color" means to a human; alternative considered: Pillow median-cut
    (faster, no sklearn dep, noticeably worse swatches on gradients). See OQ-9.
  - **motion energy**: mean dense optical-flow magnitude (OpenCV Farnebäck) on
    the 12 fps / 320 px grayscale pass, normalized by frame diagonal per frame
    pair, then mean and p95 per shot. Dimensionless, comparable across videos.
    Alternative considered: RAFT (torch) — far better flow, absurd overkill
    for a scalar energy score.
- **Captioning behind an interface**:

  ```
  class CaptionBackend(Protocol):
      name: str
      def caption(self, frame_jpeg: bytes, context: ShotContext) -> ShotCaption: ...
  ```

  - `NullBackend` (default): returns empty caption/tags with
    `backend: "null"` — pipeline is fully offline out of the box.
  - `AnthropicBackend`: Claude vision (model pinned in config, key from
    `ANTHROPIC_API_KEY`), one call per representative frame, temperature 0,
    fixed versioned prompt (`prompt_version` recorded in the doc). Returns a
    one-sentence caption + closed-ish tag list.
  - **Determinism policy for captions**: API output is not strictly
    deterministic, so captions are **cached** under
    `.cache/captions/<frame_sha256>.<prompt_version>.<model>.json` at first
    call; re-runs reuse the cache byte-for-byte. Guarantee is therefore
    "deterministic given cache", and the caption block records backend, model,
    and prompt_version so a reviewer can see exactly what produced it. See OQ-10.

### Stage 6 — Structure & alignment

- **Sections** from two evidence sources, fused:
  1. *Audio novelty*: librosa self-similarity / spectral-clustering
     segmentation (McFee & Ellis approach as implemented in
     `librosa.segment`) over beat-synchronous MFCC+chroma → candidate
     boundaries and segment-similarity clusters.
  2. *Lyric repetition*: near-duplicate line clustering (normalized token
     similarity via rapidfuzz); a line cluster that recurs ≥ 2× at similar
     musical positions is chorus evidence; unique blocks between choruses are
     verse evidence.
  Fusion: novelty proposes boundaries (snapped to the nearest downbeat);
  lyric clusters label them. When the supplied lyrics file carried section
  headers, they arrive via `lyrics.json` `supplied_markup` as a third evidence
  source that dominates label assignment (review 001, R-3). Lyric clusters
  otherwise label boundaries (`chorus` for the dominant repeated cluster,
  `verse`, `intro`/`outro` by position, `bridge` for a late unique section,
  `instrumental` when a section has no vocal activity). Every section carries
  `confidence` and `evidence: ["audio_novelty", "lyric_repetition"]` (either
  or both). Instrumental tracks get novelty-only sections with generic labels
  (`section_a`…) — labeling vocabulary in OQ-11. Alternative considered:
  **allin1** (joint beats/downbeats/structure, strong results) — rejected for
  v1: heavy dependency chain (madmom fork, NATTEN) with the same licensing
  concern as madmom.
- **Cut/beat alignment** (video inputs): for each shot cut, distance to the
  nearest beat; emit `cuts_per_bar` (global + per section), `on_beat_cut_ratio`
  (fraction of cuts within ±70 ms of a beat — window in config, OQ-12), and a
  per-cut list `{time, nearest_beat_index, offset_seconds}` so the mapping
  engine can decide for itself what "on beat" means.
- **Unified timeline restated in the doc**: `structure.json` carries no beat
  grid of its own; it references `audio_features.json`'s grid by construction
  and stores only derived indices (e.g. each section's start beat/bar index)
  as a convenience for consumers that don't want to binary-search.

## 5. Library layout & document strategy

```
library/
  3f9a1c7e2b8d4056/              # track_id (16 hex chars of file sha256)
    manifest.json                # mutable envelope: run metadata, doc index, statuses
    source.json                  # probe results, hashes, source & lyrics references
    source_audio.flac            # normalized 44.1 kHz stereo decode
    lyrics_input.lrc             # verbatim copy of supplied lyrics (if any)
    stems/
      vocals.flac  drums.flac  bass.flac  other.flac
    frames/
      shot_0001_f1.jpg  shot_0004_f1.jpg  shot_0004_f2.jpg  ...
    audio_features.json
    lyrics.json
    video.json                   # absent for audio-only tracks (manifest says why)
    structure.json
    .cache/
      captions/<frame_sha>.<prompt_ver>.<model>.json
```

**Separate documents per stage, indexed by a manifest — not one composite
document.** Rationale:

- *Independent recomputation*: stages re-run without rewriting (and
  re-diffing) unrelated results; caption re-runs must not churn
  `audio_features.json`.
- *Independent schema evolution*: each doc has its own `schema_version`;
  bumping the lyrics schema doesn't force a video-schema migration.
- *Consumer ergonomics*: the Swift app decodes only what a screen needs; an
  LLM gets `structure.json` + `lyrics.json` without 200 KB of RMS values in
  its context window.
- *Determinism auditing*: byte-identity is checked per document; the manifest
  is the one file allowed to differ between runs.

The manifest carries: track identity + display title, `has_video`, per-document
entries `{path, schema_version, status (ok | not_applicable | failed |
stale), content_sha256, config_hash (per-stage subset), run {started_at,
duration_seconds, tool_version, device}}`. Per-stage config hashing means
changing a video threshold marks only `video.json` (and its dependents) stale.

## 6. Chosen libraries (with one considered alternative each)

| Purpose | Choice | Why | Considered alternative | Why not |
|---|---|---|---|---|
| Probe/decode/extract | ffmpeg CLI via subprocess | battle-tested, one binary dep, exact seeks | PyAV | heavier build surface on arm64; subprocess is easier to sandbox & reason about |
| Stem separation | Demucs `htdemucs` | state of practice, MIT (code & weights), MPS support | Open-Unmix | lighter but audibly worse separations, esp. "other" |
| DSP features | librosa | mature, pure-Python/numpy, everything we need | Essentia | faster C++, but brittle install on Apple Silicon; overkill |
| Beat tracking | librosa beat_track + 4/4 phase-fit downbeats | zero extra deps, deterministic | madmom DBN downbeat | better downbeats, but model weights CC BY-NC-SA + py3.11 packaging pain (OQ-4) |
| Loudness (LUFS) | pyloudnorm | small, EBU R128 compliant | ffmpeg `ebur128` filter | parsing log output is fragile |
| Transcription | faster-whisper (`small`, int8) | mature, fast on CPU, word timestamps, deterministic decode | mlx-whisper | MPS/ANE-fast but younger; transcription isn't the bottleneck (OQ-7) |
| Lyric alignment | rapidfuzz token alignment against Whisper words | pure-Python, robust to sung mismatch | Montreal Forced Aligner | heavy toolchain, weak on singing without custom models |
| Shot detection | PySceneDetect ContentDetector | deterministic, fast, strong on hard cuts | TransNetV2 | better on dissolves; neural weight dep not worth it for v1 (OQ-8) |
| Optical flow | OpenCV Farnebäck | cheap, adequate for a scalar energy score | RAFT | GPU torch model for a scalar; overkill |
| Palette | scikit-learn KMeans in CIELab | perceptual clustering, seedable | Pillow median-cut | faster but worse swatches |
| Captioning | Anthropic API (vision) behind `CaptionBackend` | per spec; best caption quality | local VLM via mlx-vlm | viable future local backend; NullBackend covers offline today |
| Schemas/validation | pydantic v2 models, JSON Schema exported from them | one source of truth for shape, validation, and docs | hand-written jsonschema | drifts from the code that writes documents |
| CLI | typer | subcommands, completion, minimal boilerplate | argparse | fine, but more ceremony |
| Env/deps | uv (per constraint) | — | — | — |

## 7. Determinism strategy

The contract: **same input file + same config ⇒ byte-identical analysis
documents** (manifest exempt). Enforced in layers:

1. **Canonical serializer** (`canonical.py`): sorted keys OFF (schema order is
   fixed by pydantic model field order — stable and human-curated), UTF-8, no
   ASCII escaping, 2-space indent, trailing newline, and **explicit float
   rounding before serialization**: seconds → 3 dp, dB → 2 dp, Hz → 1 dp,
   ratios/confidences → 3 dp. Rounding is the schema's stated precision, not a
   formatting whim — it also absorbs harmless last-ulp nondeterminism from
   BLAS threading.
2. **Seeded everything**: k-means seed fixed; any torch sampling disabled;
   Whisper decode at temperature 0 with fixed beam; no wall-clock or RNG in
   any document value.
3. **Neural stages**: Demucs and Whisper are the risk. CPU execution is
   deterministic. **MPS kernels are not all deterministic**, and
   `torch.use_deterministic_algorithms(True)` is not fully supported on MPS.
   Policy: byte-identity is *guaranteed in CPU mode*; MPS is best-effort, and
   the rounding layer absorbs most float jitter — but Demucs on MPS may
   produce ±1-sample-level differences that survive into stem files (not
   documents, since features are rounded). Recommended stance in OQ-13:
   guarantee document-level identity on both devices via rounding, guarantee
   stem-file identity only on CPU, and record the device in the manifest.
4. **Captions**: deterministic given cache (§4, stage 5).
5. **CI check** (once implemented): analyze a fixture twice, `diff -r` the
   track dir minus `manifest.json`.

## 8. Error-handling philosophy

- **Fail loudly, per stage, without corrupting the library.** A stage writes
  to a temp file and renames into place only on success; a failed stage leaves
  the previous document (if any) intact and records `status: failed` +
  truncated stderr in the manifest.
- **No fallback masking.** If Demucs fails, we do not quietly analyze the mix
  as if it were stems; dependent stages refuse to run and say why. Degradation
  must be *chosen* by config, never improvised at runtime.
- **Skips are recorded.** Audio-only input ⇒ video stage `not_applicable` in
  the manifest, so absence of `video.json` is distinguishable from "not run
  yet" and from "failed".
- **In-document uncertainty over out-of-band warnings.** Anything a consumer
  should know at read time (low confidence, estimated downbeats,
  untranscribed vocal regions) lives *in the document* as flags/fields;
  logs are for humans debugging, documents are the API.
- **Exit codes**: 0 success, 1 stage failure, 2 bad invocation/missing
  prerequisites. Batch mode (`mrw analyze --all`) continues past per-track
  failures and summarizes.

## 9. Performance & MPS notes (4-minute track, M-series)

| Stage | MPS | CPU-only | Notes |
|---|---|---|---|
| Ingest | — | ~5 s | ffmpeg decode + hash |
| Demucs htdemucs | ~1–2 min | ~5–10 min | the dominant cost; MPS strongly recommended |
| Audio features | — | ~10–20 s | numpy/librosa, CPU-bound regardless |
| Whisper (faster-whisper, small, int8) | n/a (CTranslate2 = CPU) | ~20–60 s | vocal stem only; mlx-whisper would use GPU if we swap (OQ-7) |
| Video (1080p, ~4 min) | — | ~1–2 min | scene detect ≈ faster than realtime; flow pass at 12 fps/320 px |
| Captions (API) | — | network-bound | ~1–2 s per frame, ≤ ~2 × shots calls, cached |
| Structure | — | ~5–10 s | |

**Total: roughly 3–6 minutes with MPS, 8–15 without** — within the "few
minutes" budget either way, but the no-MPS path is dominated by Demucs.

## 10. Risks

- **Model downloads**: htdemucs ~320 MB, Whisper small ~460 MB (large-v3 would
  be ~3 GB — not default). First run needs network *for weights only*; plan a
  `mrw models fetch` command so batch runs never surprise-download, and
  document cache locations (`~/.cache/…`).
- **Licensing**: Demucs code & htdemucs weights MIT; Whisper MIT;
  faster-whisper MIT; librosa ISC; PySceneDetect BSD-3; OpenCV Apache-2;
  scikit-learn BSD-3; pyloudnorm MIT — all clean. The two tempting upgrades
  are the dirty ones: **madmom models are CC BY-NC-SA** (downbeats, OQ-4) and
  TransNetV2 weights need a license check before adoption (OQ-8). Whisper
  transcriptions of copyrighted lyrics stored locally are fine for a private
  analysis library; redistribution would not be — worth one line in the
  eventual README.
- **Determinism on MPS** (§7, OQ-13) — the one constraint we cannot fully
  guarantee at the stem-file level.
- **Runtime creep**: htdemucs_ft (fine-tuned, 4× slower) and Whisper large are
  quality temptations that would blow the time budget; both stay opt-in config.
- **Sung-lyric alignment quality** is the least predictable stage (it's the
  research-grade problem in this pipeline); mitigated by the honesty
  machinery (flags, coverage, untranscribed regions) rather than by promising
  accuracy.
- **PySceneDetect on dissolve-heavy videos** will merge or misplace
  boundaries; mitigated by exposing detector threshold in config and by the
  TransNetV2 upgrade path.

## 11. Milestones & smoke tests

Each milestone lands with its smoke test scripted (fixtures are generated,
tiny, and committed or built on the fly — no copyrighted media in the repo).

- **M0 — Skeleton & canonical I/O.** uv project, typer CLI stub, config
  hashing, canonical serializer, library/manifest module, pydantic models for
  `source.json` + manifest, JSON Schema export wired up.
  *Smoke*: `mrw ingest fixture.wav` twice → track dir exists; second run
  reports no-op; `source.json` byte-identical across runs; schema validates.
- **M1 — Ingest for real.** ffprobe integration, normalization to FLAC,
  audio-stream hash, lyrics-file registration, video probe fields.
  *Smoke*: ingest (a) a generated tone WAV, (b) a generated test-pattern MP4
  with a sine track → correct `has_video`, correct durations (±10 ms), FLAC
  decodes to expected sample count.
- **M2 — Stems.** Demucs wrapper, MPS/CPU device selection, retention config.
  *Smoke*: 30 s fixture (tone + click track mixed) → 4 stem files exist,
  each duration matches source ±1 frame; `retain: false` leaves no `stems/`.
- **M3 — Audio features.** Beats, onsets, envelopes, centroid, LUFS, vocal
  activity; `audio_features.json` full schema.
  *Smoke*: generated 120 BPM click track → detected BPM ∈ [119, 121]; beat
  times within 30 ms of ground truth; RMS series length = ⌈duration × 100⌉;
  document byte-identical across two runs.
- **M4 — Lyrics.** faster-whisper transcription path, alignment path,
  flags/coverage machinery.
  *Smoke*: fixture with synthesized speech (say, macOS `say` rendered over a
  quiet bed) + matching text file → alignment path produces monotonically
  increasing word times covering ≥ 90 % of words; transcription path on the
  same audio yields ≥ 80 % token overlap; a music-only fixture yields empty
  words + `untranscribed_regions` covering the vocal-activity span.
- **M5 — Video.** Shot detection, representative frames, palettes, motion,
  NullBackend captions; `video.json`.
  *Smoke*: generated video with 10 hard cuts between solid-color scenes →
  exactly 10 boundaries within 1 frame; each shot's top palette swatch within
  ΔE < 5 of the known color; static scene motion ≈ 0, moving-gradient scene
  motion > 0.
- **M6 — Structure & alignment.** Novelty segmentation, lyric-repetition
  fusion, cut/beat metrics; `structure.json`.
  *Smoke*: fixture song built as A-B-A-B-C (distinct synth textures, repeated
  "chorus" lyric lines) → ≥ 4 boundaries within 1 bar of construction points;
  the repeated section pair gets the same label; fixture video cut exactly on
  beats → `on_beat_cut_ratio ≥ 0.9`.
- **M7 — Anthropic caption backend & hardening.** API backend, cache,
  `mrw models fetch`, batch mode, end-to-end determinism CI check.
  *Smoke*: with a mock HTTP layer, two runs produce byte-identical
  `video.json` (second run: zero API calls); full-pipeline double-run diff on
  the A/V fixture is clean outside `manifest.json`.

## 12. What review should focus on

The load-bearing judgment calls are all in `OPEN_QUESTIONS.md`; the ones with
the widest blast radius are the timeline/hop convention (OQ-* / §4.3), the
separate-documents decision (§5), downbeat licensing (OQ-4), and the MPS
determinism stance (OQ-13). Schemas in `schemas/` are drafted against a
hypothetical 4-minute track and are the primary review surface for the two
downstream consumers.
