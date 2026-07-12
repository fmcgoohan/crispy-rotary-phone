# Analysis document schemas (draft, Phase 0)

JSON Schema (draft 2020-12) for every analysis document, plus a realistic
worked example for each under `examples/`, drafted for a hypothetical
4-minute track ("Neon Reverie", 242.5 s, ~120 BPM, 1080p24 music video,
`track_id: 3f9a1c7e2b8d4056`).

## Documents

| Schema | Example | Written by | Purpose |
|---|---|---|---|
| `manifest.schema.json` | `examples/manifest.json` | every stage | mutable envelope: run metadata, document index, statuses. **The only file allowed to differ between identical runs.** |
| `source.schema.json` | `examples/source.json` | ingest | probe results, hashes, source & lyrics references |
| `audio_features.schema.json` | `examples/audio_features.json` | features | tempo/beats, onsets, envelopes, centroid, loudness, vocal activity |
| `lyrics.schema.json` | `examples/lyrics.json` | lyrics | timed words/lines with confidence and honesty flags |
| `video.schema.json` | `examples/video.json` | video | shots, frames, palettes, motion, captions |
| `structure.schema.json` | `examples/structure.json` | structure | sections + cut/beat alignment |

`common.schema.json` holds shared `$defs` (`timeseries`, `interval`,
`event_list`) referenced by the others.

## Conventions (binding on all documents)

- **Keys**: `snake_case`. Swift decodes with
  `keyDecodingStrategy = .convertFromSnakeCase`; no dynamic keys anywhere —
  every map a decoder sees has a fixed, schema-declared shape.
- **Time**: seconds (JSON number) from t = 0 = first sample of the normalized
  audio decode. One timeline for everything, including video. Beat/bar indices
  are derivable from `audio_features.beats`.
- **Precision** (also the determinism-rounding contract): seconds 3 dp,
  dB 2 dp, Hz 1 dp, BPM 1 dp, ratios/confidences 3 dp.
- **Enums are lowercase strings**; unknown future values must not crash a
  consumer (Swift: decode as raw `String` or provide an `unknown` case).
- **Optionality**: absent means absent; `null` is never written.
- **Every document** carries `schema_version` (semver string; bump minor for
  additive change, major for breaking) and `track_id`.
- **No run metadata** (timestamps, hostnames, durations, tool versions) in any
  document except `manifest.json`.
- **`flags`** are arrays of lowercase snake_case codes from a vocabulary
  enumerated in each schema; consumers must tolerate unknown codes.
- **Parameter embedding**: wherever a consumer might need to reinterpret or
  compare a field across tracks (detector thresholds, hysteresis levels,
  on-beat windows, normalization references), the producing stage embeds a
  small params block next to the data; everywhere else the manifest's
  per-stage `config_hash` suffices.
- Keys prefixed with `_` (used in the examples for truncation notes) are
  documentation-only and never produced by the tool; schemas don't declare
  them and consumers ignore unknown keys.

## Dense time-series convention (`common.schema.json#/$defs/timeseries`)

The one place compactness beats self-description. A uniform-hop series is:

```json
{
  "unit": "dbfs",
  "start_seconds": 0.0,
  "hop_seconds": 0.01,
  "values": [-60.0, -58.21, -55.7]
}
```

- `values[i]` is the measurement at `start_seconds + i * hop_seconds`.
- Sampling at arbitrary `t`: `x = (t - start_seconds) / hop_seconds`;
  linearly interpolate `values[floor(x)]` … `values[ceil(x)]`; clamp at the
  ends. O(1).
- All series in one document share the same hop (10 ms) unless a field says
  otherwise; the hop is still stated per-series so each series is
  self-contained.
- Values are plain JSON numbers (rounded per the precision contract);
  a 4-minute track ⇒ 24,250 points ⇒ ~150–200 KB per series, gzip-friendly.
  Swift decodes `values` as `[Double]`.

## Interval & event conventions

- **Interval lists** (`$defs/interval`): objects with `start_seconds` /
  `end_seconds`, sorted by start, non-overlapping within one list.
- **Event lists**: sorted ascending arrays of times (plain `[Double]`), or
  arrays of objects each carrying a `time_seconds`, sorted. Query by binary
  search.
