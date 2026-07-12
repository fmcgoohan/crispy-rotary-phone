# Review 003 — M0: skeleton, canonical I/O, WAV ingest

**Scope:** commits `3c702c4..2644331` on main (Phase 0 baseline + reviews
applied + M0 implementation), read in full and executed independently.
**Verdict:** **M0 approved. Proceed to M1.** Two documentation follow-ups
below; no code changes requested.

## Independent verification

Run in a separate environment (Linux, Python 3.12, system ffmpeg) — i.e. a
different OS and interpreter from the development Mac:

- All 4 smoke tests pass (`4 passed in 2.02s`), including double-ingest
  byte-identity across separate library roots and schema validation against
  the hand-drafted contract. Cross-platform byte-identity of `source.json`
  is a stronger determinism result than a same-machine double-run.
- Extra probe: identical file bytes ingested from two different paths produce
  the same `track_id` and a `source.json` differing only in
  `file.original_path` (see F-1).
- R-9 (silent-channel conventions), N-1 (bar-index clamp), N-2 (real pinned
  model id), and R-7 (`ingest.copy_video` in config) all verified in the tree.

Noted with approval: atomic temp-and-rename in `canonical.write`; the
`attached_pic` cover-art exclusion in stream selection (a real-world catch —
audio files with embedded art would otherwise register as videos); duration
derived from the decoded PCM byte count rather than container metadata; and
the `export-schemas` command for diffing generated model shapes against the
reviewed contract.

## Findings

### F-1 — Scope the determinism contract wording (PLAN §7) — doc only

`source.json` records `file.original_path`, so byte-identity holds for
re-runs of the same file *at the same path*; identity (`track_id`) is over
bytes alone. Add one sentence to §7 making this explicit, e.g.: "Identity is
a function of file bytes; `source.json` additionally records the ingest-time
path, so its byte-identity guarantee applies to re-runs of the same file at
the same location." Prevents a future CI check or reviewer from reading the
current wording as a stricter promise than intended.

### F-2 — Root README is still the stub; R-8 remains unlanded — doc only

`README.md` contains the repo title twice and nothing else. It needs: what
mrw is (two sentences), install/run basics, and the review-001 R-8 note —
track directories contain copyrighted material (lyric text and
transcriptions, `frames/` stills, `stems/` derivative audio) and must not be
published or synced publicly. This closes the last open item from review 001.

---

**Next:** M1 (full ingest: video probe fields exercised on a real container,
lyrics registration, generated-fixture MP4 smoke). Open it as a PR — the
repo is public and the reviewer can fetch PR branches directly.
