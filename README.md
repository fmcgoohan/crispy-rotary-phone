# Music Reinterpretation Workbench (mrw)

`mrw` is the Mac-side analysis workbench for a music-reinterpretation
project: it takes a local song (audio file or music video) plus optional
lyrics, and produces deterministic, versioned analysis documents — stems,
beat grid, loudness envelopes, timed lyrics, shot/palette/motion analysis,
and song structure — in a local library. Downstream consumers (a mapping
engine and a generative-visuals renderer on iPad) read those documents; this
repo's whole job is to make them correct, stable, and reusable.

Design docs: [`PLAN.md`](PLAN.md) (architecture, milestones),
[`schemas/`](schemas/) (document contracts + worked examples),
[`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md), [`docs/reviews/`](docs/reviews/)
(design-review record).

## Setup

Requires macOS (Apple Silicon), Python 3.11+, [`uv`](https://docs.astral.sh/uv/),
and `ffmpeg`/`ffprobe` on PATH (`brew install ffmpeg uv`).

```sh
uv sync            # install dependencies into .venv
uv run pytest -q   # run the smoke tests
```

## Usage

```sh
uv run mrw ingest path/to/song.wav --lyrics path/to/song.lrc --title "Song Title"
uv run mrw stems <track_id>        # Demucs separation → stems/{vocals,drums,bass,other}.flac
uv run mrw status                  # list tracks and per-stage statuses
uv run mrw export-schemas          # dump model-generated schemas for diffing
```

## Model weights

Neural stages download model weights on first use (Demucs `htdemucs`,
~80 MB). Run `uv run mrw models fetch` once up front so batch runs never
surprise-download. Weights are cached in torch's hub cache —
`~/.cache/torch/hub/checkpoints/` by default (override with `TORCH_HOME`).
Stem separation runs on Apple-Silicon GPU (`mps`) when available;
stem-file byte determinism is guaranteed on `cpu` only (PLAN §7 / OQ-13) —
set `[stems] device = "cpu"` in `mrw.toml` when you need it.

Tracks land in `./library/<track_id>/` (override with `--library` or
`MRW_LIBRARY`). Configuration lives in `mrw.toml` (see `mrw/config.py` for
defaults). Analysis stages beyond ingest arrive milestone by milestone —
see PLAN.md §11.

## Do not publish library contents

Track directories contain copyrighted material derived from the source
media: lyric text and transcriptions (`lyrics_input.*`, `lyrics.json`),
video stills (`frames/`), and derivative audio (`stems/`,
`source_audio.flac`). They are for private, local analysis only — do not
publish, share, or sync a library directory publicly. The `library/` default
location is gitignored for this reason.
