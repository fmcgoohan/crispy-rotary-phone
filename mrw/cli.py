"""mrw CLI. Exit codes: 0 success, 1 stage failure, 2 bad invocation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import __version__, config
from .library import Library

app = typer.Typer(
    name="mrw",
    help="Music Reinterpretation Workbench — analysis pipeline.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

_LIBRARY_OPT = typer.Option(
    Path("library"), "--library", envvar="MRW_LIBRARY", help="Library root directory."
)
_CONFIG_OPT = typer.Option(
    None, "--config", help="Path to mrw.toml (default: ./mrw.toml if present)."
)


def _load_config(config_path: Optional[Path]) -> config.Config:
    try:
        return config.load(config_path)
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)


@app.command()
def version() -> None:
    """Print the mrw version."""
    typer.echo(f"mrw {__version__}")


@app.command()
def ingest(
    media: Path = typer.Argument(..., help="Local media file (audio or video)."),
    lyrics: Optional[Path] = typer.Option(
        None, "--lyrics", help="Optional lyrics file (.lrc or .txt)."
    ),
    title: Optional[str] = typer.Option(None, "--title", help="Display title override."),
    library: Path = _LIBRARY_OPT,
    config_path: Optional[Path] = _CONFIG_OPT,
) -> None:
    """Register a track: probe, hash, normalize audio, write source.json."""
    from .ingest import IngestError, ingest as run_ingest

    if not media.is_file():
        typer.echo(f"error: no such file: {media}", err=True)
        raise typer.Exit(2)
    if lyrics is not None and not lyrics.is_file():
        typer.echo(f"error: no such lyrics file: {lyrics}", err=True)
        raise typer.Exit(2)

    cfg = _load_config(config_path)
    try:
        result = run_ingest(
            media, Library(library), cfg, lyrics_path=lyrics, title=title
        )
    except IngestError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)

    if result.already_ingested:
        typer.echo(f"already ingested: {result.track_id} ({result.title}) — no-op")
    else:
        kind = "video" if result.has_video else "audio"
        typer.echo(f"ingested {result.track_id} ({result.title}) [{kind}]")


@app.command()
def stems(
    track: str = typer.Argument(
        ..., help="Track id, or a media path (re-resolved through the library)."
    ),
    library: Path = _LIBRARY_OPT,
    config_path: Optional[Path] = _CONFIG_OPT,
) -> None:
    """Separate the normalized audio into vocals/drums/bass/other stems."""
    from .stems import PrerequisiteError, StemsError, run_stems

    cfg = _load_config(config_path)
    try:
        result = run_stems(track, Library(library), cfg)
    except PrerequisiteError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    except StemsError as e:
        typer.echo(f"error: stems failed: {e}", err=True)
        raise typer.Exit(1)

    if result.already_done:
        typer.echo(f"stems up to date: {result.track_id} — no-op")
    else:
        if result.mps_fallback_error:
            typer.echo(
                "warning: mps separation failed, fell back to cpu: "
                f"{result.mps_fallback_error}",
                err=True,
            )
        kept = "retained" if result.retained else "not retained (per config)"
        typer.echo(f"stems {result.track_id} [device={result.device}] {kept}")


@app.command()
def features(
    track: str = typer.Argument(
        ..., help="Track id, or a media path (re-resolved through the library)."
    ),
    library: Path = _LIBRARY_OPT,
    config_path: Optional[Path] = _CONFIG_OPT,
) -> None:
    """Extract audio features (beats, envelopes, onsets, LUFS, vocal activity)."""
    from .features import FeaturesError, run_features
    from .library import PrerequisiteError

    cfg = _load_config(config_path)
    try:
        result = run_features(track, Library(library), cfg)
    except PrerequisiteError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    except FeaturesError as e:
        typer.echo(f"error: features failed: {e}", err=True)
        raise typer.Exit(1)

    if result.already_done:
        typer.echo(f"features up to date: {result.track_id} — no-op")
    else:
        typer.echo(
            f"features {result.track_id}: bpm={result.bpm_global} "
            f"vocal_regions={result.n_vocal_regions}"
        )


@app.command()
def lyrics(
    track: str = typer.Argument(
        ..., help="Track id, or a media path (re-resolved through the library)."
    ),
    library: Path = _LIBRARY_OPT,
    config_path: Optional[Path] = _CONFIG_OPT,
) -> None:
    """Align supplied lyrics or transcribe the vocal stem (mode per ingest)."""
    from .library import PrerequisiteError
    from .lyrics import LyricsError, run_lyrics

    cfg = _load_config(config_path)
    try:
        result = run_lyrics(track, Library(library), cfg)
    except PrerequisiteError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    except LyricsError as e:
        typer.echo(f"error: lyrics failed: {e}", err=True)
        raise typer.Exit(1)

    if result.already_done:
        typer.echo(f"lyrics up to date: {result.track_id} — no-op")
    else:
        typer.echo(
            f"lyrics {result.track_id} [{result.mode}]: {result.n_lines} lines, "
            f"{result.n_untranscribed} untranscribed regions"
        )


@app.command()
def video(
    track: str = typer.Argument(
        ..., help="Track id, or a media path (re-resolved through the library)."
    ),
    estimate: bool = typer.Option(
        False, "--estimate",
        help="Print the caption cost estimate and exit without spending.",
    ),
    library: Path = _LIBRARY_OPT,
    config_path: Optional[Path] = _CONFIG_OPT,
) -> None:
    """Analyze the video: shots, frames, palettes, motion, captions."""
    from .library import PrerequisiteError
    from .video import VideoError, run_video

    cfg = _load_config(config_path)
    try:
        result = run_video(track, Library(library), cfg, estimate_only=estimate)
    except PrerequisiteError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    except VideoError as e:
        typer.echo(f"error: video failed: {e}", err=True)
        raise typer.Exit(1)

    if result.estimated_only:
        if result.estimate is not None:
            typer.echo(
                f"estimate for {result.track_id} ({result.n_shots} shots, "
                f"uncached only): {result.estimate.describe()}"
            )
        else:
            typer.echo(
                f"estimate for {result.track_id} ({result.n_shots} shots): "
                "$0 — caption backend is 'null'"
            )
        return
    if result.already_done:
        typer.echo(f"video up to date: {result.track_id} — no-op")
        return
    line = f"video {result.track_id}: {result.n_shots} shots [captions={result.backend}]"
    if result.usage is not None:
        line += (
            f" api: {result.usage.calls} calls, ${result.usage.usd:.4f}"
            f" (estimated ${result.usage.estimated_usd:.4f})"
        )
    typer.echo(line)


@app.command()
def costs(library: Path = _LIBRARY_OPT) -> None:
    """Sum recorded API spend across the library (estimate vs actual)."""
    import json as _json

    from . import costs as costs_mod

    lib = Library(library)
    manifests = []
    for track_id in lib.track_ids():
        manifests.append(
            _json.loads(lib.manifest_path(track_id).read_text(encoding="utf-8"))
        )
    totals = costs_mod.aggregate_ledger(manifests)
    typer.echo(
        f"{totals['runs']} API-spending run(s): {totals['calls']} calls, "
        f"{totals['input_tokens']} in / {totals['output_tokens']} out tokens, "
        f"${totals['usd']:.4f} actual vs ${totals['estimated_usd']:.4f} estimated"
    )


models_app = typer.Typer(help="Manage model weights.", no_args_is_help=True)
app.add_typer(models_app, name="models")


@models_app.command("fetch")
def models_fetch(config_path: Optional[Path] = _CONFIG_OPT) -> None:
    """Download model weights up front so batch runs never surprise-download."""
    from .lyrics import fetch_whisper_model
    from .stems import fetch_model

    cfg = _load_config(config_path)
    typer.echo(f"fetching '{cfg.stems.model}' (first run downloads ~80-320 MB)...")
    cache_dir = fetch_model(cfg.stems.model)
    typer.echo(f"model '{cfg.stems.model}' ready; cache: {cache_dir}")
    typer.echo(f"fetching whisper '{cfg.lyrics.model}' (first run ~75-500 MB)...")
    whisper_dir = fetch_whisper_model(cfg.lyrics.model)
    typer.echo(f"whisper '{cfg.lyrics.model}' ready; cache: {whisper_dir}")


@app.command()
def status(
    library: Path = _LIBRARY_OPT,
    config_path: Optional[Path] = _CONFIG_OPT,
) -> None:
    """List tracks and per-document statuses.

    A document recorded `ok` under a different config than the current one
    is displayed as `stale` — derived here by comparing the stored
    config_hash against the current stage hash; the manifest itself is
    never rewritten by this command (review 006, note 2).
    """
    cfg = _load_config(config_path)

    def _display(name: str, status: str, stored_hash: Optional[str]) -> str:
        section = config.STAGE_SECTION.get(name)
        if status == "ok" and stored_hash and section:
            if stored_hash != config.stage_hash(getattr(cfg, section)):
                return "stale"
        return status

    lib = Library(library)
    track_ids = lib.track_ids()
    if not track_ids:
        typer.echo(f"no tracks in {library}")
        return
    for track_id in track_ids:
        manifest = lib.read_manifest(track_id)
        if manifest is None:
            continue
        docs = manifest.documents
        parts = [
            "source="
            + _display("source", docs.source.status, docs.source.config_hash),
            "features="
            + _display(
                "audio_features",
                docs.audio_features.status,
                docs.audio_features.config_hash,
            ),
            f"lyrics={docs.lyrics.status}",
            f"video={docs.video.status}",
            f"structure={docs.structure.status}",
        ]
        if manifest.stems is not None:
            parts.insert(
                1,
                "stems="
                + _display("stems", manifest.stems.status, manifest.stems.config_hash),
            )
        typer.echo(f"{track_id}  {manifest.title}  " + " ".join(parts))


@app.command("export-schemas")
def export_schemas(
    out: Path = typer.Option(
        Path("schemas/generated"), "--out", help="Output directory."
    ),
) -> None:
    """Export JSON Schemas generated from the pydantic models.

    The hand-drafted schemas in schemas/ are the reviewed contract; this
    export exists to diff the implementation's shape against them.
    """
    from . import canonical
    from .models import Manifest, SourceDocument

    out.mkdir(parents=True, exist_ok=True)
    for name, model in [("source", SourceDocument), ("manifest", Manifest)]:
        canonical.write(out / f"{name}.schema.json", model.model_json_schema())
        typer.echo(f"wrote {out / f'{name}.schema.json'}")


if __name__ == "__main__":
    app()
