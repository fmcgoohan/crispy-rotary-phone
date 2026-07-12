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


models_app = typer.Typer(help="Manage model weights.", no_args_is_help=True)
app.add_typer(models_app, name="models")


@models_app.command("fetch")
def models_fetch(config_path: Optional[Path] = _CONFIG_OPT) -> None:
    """Download model weights up front so batch runs never surprise-download."""
    from .stems import fetch_model

    cfg = _load_config(config_path)
    typer.echo(f"fetching '{cfg.stems.model}' (first run downloads ~80-320 MB)...")
    cache_dir = fetch_model(cfg.stems.model)
    typer.echo(f"model '{cfg.stems.model}' ready; cache: {cache_dir}")


@app.command()
def status(library: Path = _LIBRARY_OPT) -> None:
    """List tracks and per-document statuses."""
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
            f"source={docs.source.status}",
            f"features={docs.audio_features.status}",
            f"lyrics={docs.lyrics.status}",
            f"video={docs.video.status}",
            f"structure={docs.structure.status}",
        ]
        if manifest.stems is not None:
            parts.insert(1, f"stems={manifest.stems.status}")
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
