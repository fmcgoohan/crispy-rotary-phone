"""Stage 2 — stem separation (Demucs htdemucs) per PLAN §4 stage 2.

Stems are files tracked by the manifest `stems` block, not documents.
Determinism (OQ-13): stem-file byte identity is guaranteed on CPU only —
torch threads are pinned to 1 there, because stem files have no rounding
layer to absorb thread-order float jitter. The manifest records the device
actually used so the guarantee is auditable. MPS is best-effort.

Demucs note: pypi demucs 4.0.1 has no `demucs.api` module (that shipped only
in unreleased 4.1 alphas); we use the stable lower-level surface
(`demucs.pretrained.get_model` + `demucs.apply.apply_model`), replicating
`demucs.separate`'s mean/std normalization around the model call.
"""

from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import TOOL_VERSION, config, hashing
from .library import Library
from .models import RunMetadata, StemsState

STEM_NAMES = ("vocals", "drums", "bass", "other")


class StemsError(RuntimeError):
    """Stage failure — recorded in the manifest as status=failed; exit 1."""


class PrerequisiteError(RuntimeError):
    """Bad invocation / missing prerequisites — nothing recorded; exit 2."""


@dataclass
class StemsResult:
    track_id: str
    device: str
    retained: bool
    already_done: bool
    # Set when MPS failed and the stage fell back to CPU (review 005 F-1,
    # H1: the fallback is surfaced, not swallowed).
    mps_fallback_error: str | None = None


def resolve_track_id(track: str, library: Library) -> str:
    """Accept a track_id or a media path (re-resolved by content hash)."""
    if re.fullmatch(r"[0-9a-f]{16}", track):
        if library.manifest_path(track).is_file():
            return track
        raise PrerequisiteError(f"no such track in library: {track}")
    path = Path(track)
    if path.is_file():
        track_id = hashing.track_id_from_sha(hashing.sha256_file(path))
        if library.manifest_path(track_id).is_file():
            return track_id
        raise PrerequisiteError(
            f"{path.name} (track {track_id}) is not ingested — run `mrw ingest` first"
        )
    raise PrerequisiteError(f"not a track_id or existing file: {track}")


def _resolve_device(setting: str) -> str:
    import torch

    if setting == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if setting not in ("cpu", "mps"):
        raise PrerequisiteError(f"stems.device must be auto|cpu|mps, got {setting!r}")
    return setting


def fetch_model(model_name: str) -> Path:
    """Download the model weights if absent; returns the cache directory."""
    import torch.hub
    from demucs.pretrained import get_model

    get_model(model_name)
    return Path(torch.hub.get_dir()) / "checkpoints"


def _separate(
    flac_path: Path, out_dir: Path, model_name: str, device: str, cpu_threads: int = 1
) -> None:
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    torch.manual_seed(0)
    if device == "cpu":
        # OQ-13: stem files carry no rounding layer, so cross-run byte
        # identity on CPU requires eliminating thread-order float jitter —
        # the guarantee holds at cpu_threads = 1 (the default); higher
        # values trade it for speed (review 005, F-2). Process-global and
        # deliberately not restored — fine for the one-shot CLI; a future
        # combined-stage run sharing this process must set its own
        # threading policy after separation (D5).
        torch.set_num_threads(max(1, cpu_threads))

    model = get_model(model_name)
    model.eval()

    data, sample_rate = sf.read(flac_path, dtype="float32", always_2d=True)
    if sample_rate != model.samplerate:
        raise StemsError(
            f"{flac_path.name} is {sample_rate} Hz; model expects {model.samplerate}"
        )
    wav = torch.from_numpy(data.T.copy())  # (channels, samples)
    if wav.shape[0] != model.audio_channels:
        raise StemsError(
            f"{flac_path.name} has {wav.shape[0]} channels; "
            f"model expects {model.audio_channels}"
        )

    # demucs.separate's normalization, replicated for parity with the CLI —
    # except that a silent or DC-only input has ref.std() == 0 and would
    # divide to NaN in every stem (M2 review, H2). Convention: (near-)silent
    # input separates unnormalized to (near-)silent stems.
    ref = wav.mean(0)
    ref_mean, ref_std = ref.mean(), ref.std()
    normalize = ref_std.item() > 0.0
    if normalize:
        wav = (wav - ref_mean) / ref_std
    with torch.no_grad():
        sources = apply_model(
            model,
            wav[None],
            device=device,
            shifts=0,  # shift augmentation is randomized; keep it off
            split=True,
            overlap=0.25,
            progress=False,
            num_workers=0,
        )[0]
    if normalize:
        sources = sources * ref_std + ref_mean
    if not torch.isfinite(sources).all():
        raise StemsError("separation produced non-finite samples (NaN/Inf)")

    for name, tensor in zip(model.sources, sources):
        pcm = (
            tensor.clamp(-1.0, 1.0)
            .mul(32767.0)
            .round()
            .to(torch.int16)
            .cpu()
            .numpy()
        )
        sf.write(out_dir / f"{name}.flac", pcm.T, sample_rate, subtype="PCM_16")


def run_stems(track: str, library: Library, cfg: config.Config) -> StemsResult:
    track_id = resolve_track_id(track, library)
    manifest = library.read_manifest(track_id)
    if manifest is None or manifest.documents.source.status != "ok":
        raise PrerequisiteError(f"track {track_id} has no successful ingest")

    stems_hash = config.stage_hash(cfg.stems)
    track_dir = library.track_dir(track_id)
    stems_dir = track_dir / "stems"

    prior = manifest.stems
    if (
        prior is not None
        and prior.status == "ok"
        and prior.config_hash == stems_hash
        and (
            not prior.retained
            or all((stems_dir / f"{n}.flac").is_file() for n in STEM_NAMES)
        )
    ):
        return StemsResult(
            track_id=track_id,
            device=prior.run.device if prior.run and prior.run.device else "cpu",
            retained=prior.retained,
            already_done=True,
        )

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.monotonic()
    tmp_dir = track_dir / ".stems.tmp"
    mps_fallback_error: str | None = None

    def _record(state: StemsState) -> None:
        manifest.stems = state
        library.write_manifest(track_id, manifest)

    try:
        device = _resolve_device(cfg.stems.device)
        flac_path = track_dir / "source_audio.flac"
        if not flac_path.is_file():
            raise StemsError(f"missing {flac_path.name} — library is damaged?")

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()
        try:
            _separate(flac_path, tmp_dir, cfg.stems.model, device, cfg.stems.cpu_threads)
        except Exception as exc:
            if device != "mps":
                raise
            # Review 005, field finding F-1: htdemucs on MPS can fail at
            # runtime on real tracks. Retry on CPU rather than failing the
            # stage; the manifest records the device that actually ran and
            # the MPS error is surfaced to the caller (H1), not swallowed.
            mps_fallback_error = str(exc)[:200]
            shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir()
            device = "cpu"
            _separate(flac_path, tmp_dir, cfg.stems.model, device, cfg.stems.cpu_threads)

        run = RunMetadata(
            started_at=started_at,
            duration_seconds=round(time.monotonic() - t0, 2),
            tool_version=TOOL_VERSION,
            device=device,
        )
        if cfg.stems.retain:
            if stems_dir.exists():
                shutil.rmtree(stems_dir)
            tmp_dir.rename(stems_dir)
        else:
            # Dependent stages consume the stems here in a combined run
            # (none exist yet — M3+); then the files go away and the
            # manifest tells any later re-run to regenerate.
            shutil.rmtree(tmp_dir)
            if stems_dir.exists():
                # retain flipped true→false: drop the previously-retained
                # stems too, so `retained: false` reflects disk reality (H2).
                shutil.rmtree(stems_dir)

        _record(
            StemsState(
                status="ok",
                retained=cfg.stems.retain,
                config_hash=stems_hash,
                run=run,
                warning=(
                    f"mps separation failed, fell back to cpu: {mps_fallback_error}"
                    if mps_fallback_error
                    else None
                ),
            )
        )
        return StemsResult(
            track_id=track_id,
            device=device,
            retained=cfg.stems.retain,
            already_done=False,
            mps_fallback_error=mps_fallback_error,
        )
    except PrerequisiteError:
        raise
    except Exception as exc:  # fail loudly, never leave partial stems/
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if mps_fallback_error:
            # Don't discard the original MPS failure when the CPU retry
            # also fails (PR #5 review round 2) — and cap each part
            # independently so the 500-char limit can never swallow the
            # CPU-side detail (round 3): 200 + 250 + framing < 500.
            message = (
                f"cpu retry failed after mps failure ({mps_fallback_error}): "
                f"{str(exc)[:250]}"
            )
        else:
            message = str(exc)
        _record(
            StemsState(
                status="failed",
                retained=cfg.stems.retain,
                config_hash=stems_hash,
                error=message[:500],
            )
        )
        raise StemsError(message[:500]) from exc
