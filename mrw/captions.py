"""Pluggable shot-caption backends (PLAN §4 stage 5, OQ-10).

`null` (default): offline no-op — caption absent from the document, which
consumers must distinguish from an empty-but-real caption (schema).

`anthropic`: Claude vision, one call per shot (its 1–2 representative
frames attached). Determinism is OQ-10's "deterministic given cache": the
response is cached inside the library keyed on (combined frame sha256s,
prompt_version, model); re-runs replay byte-for-byte with zero API calls.
Key from ANTHROPIC_API_KEY. CI never calls the API — the review workflow
runs the null backend, and a mocked-cache test proves the replay path.

Cost: the video stage pre-flight estimates uncached calls through
mrw/costs.py and gates on caption_budget_usd_per_run before any spend.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Protocol

from . import canonical, config
from .models import ShotCaption

PROMPT_TEMPLATE = (
    "You are captioning shots of a music video for a generative-visuals "
    "pipeline. Look at the attached frame(s) — they are representative "
    "frames of ONE shot. Reply with STRICT JSON, no prose: "
    '{"text": <one literal sentence describing what is on screen>, '
    '"tags": [<3-8 lowercase snake_case tags: subjects, setting, lighting, '
    "mood, camera>]}"
)
# Rough, conservative token count for PROMPT_TEMPLATE + JSON scaffolding;
# used only for cost estimation (mrw/costs.py), never billing.
PROMPT_TOKENS_ESTIMATE = 160


def cache_key(frame_sha256s: list[str], prompt_version: str, model: str) -> str:
    combined = hashlib.sha256("".join(frame_sha256s).encode()).hexdigest()
    return f"{combined}.{prompt_version}.{model}"


class CaptionBackend(Protocol):
    name: str

    def caption(
        self, frame_jpegs: list[bytes], frame_sha256s: list[str]
    ) -> Optional[ShotCaption]: ...


class NullBackend:
    """Offline default: no caption (absent field, not empty text)."""

    name = "null"

    def caption(self, frame_jpegs, frame_sha256s):
        return None


class CaptionCache:
    """Library-resident replay cache (OQ-10)."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional[ShotCaption]:
        path = self.path(key)
        if not path.is_file():
            return None
        return ShotCaption.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, key: str, caption: ShotCaption) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        canonical.write(self.path(key), caption.model_dump(mode="json"))


class AnthropicBackend:
    """Claude vision captions, cached per OQ-10. One call per shot."""

    name = "anthropic"

    def __init__(self, vcfg: config.VideoConfig, cache: CaptionCache):
        self.model = vcfg.caption_model
        self.prompt_version = vcfg.caption_prompt_version
        self.max_output_tokens = vcfg.caption_max_output_tokens
        self.cache = cache
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "caption_backend = anthropic requires ANTHROPIC_API_KEY in the "
                "environment"
            )

    def caption(self, frame_jpegs, frame_sha256s):
        key = cache_key(frame_sha256s, self.prompt_version, self.model)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        import base64

        import anthropic

        client = anthropic.Anthropic()
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(jpeg).decode(),
                },
            }
            for jpeg in frame_jpegs
        ]
        content.append({"type": "text", "text": PROMPT_TEMPLATE})
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_output_tokens,
            messages=[{"role": "user", "content": content}],
        )
        self.calls += 1
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        text_block = next(b.text for b in response.content if b.type == "text")
        parsed = json.loads(text_block)
        caption = ShotCaption(
            text=str(parsed["text"]),
            tags=[str(tag) for tag in parsed.get("tags", [])],
        )
        self.cache.put(key, caption)
        return caption


def make_backend(
    vcfg: config.VideoConfig, cache_dir: Path
) -> tuple[CaptionBackend, CaptionCache]:
    cache = CaptionCache(cache_dir)
    if vcfg.caption_backend == "null":
        return NullBackend(), cache
    if vcfg.caption_backend == "anthropic":
        return AnthropicBackend(vcfg, cache), cache
    raise ValueError(
        f"video.caption_backend must be null|anthropic, got {vcfg.caption_backend!r}"
    )
