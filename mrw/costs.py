"""Cost prediction and budget guard for API-spending stages (M5 addendum).

Reusable: captions are the first client, later API features (e.g. a future
non-null lyrics backend) estimate through the same module. No network, no
SDK imports — pure arithmetic over a shipped price table.

Policy: the estimator ERRORS on unknown model ids instead of guessing, and
estimates are deliberately conservative (max output tokens assumed spent on
every call).
"""

from __future__ import annotations

from dataclasses import dataclass


class UnknownModelError(ValueError):
    """Model id absent from the price table — refuse to guess."""


# USD per million tokens (input, output). As of 2026-07 (Anthropic list
# prices; sonnet-5 shown at its non-introductory rate). Update alongside
# any caption-model default change.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
}


def image_tokens(width: int, height: int) -> int:
    """Anthropic vision token approximation: tokens ≈ (w × h) / 750."""
    return max(1, round((width * height) / 750))


@dataclass
class CostEstimate:
    calls: int
    input_tokens: int
    output_tokens: int
    usd: float

    def describe(self) -> str:
        return (
            f"{self.calls} API call(s), ~{self.input_tokens} input + "
            f"up to {self.output_tokens} output tokens ≈ ${self.usd:.4f}"
        )


def estimate(
    model: str,
    frame_dims_per_call: list[list[tuple[int, int]]],
    prompt_tokens_per_call: int,
    max_output_tokens_per_call: int,
) -> CostEstimate:
    """Conservative estimate for a batch of vision calls.

    frame_dims_per_call: one entry per uncached call, each a list of the
    (width, height) of the frames attached to that call.
    """
    if model not in PRICES_PER_MTOK:
        raise UnknownModelError(
            f"no price entry for model {model!r} — update mrw/costs.py "
            "PRICES_PER_MTOK (with its as-of date) before spending"
        )
    in_price, out_price = PRICES_PER_MTOK[model]
    input_tokens = sum(
        prompt_tokens_per_call + sum(image_tokens(w, h) for w, h in dims)
        for dims in frame_dims_per_call
    )
    calls = len(frame_dims_per_call)
    output_tokens = calls * max_output_tokens_per_call
    usd = (input_tokens * in_price + output_tokens * out_price) / 1_000_000
    return CostEstimate(
        calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=round(usd, 4),
    )


def aggregate_ledger(manifests: list[dict]) -> dict:
    """Sum recorded API usage across manifests (`mrw costs`).

    Reads the additive run.api_usage blocks; returns totals plus
    estimate-vs-actual where both were recorded.
    """
    totals = {
        "runs": 0,
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "usd": 0.0,
        "estimated_usd": 0.0,
    }
    for manifest in manifests:
        blocks = []
        documents = manifest.get("documents", {})
        for entry in documents.values():
            run = entry.get("run") or {}
            if run.get("api_usage"):
                blocks.append(run["api_usage"])
        stems_run = (manifest.get("stems") or {}).get("run") or {}
        if stems_run.get("api_usage"):
            blocks.append(stems_run["api_usage"])
        for usage in blocks:
            totals["runs"] += 1
            totals["calls"] += usage.get("calls", 0)
            totals["input_tokens"] += usage.get("input_tokens", 0)
            totals["output_tokens"] += usage.get("output_tokens", 0)
            totals["usd"] = round(totals["usd"] + usage.get("usd", 0.0), 4)
            totals["estimated_usd"] = round(
                totals["estimated_usd"] + usage.get("estimated_usd", 0.0), 4
            )
    return totals
