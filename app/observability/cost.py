"""Rough USD cost estimates for LLM calls.

Prices are approximate list rates (per 1M tokens) for portfolio cost awareness.
Local / free paths (Ollama, default HF Inference) are treated as ``0.0``.
Unknown models fall back to the provider default or zero.
"""

from __future__ import annotations

# (input_usd_per_1m, output_usd_per_1m)
_MODEL_RATES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
}

_PROVIDER_DEFAULTS: dict[str, tuple[float, float]] = {
    "openai": (0.15, 0.60),  # assume mini-class if model unknown
    "anthropic": (0.80, 4.00),  # assume haiku-class if model unknown
    "huggingface": (0.0, 0.0),
    "ollama": (0.0, 0.0),
}


def _lookup_rates(provider: str, model: str) -> tuple[float, float]:
    lowered = model.lower()
    for key, rates in _MODEL_RATES.items():
        if key in lowered:
            return rates
    return _PROVIDER_DEFAULTS.get(provider.lower(), (0.0, 0.0))


def estimate_cost_usd(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate USD cost for one generation from token counts."""

    in_rate, out_rate = _lookup_rates(provider, model)
    cost = (input_tokens / 1_000_000.0) * in_rate + (output_tokens / 1_000_000.0) * out_rate
    return round(cost, 8)
