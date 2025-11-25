"""Model pricing utilities for Synth AI SDK.

This module provides per-token pricing data for supported models,
used by status commands and cost estimation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenRates:
    """Per-token pricing rates in USD."""

    input_usd: float
    output_usd: float


# Pricing data - updated Nov 2025
# Organized by provider -> model -> rates
MODEL_PRICES: dict[str, dict[str, TokenRates]] = {
    "openai": {
        # GPT-5 family
        "gpt-5": TokenRates(input_usd=0.00000125, output_usd=0.00001000),
        "gpt-5-mini": TokenRates(input_usd=0.00000025, output_usd=0.00000200),
        "gpt-5-nano": TokenRates(input_usd=0.00000005, output_usd=0.00000040),
        # GPT-4.1 family
        "gpt-4.1": TokenRates(input_usd=0.00000200, output_usd=0.00000800),
        "gpt-4.1-mini": TokenRates(input_usd=0.00000040, output_usd=0.00000160),
        "gpt-4.1-nano": TokenRates(input_usd=0.00000010, output_usd=0.00000040),
        # GPT-4o family
        "gpt-4o": TokenRates(input_usd=0.00000250, output_usd=0.00001000),
        "gpt-4o-mini": TokenRates(input_usd=0.00000015, output_usd=0.00000060),
    },
    "anthropic": {
        "claude-3-5-sonnet": TokenRates(input_usd=0.000003, output_usd=0.000015),
        "claude-3-5-haiku": TokenRates(input_usd=0.00000025, output_usd=0.00000125),
        "claude-3-opus": TokenRates(input_usd=0.000015, output_usd=0.000075),
    },
    "google": {
        "gemini-2.5-pro": TokenRates(input_usd=0.00000125, output_usd=0.00001000),
        "gemini-2.5-flash": TokenRates(input_usd=0.00000030, output_usd=0.00000250),
        "gemini-2.5-flash-lite": TokenRates(input_usd=0.00000010, output_usd=0.00000040),
    },
    "groq": {
        "llama-3.3-70b-versatile": TokenRates(
            input_usd=0.000000590, output_usd=0.000000790
        ),
        "llama-3.1-8b-instant": TokenRates(
            input_usd=0.000000050, output_usd=0.000000080
        ),
    },
}


def get_token_rates(model: str, provider: str | None = None) -> TokenRates | None:
    """Get token rates for a model.

    Args:
        model: Model identifier (e.g., "gpt-4o")
        provider: Optional provider hint (e.g., "openai")

    Returns:
        TokenRates if found, None otherwise
    """
    if provider:
        provider_rates = MODEL_PRICES.get(provider.lower())
        if provider_rates:
            return provider_rates.get(model.lower())

    # Search all providers
    for provider_rates in MODEL_PRICES.values():
        if model.lower() in provider_rates:
            return provider_rates[model.lower()]

    return None


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    provider: str | None = None,
) -> float | None:
    """Estimate cost for token usage.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model identifier
        provider: Optional provider hint

    Returns:
        Estimated cost in USD, or None if model not found
    """
    rates = get_token_rates(model, provider)
    if rates is None:
        return None
    return (input_tokens * rates.input_usd) + (output_tokens * rates.output_usd)


__all__ = [
    "TokenRates",
    "MODEL_PRICES",
    "get_token_rates",
    "estimate_cost",
]

