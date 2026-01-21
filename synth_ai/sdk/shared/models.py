"""Model identifier utilities.

This module provides basic model identifier handling for the SDK.
"""

from __future__ import annotations


class UnsupportedModelError(ValueError):
    """Raised when an unsupported model identifier is provided."""

    pass


def normalize_model_identifier(
    model: str,
    *,
    allow_finetuned_prefixes: bool = False,
) -> str:
    """Normalize a model identifier for API requests.

    This function normalizes model identifiers to ensure consistent formatting
    when making inference requests. The Synth backend handles routing to the
    appropriate provider.

    Args:
        model: Model identifier (e.g., "gpt-4o-mini", "claude-3-5-sonnet", "Qwen/Qwen3-4B")
        allow_finetuned_prefixes: Whether to allow finetuned model prefixes (e.g., "ft:gpt-4o:")

    Returns:
        Normalized model identifier string.

    Examples:
        >>> normalize_model_identifier("gpt-4o-mini")
        'gpt-4o-mini'
        >>> normalize_model_identifier("  GPT-4O-MINI  ")
        'gpt-4o-mini'
    """
    # Basic normalization: strip whitespace and lowercase for known providers
    normalized = model.strip()

    # Don't lowercase HuggingFace-style identifiers (contain /)
    if "/" not in normalized:
        normalized = normalized.lower()

    if not normalized:
        raise UnsupportedModelError("Model identifier cannot be empty")

    return normalized


__all__ = [
    "UnsupportedModelError",
    "normalize_model_identifier",
]
