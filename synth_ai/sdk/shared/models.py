"""Model identifier utilities.

This module provides basic model identifier handling for the SDK.
"""

from __future__ import annotations

from typing import Any, Dict

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for sdk.shared.models.") from exc


def _require_rust() -> Any:
    if synth_ai_py is None or not hasattr(synth_ai_py, "normalize_model_identifier"):
        raise UnsupportedModelError(
            "Rust core model utilities required; synth_ai_py is unavailable."
        )
    return synth_ai_py


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
    rust = _require_rust()
    return rust.normalize_model_identifier(model, allow_finetuned_prefixes=allow_finetuned_prefixes)


def detect_model_provider(model: str) -> str:
    """Detect the model provider from a model identifier."""
    rust = _require_rust()
    if not hasattr(rust, "detect_model_provider"):
        raise UnsupportedModelError(
            "Rust core model utilities required; synth_ai_py is unavailable."
        )
    return rust.detect_model_provider(model)


def supported_models() -> Dict[str, Any]:
    """Return the supported model registry from Rust core."""
    rust = _require_rust()
    if not hasattr(rust, "supported_models"):
        raise UnsupportedModelError(
            "Rust core supported models required; synth_ai_py is unavailable."
        )
    return rust.supported_models()


__all__ = [
    "UnsupportedModelError",
    "normalize_model_identifier",
    "detect_model_provider",
    "supported_models",
]
