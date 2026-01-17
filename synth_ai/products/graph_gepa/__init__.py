"""Backward-compatible Graph GEPA package alias."""

from __future__ import annotations

from synth_ai.products.graph_evolve import (
    ConversionError,
    ConversionResult,
    ConversionWarning,
    GraphOptimizationClient,
    GraphOptimizationConfig,
    convert_openai_sft,
    preview_conversion,
)

__all__ = [
    "GraphOptimizationConfig",
    "GraphOptimizationClient",
    "convert_openai_sft",
    "preview_conversion",
    "ConversionResult",
    "ConversionWarning",
    "ConversionError",
]
