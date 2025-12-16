"""Dataset converters for Graph GEPA.

This module provides converters to transform common dataset formats
into ADAS format for use with Graph GEPA optimization.

Supported formats:
- OpenAI SFT: JSONL with messages array (system, user, assistant roles)

Example:
    >>> from synth_ai.products.graph_gepa.converters import convert_openai_sft
    >>>
    >>> # Convert from file
    >>> result = convert_openai_sft("training_data.jsonl")
    >>> adas_dataset = result.dataset
    >>>
    >>> # Use in GraphOptimizationConfig
    >>> from synth_ai.products.graph_gepa import GraphOptimizationConfig
    >>> config = GraphOptimizationConfig(
    ...     dataset_name="my_qa_task",
    ...     dataset=adas_dataset,
    ...     graph_type="policy",
    ...     ...
    ... )
"""

from __future__ import annotations

from .openai_sft import (
    ConversionError,
    ConversionResult,
    ConversionWarning,
    convert_openai_sft,
    preview_conversion,
)

__all__ = [
    "convert_openai_sft",
    "preview_conversion",
    "ConversionResult",
    "ConversionWarning",
    "ConversionError",
]
