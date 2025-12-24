"""Compatibility wrapper for OpenAI SFT converters."""

from __future__ import annotations

from synth_ai.products.graph_evolve.converters.openai_sft import (
    ConversionError,
    ConversionResult,
    ConversionWarning,
    convert_openai_sft,
    detect_system_prompt,
    extract_fields,
    infer_template,
    parse_sft_example,
    preview_conversion,
    validate_sft_examples,
)

__all__ = [
    "ConversionError",
    "ConversionResult",
    "ConversionWarning",
    "convert_openai_sft",
    "detect_system_prompt",
    "extract_fields",
    "infer_template",
    "parse_sft_example",
    "preview_conversion",
    "validate_sft_examples",
]
