"""GEPA configuration constants shared across DSPy adapters.

This file contains constants that match the centralized defaults in the monorepo
to ensure consistency across adapters.
"""
from __future__ import annotations

# DSPy GEPA reflection model default (matches monorepo/backend/app/routes/prompt_learning/algorithm/gepa/gepa_config.py)
DEFAULT_DSPY_GEPA_REFLECTION_MODEL: str = "groq/llama-3.3-70b-versatile"
DEFAULT_DSPY_GEPA_REFLECTION_PROVIDER: str = "groq"
DEFAULT_DSPY_GEPA_REFLECTION_INFERENCE_URL: str = "https://api.groq.com/openai/v1"

__all__ = [
    "DEFAULT_DSPY_GEPA_REFLECTION_MODEL",
    "DEFAULT_DSPY_GEPA_REFLECTION_PROVIDER",
    "DEFAULT_DSPY_GEPA_REFLECTION_INFERENCE_URL",
]




