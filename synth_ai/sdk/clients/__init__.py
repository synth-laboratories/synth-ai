"""Synth-wrapped LLM clients for interceptor routing."""

from __future__ import annotations

from .openai import AsyncOpenAI

__all__ = ["AsyncOpenAI"]

try:
    from .anthropic import AsyncAnthropic

    __all__.append("AsyncAnthropic")
except ImportError:
    pass  # anthropic is an optional dependency
