"""Unified abstractions for recording LLM API calls (inputs and results).

BACKWARD COMPATIBILITY SHIM: This module now re-exports from synth_ai.data.llm_calls.
All classes have moved to synth_ai.data.llm_calls as the canonical location.

For new code, import directly from synth_ai.data.llm_calls:
    from synth_ai.data.llm_calls import LLMCallRecord, LLMUsage, ...

This module is preserved for backward compatibility with existing imports.
"""

from __future__ import annotations

# Re-export all LLM call record classes from the canonical location
from synth_ai.data.llm_calls import (
    LLMCallRecord,
    LLMChunk,
    LLMContentPart,
    LLMMessage,
    LLMRequestParams,
    LLMUsage,
    ToolCallResult,
    ToolCallSpec,
)


def compute_latency_ms(record: LLMCallRecord) -> int | None:
    """Compute and update latency_ms from timestamps if available.

    This helper function remains here as it contains logic, not just data.
    """
    if record.started_at and record.completed_at:
        delta = int((record.completed_at - record.started_at).total_seconds() * 1000)
        record.latency_ms = delta
        return delta
    return record.latency_ms


__all__ = [
    "LLMUsage",
    "LLMRequestParams",
    "LLMContentPart",
    "LLMMessage",
    "ToolCallSpec",
    "ToolCallResult",
    "LLMChunk",
    "LLMCallRecord",
    "compute_latency_ms",
]
