"""Core data structures for tracing v3.

BACKWARD COMPATIBILITY SHIM: This module now re-exports from synth_ai.data.traces.
All classes have moved to synth_ai.data.traces as the canonical location.

For new code, import directly from synth_ai.data.traces:
    from synth_ai.data.traces import SessionTrace, BaseEvent, ...

This module is preserved for backward compatibility with existing imports.
"""

from __future__ import annotations

# Re-export LLMCallRecord from its canonical location
from synth_ai.data.llm_calls import LLMCallRecord

# Re-export all trace classes from the canonical location
from synth_ai.data.traces import (
    BaseEvent,
    EnvironmentEvent,
    LMCAISEvent,
    RuntimeEvent,
    SessionEventMarkovBlanketMessage,
    SessionMessageContent,
    SessionTimeStep,
    SessionTrace,
    TimeRecord,
)

__all__ = [
    "SessionTrace",
    "SessionTimeStep",
    "BaseEvent",
    "RuntimeEvent",
    "EnvironmentEvent",
    "LMCAISEvent",
    "SessionEventMarkovBlanketMessage",
    "SessionMessageContent",
    "TimeRecord",
    "LLMCallRecord",
]
