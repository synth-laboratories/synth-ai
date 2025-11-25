"""Tracing SDK - session trace types and utilities.

This module provides a cleaner import path for tracing types.
The underlying implementation remains in tracing_v3/.

Example:
    from synth_ai.sdk.tracing import SessionTrace
    
    trace = SessionTrace(session_id="...", time_steps=[...])
"""

from __future__ import annotations

# Re-export from data layer (which re-exports from tracing_v3)
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
]


