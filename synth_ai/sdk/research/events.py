"""Strict decoder for versioned Research swarm SSE events.

# See: testing/specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from synth_ai.core.http.streaming import SseEvent
from synth_ai.sdk.research.contracts.runtime_stream import (
    SwarmEvent,
    SwarmEventKind,
    SwarmEventPayload,
    SwarmHeartbeat,
)


def decode_swarm_event(event: SseEvent) -> SwarmEvent:
    """Decode exactly one ``research.swarm_event.v1`` SSE frame."""
    return SwarmEvent.from_sse(event)


__all__ = [
    "SwarmEvent",
    "SwarmEventKind",
    "SwarmEventPayload",
    "SwarmHeartbeat",
    "decode_swarm_event",
]
