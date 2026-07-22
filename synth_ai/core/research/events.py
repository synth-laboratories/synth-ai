"""Strict decoder for versioned Research swarm SSE events.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from synth_ai.core.http.streaming import SseEvent
from synth_ai.core.research.contracts.runtime_stream import (
    SwarmEvent,
    SwarmEventKind,
    SwarmEventPayload,
    SwarmHeartbeat,
)

KnownSwarmEvent = SwarmEvent
ResearchSwarmEvent = SwarmEvent
ResearchSwarmEventKind = SwarmEventKind


def decode_swarm_event(event: SseEvent) -> SwarmEvent:
    """Decode exactly one ``research.swarm_event.v1`` SSE frame."""
    return SwarmEvent.from_sse(event)


__all__ = [
    "KnownSwarmEvent",
    "ResearchSwarmEvent",
    "ResearchSwarmEventKind",
    "SwarmEvent",
    "SwarmEventKind",
    "SwarmEventPayload",
    "SwarmHeartbeat",
    "decode_swarm_event",
]
