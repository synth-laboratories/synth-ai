"""Typed live event boundary for Research swarms."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from synth_ai.core.contracts.json_value import JsonValue
from synth_ai.core.http.streaming import SseEvent


class ResearchSwarmEventKind(StrEnum):
    SNAPSHOT = "snapshot"
    MESSAGE = "message"
    DELIVERY = "delivery"
    TIMELINE = "timeline"
    WORKER = "worker"
    HEARTBEAT = "heartbeat"
    TRANSCRIPT = "transcript"


@dataclass(frozen=True, slots=True)
class ResearchSwarmEvent:
    kind: ResearchSwarmEventKind
    payload: JsonValue
    event_id: str | None = None
    sequence: str | None = None
    swarm_id: str | None = None
    occurred_at: datetime | None = None
    transcript_cursor: str | None = None

    @classmethod
    def from_sse(cls, event: SseEvent) -> ResearchSwarmEvent:
        value = event.json_data()
        if not isinstance(value, dict):
            raise ValueError("swarm SSE data must be an object")
        kind = value.get("kind")
        if not isinstance(kind, str):
            raise ValueError("swarm SSE data requires string kind")
        timestamp = value.get("ts")
        occurred_at = None
        if timestamp is not None:
            if not isinstance(timestamp, str):
                raise ValueError("swarm SSE ts must be an ISO-8601 string")
            occurred_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if occurred_at.tzinfo is None:
                raise ValueError("swarm SSE ts must include a timezone")
        return cls(
            kind=ResearchSwarmEventKind(kind),
            payload=value.get("payload"),
            event_id=event.event_id,
            sequence=_optional_string(value.get("sequence"), "sequence"),
            swarm_id=_optional_string(value.get("run_id"), "run_id"),
            occurred_at=occurred_at,
            transcript_cursor=_optional_string(
                value.get("transcript_cursor"),
                "transcript_cursor",
            ),
        )


def _optional_string(value: JsonValue, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"swarm SSE {name} must be a string")
    return value


__all__ = ["ResearchSwarmEvent", "ResearchSwarmEventKind"]
