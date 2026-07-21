"""Versioned, correlation-aware event protocol for Research swarms.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TypeAlias

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.streaming import SseEvent
from synth_ai.core.research.contracts.common import SwarmId


class SwarmEventKind(StrEnum):
    SNAPSHOT = "snapshot"
    MESSAGE = "message"
    DELIVERY = "delivery"
    TIMELINE = "timeline"
    WORKER = "worker"
    HEARTBEAT = "heartbeat"
    TRANSCRIPT = "transcript"


@dataclass(frozen=True, slots=True)
class SwarmTelemetry:
    """Identity and ordering envelope shared by every event consumer."""

    schema_version: str
    swarm_id: SwarmId | None
    event_id: str | None
    sequence: str | None
    occurred_at: datetime | None
    request_id: str | None
    correlation_id: str | None
    causation_id: str | None
    transcript_cursor: str | None


@dataclass(frozen=True, slots=True)
class KnownSwarmEvent:
    kind: SwarmEventKind
    payload: JsonValue
    telemetry: SwarmTelemetry

    @classmethod
    def from_sse(cls, event: SseEvent) -> SwarmEvent:
        return decode_swarm_event(event)

    @property
    def event_id(self) -> str | None:
        return self.telemetry.event_id

    @property
    def sequence(self) -> str | None:
        return self.telemetry.sequence

    @property
    def swarm_id(self) -> SwarmId | None:
        return self.telemetry.swarm_id

    @property
    def occurred_at(self) -> datetime | None:
        return self.telemetry.occurred_at

    @property
    def transcript_cursor(self) -> str | None:
        return self.telemetry.transcript_cursor


@dataclass(frozen=True, slots=True)
class UnknownSwarmEvent:
    """Forward-compatible event whose kind is newer than this SDK."""

    kind: str
    payload: JsonValue
    telemetry: SwarmTelemetry
    raw: JsonObject

    @property
    def event_id(self) -> str | None:
        return self.telemetry.event_id

    @property
    def sequence(self) -> str | None:
        return self.telemetry.sequence

    @property
    def swarm_id(self) -> SwarmId | None:
        return self.telemetry.swarm_id

    @property
    def occurred_at(self) -> datetime | None:
        return self.telemetry.occurred_at

    @property
    def transcript_cursor(self) -> str | None:
        return self.telemetry.transcript_cursor


SwarmEvent: TypeAlias = KnownSwarmEvent | UnknownSwarmEvent
ResearchSwarmEventKind = SwarmEventKind
ResearchSwarmEvent = KnownSwarmEvent


def decode_swarm_event(event: SseEvent) -> SwarmEvent:
    value = event.json_data()
    if not isinstance(value, dict):
        raise ValueError("swarm SSE data must be an object")
    kind_value = value.get("kind")
    if not isinstance(kind_value, str) or not kind_value.strip():
        raise ValueError("swarm SSE data requires non-empty string kind")
    timestamp = value.get("ts")
    occurred_at = None
    if timestamp is not None:
        if not isinstance(timestamp, str):
            raise ValueError("swarm SSE ts must be an ISO-8601 string")
        occurred_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if occurred_at.tzinfo is None:
            raise ValueError("swarm SSE ts must include a timezone")
    swarm_id = _optional_string(value.get("run_id"), "run_id")
    telemetry = SwarmTelemetry(
        schema_version=_optional_string(value.get("schema_version"), "schema_version")
        or "research.swarm_event.v1",
        swarm_id=SwarmId(swarm_id) if swarm_id is not None else None,
        event_id=event.event_id,
        sequence=_optional_string(value.get("sequence"), "sequence"),
        occurred_at=occurred_at,
        request_id=_optional_string(value.get("request_id"), "request_id"),
        correlation_id=_optional_string(value.get("correlation_id"), "correlation_id"),
        causation_id=_optional_string(value.get("causation_id"), "causation_id"),
        transcript_cursor=_optional_string(
            value.get("transcript_cursor"),
            "transcript_cursor",
        ),
    )
    try:
        kind = SwarmEventKind(kind_value)
    except ValueError:
        return UnknownSwarmEvent(
            kind=kind_value,
            payload=value.get("payload"),
            telemetry=telemetry,
            raw=dict(value),
        )
    return KnownSwarmEvent(
        kind=kind,
        payload=value.get("payload"),
        telemetry=telemetry,
    )


def _optional_string(value: JsonValue, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"swarm SSE {name} must be a string")
    return value


__all__ = [
    "ResearchSwarmEvent",
    "ResearchSwarmEventKind",
    "SwarmEvent",
    "SwarmEventKind",
    "KnownSwarmEvent",
    "SwarmTelemetry",
    "UnknownSwarmEvent",
    "decode_swarm_event",
]
