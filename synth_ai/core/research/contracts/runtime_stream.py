"""Versioned runtime-stream event envelope for Research swarms.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TypeAlias

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.streaming import SseEvent
from synth_ai.core.research.contracts._transcript_wire import (
    exact_object,
    optional_string,
    required_datetime,
    required_string,
)
from synth_ai.core.research.contracts.common import SwarmId
from synth_ai.core.research.contracts.runtime_stream_snapshot import (
    SwarmRuntimeSnapshot,
)
from synth_ai.core.research.contracts.transcript import SwarmTranscriptEvent

_RUNTIME_STREAM_SOURCE = "smr_control_plane.redis.runtime_stream.v1"


class SwarmEventKind(StrEnum):
    SNAPSHOT = "snapshot"
    TRANSCRIPT = "transcript"
    HEARTBEAT = "heartbeat"


@dataclass(frozen=True, slots=True)
class SwarmHeartbeat:
    version: str

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmHeartbeat:
        payload = exact_object(
            value,
            label="swarm heartbeat",
            fields=frozenset({"version", "source"}),
        )
        if required_string(payload, "source") != _RUNTIME_STREAM_SOURCE:
            raise ValueError("unsupported swarm heartbeat source")
        return cls(version=required_string(payload, "version"))

    def to_wire(self) -> JsonObject:
        return {"version": self.version, "source": _RUNTIME_STREAM_SOURCE}


SwarmEventPayload: TypeAlias = (
    SwarmRuntimeSnapshot | SwarmTranscriptEvent | SwarmHeartbeat
)


@dataclass(frozen=True, slots=True)
class SwarmEvent:
    """One strict ``research.swarm_event.v1`` SSE envelope."""

    sequence: str
    swarm_id: SwarmId
    occurred_at: datetime
    kind: SwarmEventKind
    payload: SwarmEventPayload
    state_version: str
    transcript_cursor: str | None

    @property
    def event_id(self) -> str:
        """SSE resume token; identical to the versioned envelope sequence."""
        return self.sequence

    @classmethod
    def from_sse(cls, event: SseEvent) -> SwarmEvent:
        value = event.json_data()
        payload = exact_object(
            value,
            label="swarm SSE envelope",
            fields=frozenset(
                {
                    "schema_version",
                    "sequence",
                    "run_id",
                    "ts",
                    "kind",
                    "payload",
                    "state_version",
                    "transcript_cursor",
                }
            ),
        )
        if required_string(payload, "schema_version") != "research.swarm_event.v1":
            raise ValueError("unsupported swarm SSE schema_version")
        kind = SwarmEventKind(required_string(payload, "kind"))
        if event.event != kind.value:
            raise ValueError("SSE event field must match the envelope kind")
        sequence = required_string(payload, "sequence")
        if event.event_id != sequence:
            raise ValueError("SSE id field must match the envelope sequence")
        event_payload = _decode_payload(kind, payload.get("payload"))
        swarm_id = SwarmId(required_string(payload, "run_id"))
        if isinstance(event_payload, SwarmRuntimeSnapshot):
            if event_payload.swarm_id != swarm_id:
                raise ValueError("snapshot payload run_id must match its envelope")
        elif isinstance(event_payload, SwarmTranscriptEvent):
            if event_payload.swarm_id != swarm_id:
                raise ValueError("transcript payload run_id must match its envelope")
        return cls(
            sequence=sequence,
            swarm_id=swarm_id,
            occurred_at=required_datetime(payload, "ts"),
            kind=kind,
            payload=event_payload,
            state_version=required_string(payload, "state_version"),
            transcript_cursor=optional_string(payload, "transcript_cursor"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "schema_version": "research.swarm_event.v1",
            "sequence": self.sequence,
            "run_id": self.swarm_id,
            "ts": self.occurred_at.isoformat(),
            "kind": self.kind.value,
            "payload": self.payload.to_wire(),
            "state_version": self.state_version,
            "transcript_cursor": self.transcript_cursor,
        }


def _decode_payload(kind: SwarmEventKind, value: JsonValue) -> SwarmEventPayload:
    if kind is SwarmEventKind.SNAPSHOT:
        return SwarmRuntimeSnapshot.from_wire(value)
    if kind is SwarmEventKind.TRANSCRIPT:
        return SwarmTranscriptEvent.from_wire(value)
    if kind is SwarmEventKind.HEARTBEAT:
        return SwarmHeartbeat.from_wire(value)
    raise AssertionError(f"unhandled swarm event kind {kind.value}")


__all__ = [
    "SwarmEvent",
    "SwarmEventKind",
    "SwarmEventPayload",
    "SwarmHeartbeat",
]
