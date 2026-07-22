"""Typed snapshot payload carried by the public swarm runtime stream.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.research.contracts._transcript_wire import (
    exact_object,
    non_negative_int,
    non_negative_int_mapping,
    object_tuple,
    optional_non_negative_int,
    optional_string,
    required_bool,
    required_datetime,
    required_string,
    string_mapping,
    string_tuple,
)
from synth_ai.core.research.contracts.common import ProjectId, SwarmId

_RUNTIME_STREAM_SOURCE = "smr_control_plane.redis.runtime_stream.v1"


class RuntimeTerminalPhase(StrEnum):
    TERMINAL = "terminal"
    NONTERMINAL = "nonterminal"


@dataclass(frozen=True, slots=True)
class RuntimeWorkersSnapshot:
    project_id: ProjectId
    swarm_id: SwarmId
    status: str
    last_tick_at: datetime
    active_count: int
    live_workers_limit: int | None
    active_workers: tuple[str, ...]
    worker_ids: tuple[str, ...]
    worker_counts: dict[str, int]
    workers: tuple[JsonObject, ...]
    links: dict[str, str]

    @classmethod
    def from_wire(cls, value: JsonValue) -> RuntimeWorkersSnapshot:
        payload = exact_object(
            value,
            label="runtime workers snapshot",
            fields=frozenset(
                {
                    "project_id",
                    "run_id",
                    "status",
                    "last_tick_at",
                    "active_count",
                    "live_workers_limit",
                    "active_workers",
                    "worker_ids",
                    "worker_counts",
                    "partial",
                    "partial_reason",
                    "workers",
                    "links",
                }
            ),
        )
        if not required_bool(payload, "partial"):
            raise ValueError("runtime workers snapshot must declare partial=true")
        if required_string(payload, "partial_reason") != ("smr_control_plane_runtime_stream"):
            raise ValueError("unsupported runtime workers partial_reason")
        return cls(
            project_id=ProjectId(required_string(payload, "project_id")),
            swarm_id=SwarmId(required_string(payload, "run_id")),
            status=required_string(payload, "status"),
            last_tick_at=required_datetime(payload, "last_tick_at"),
            active_count=non_negative_int(payload, "active_count"),
            live_workers_limit=optional_non_negative_int(
                payload,
                "live_workers_limit",
            ),
            active_workers=string_tuple(payload, "active_workers"),
            worker_ids=string_tuple(payload, "worker_ids"),
            worker_counts=non_negative_int_mapping(payload, "worker_counts"),
            workers=object_tuple(payload, "workers"),
            links=string_mapping(payload, "links"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "project_id": self.project_id,
            "run_id": self.swarm_id,
            "status": self.status,
            "last_tick_at": self.last_tick_at.isoformat(),
            "active_count": self.active_count,
            "live_workers_limit": self.live_workers_limit,
            "active_workers": list(self.active_workers),
            "worker_ids": list(self.worker_ids),
            "worker_counts": dict(self.worker_counts),
            "partial": True,
            "partial_reason": "smr_control_plane_runtime_stream",
            "workers": [dict(worker) for worker in self.workers],
            "links": dict(self.links),
        }


@dataclass(frozen=True, slots=True)
class RuntimeTimelineMetadata:
    last_event_id: str | None

    @classmethod
    def from_wire(cls, value: JsonValue) -> RuntimeTimelineMetadata:
        payload = exact_object(
            value,
            label="runtime timeline metadata",
            fields=frozenset({"source", "last_event_id"}),
        )
        _require_runtime_source(payload)
        return cls(last_event_id=optional_string(payload, "last_event_id"))

    def to_wire(self) -> JsonObject:
        return {"source": _RUNTIME_STREAM_SOURCE, "last_event_id": self.last_event_id}


@dataclass(frozen=True, slots=True)
class RuntimeLifecycleSnapshot:
    terminal_phase: RuntimeTerminalPhase
    terminal_outcome: str | None
    updated_at: datetime
    metadata: RuntimeTimelineMetadata

    @classmethod
    def from_wire(cls, value: JsonValue) -> RuntimeLifecycleSnapshot:
        payload = exact_object(
            value,
            label="runtime lifecycle snapshot",
            fields=frozenset(
                {
                    "authority_phase",
                    "terminal_phase",
                    "terminal_outcome",
                    "updated_at",
                    "metadata",
                }
            ),
        )
        if required_string(payload, "authority_phase") != "control_plane_stream":
            raise ValueError("unsupported runtime lifecycle authority_phase")
        return cls(
            terminal_phase=RuntimeTerminalPhase(required_string(payload, "terminal_phase")),
            terminal_outcome=optional_string(payload, "terminal_outcome"),
            updated_at=required_datetime(payload, "updated_at"),
            metadata=RuntimeTimelineMetadata.from_wire(payload.get("metadata")),
        )

    def to_wire(self) -> JsonObject:
        return {
            "authority_phase": "control_plane_stream",
            "terminal_phase": self.terminal_phase.value,
            "terminal_outcome": self.terminal_outcome,
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class RuntimeTimelineEvent:
    event_id: str
    created_at: datetime
    summary: str
    state: str
    last_event_id: str | None

    @classmethod
    def from_wire(cls, value: JsonValue) -> RuntimeTimelineEvent:
        payload = exact_object(
            value,
            label="runtime timeline event",
            fields=frozenset(
                {
                    "event_id",
                    "created_at",
                    "kind",
                    "source",
                    "summary",
                    "state",
                    "detail",
                }
            ),
        )
        if required_string(payload, "kind") != "control_plane_snapshot":
            raise ValueError("unsupported runtime timeline event kind")
        _require_runtime_source(payload)
        detail = exact_object(
            payload.get("detail"),
            label="runtime timeline event detail",
            fields=frozenset({"source", "last_event_id"}),
        )
        _require_runtime_source(detail)
        return cls(
            event_id=required_string(payload, "event_id"),
            created_at=required_datetime(payload, "created_at"),
            summary=required_string(payload, "summary"),
            state=required_string(payload, "state"),
            last_event_id=optional_string(detail, "last_event_id"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "event_id": self.event_id,
            "created_at": self.created_at.isoformat(),
            "kind": "control_plane_snapshot",
            "source": _RUNTIME_STREAM_SOURCE,
            "summary": self.summary,
            "state": self.state,
            "detail": {
                "source": _RUNTIME_STREAM_SOURCE,
                "last_event_id": self.last_event_id,
            },
        }


@dataclass(frozen=True, slots=True)
class RuntimeTimelineSnapshot:
    project_id: ProjectId
    swarm_id: SwarmId
    authority_state: str
    authority_persisted_state: str
    lifecycle: RuntimeLifecycleSnapshot
    latest_summary: str
    events: tuple[RuntimeTimelineEvent, ...]

    @classmethod
    def from_wire(cls, value: JsonValue) -> RuntimeTimelineSnapshot:
        payload = exact_object(
            value,
            label="runtime timeline snapshot",
            fields=frozenset(
                {
                    "project_id",
                    "run_id",
                    "authority_source",
                    "authority_state",
                    "authority_persisted_state",
                    "lifecycle",
                    "latest_kind",
                    "latest_summary",
                    "events",
                }
            ),
        )
        if required_string(payload, "authority_source") != _RUNTIME_STREAM_SOURCE:
            raise ValueError("unsupported runtime timeline authority_source")
        if required_string(payload, "latest_kind") != "control_plane_snapshot":
            raise ValueError("unsupported runtime timeline latest_kind")
        raw_events = payload.get("events")
        if not isinstance(raw_events, list):
            raise ValueError("runtime timeline events must be an array")
        return cls(
            project_id=ProjectId(required_string(payload, "project_id")),
            swarm_id=SwarmId(required_string(payload, "run_id")),
            authority_state=required_string(payload, "authority_state"),
            authority_persisted_state=required_string(
                payload,
                "authority_persisted_state",
            ),
            lifecycle=RuntimeLifecycleSnapshot.from_wire(payload.get("lifecycle")),
            latest_summary=required_string(payload, "latest_summary"),
            events=tuple(RuntimeTimelineEvent.from_wire(item) for item in raw_events),
        )

    def to_wire(self) -> JsonObject:
        return {
            "project_id": self.project_id,
            "run_id": self.swarm_id,
            "authority_source": _RUNTIME_STREAM_SOURCE,
            "authority_state": self.authority_state,
            "authority_persisted_state": self.authority_persisted_state,
            "lifecycle": self.lifecycle.to_wire(),
            "latest_kind": "control_plane_snapshot",
            "latest_summary": self.latest_summary,
            "events": [event.to_wire() for event in self.events],
        }


@dataclass(frozen=True, slots=True)
class SwarmRuntimeSnapshot:
    project_id: ProjectId
    swarm_id: SwarmId
    workers: RuntimeWorkersSnapshot
    actors: tuple[JsonObject, ...]
    messages: tuple[JsonObject, ...]
    deliveries: tuple[JsonObject, ...]
    timeline: RuntimeTimelineSnapshot

    def __post_init__(self) -> None:
        identities = (
            self.workers.project_id,
            self.workers.swarm_id,
            self.timeline.project_id,
            self.timeline.swarm_id,
        )
        if identities != (
            self.project_id,
            self.swarm_id,
            self.project_id,
            self.swarm_id,
        ):
            raise ValueError("runtime snapshot nested identities must match")

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmRuntimeSnapshot:
        payload = exact_object(
            value,
            label="swarm runtime snapshot",
            fields=frozenset(
                {
                    "project_id",
                    "run_id",
                    "workers",
                    "actors",
                    "messages",
                    "deliveries",
                    "timeline",
                }
            ),
        )
        return cls(
            project_id=ProjectId(required_string(payload, "project_id")),
            swarm_id=SwarmId(required_string(payload, "run_id")),
            workers=RuntimeWorkersSnapshot.from_wire(payload.get("workers")),
            actors=object_tuple(payload, "actors"),
            messages=object_tuple(payload, "messages"),
            deliveries=object_tuple(payload, "deliveries"),
            timeline=RuntimeTimelineSnapshot.from_wire(payload.get("timeline")),
        )

    def to_wire(self) -> JsonObject:
        return {
            "project_id": self.project_id,
            "run_id": self.swarm_id,
            "workers": self.workers.to_wire(),
            "actors": [dict(actor) for actor in self.actors],
            "messages": [dict(message) for message in self.messages],
            "deliveries": [dict(delivery) for delivery in self.deliveries],
            "timeline": self.timeline.to_wire(),
        }


def _require_runtime_source(payload: JsonObject) -> None:
    if required_string(payload, "source") != _RUNTIME_STREAM_SOURCE:
        raise ValueError("unsupported runtime stream source")


__all__ = [
    "RuntimeLifecycleSnapshot",
    "RuntimeTerminalPhase",
    "RuntimeTimelineEvent",
    "RuntimeTimelineMetadata",
    "RuntimeTimelineSnapshot",
    "RuntimeWorkersSnapshot",
    "SwarmRuntimeSnapshot",
]
