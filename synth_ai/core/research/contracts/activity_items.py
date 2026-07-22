"""Typed actors, tasks, messages, events, and outputs in swarm activity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.research.contracts._activity_wire import (
    exact_object,
    optional_datetime,
    text_tuple,
)
from synth_ai.core.research.contracts._wire import (
    optional_text,
    required_datetime,
    required_text,
)
from synth_ai.core.research.contracts.common import (
    ActivityEventId,
    ActorId,
    MessageId,
    TaskId,
    WorkProductId,
)
from synth_ai.core.research.contracts.evidence import (
    WorkProductKind,
    WorkProductStatus,
)


class ActivityActorState(StrEnum):
    CREATED = "created"
    CLAIMED = "claimed"
    STARTING = "starting"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    CANCELED = "canceled"
    CANCELLED = "cancelled"
    STALE = "stale"


class ActivityTaskState(StrEnum):
    PLANNED = "planned"
    READY = "ready"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    EXECUTING = "executing"
    RUNNING = "running"
    REVIEW_REQUIRED = "review_required"
    REPAIR_REQUIRED = "repair_required"
    BLOCKED = "blocked"
    SUPERSEDED = "superseded"
    DONE = "done"
    FAILED = "failed"
    STOPPED = "stopped"


class ActivityVisibility(StrEnum):
    PUBLIC = "public"


@dataclass(frozen=True, slots=True)
class ActivityActor:
    actor_id: ActorId
    role: str
    label: str
    state: ActivityActorState
    task_ids: tuple[TaskId, ...]
    task_keys: tuple[str, ...]
    spawned_at: datetime | None
    completed_at: datetime | None
    latest_event_at: datetime | None
    latest_summary: str | None

    @classmethod
    def from_wire(cls, value: JsonValue) -> ActivityActor:
        payload = exact_object(
            value,
            label="swarm activity actor",
            fields=frozenset(
                {
                    "actor_id",
                    "role",
                    "label",
                    "state",
                    "task_ids",
                    "task_keys",
                    "spawned_at",
                    "completed_at",
                    "latest_event_at",
                    "latest_summary",
                }
            ),
        )
        return cls(
            actor_id=ActorId(required_text(payload, "actor_id")),
            role=required_text(payload, "role"),
            label=required_text(payload, "label"),
            state=ActivityActorState(required_text(payload, "state")),
            task_ids=tuple(TaskId(item) for item in text_tuple(payload, "task_ids")),
            task_keys=text_tuple(payload, "task_keys"),
            spawned_at=optional_datetime(payload, "spawned_at"),
            completed_at=optional_datetime(payload, "completed_at"),
            latest_event_at=optional_datetime(payload, "latest_event_at"),
            latest_summary=optional_text(payload, "latest_summary"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "actor_id": self.actor_id,
            "role": self.role,
            "label": self.label,
            "state": self.state.value,
            "task_ids": list(self.task_ids),
            "task_keys": list(self.task_keys),
            "spawned_at": self.spawned_at.isoformat() if self.spawned_at else None,
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
            "latest_event_at": (self.latest_event_at.isoformat() if self.latest_event_at else None),
            "latest_summary": self.latest_summary,
        }


@dataclass(frozen=True, slots=True)
class ActivityTask:
    task_id: TaskId
    task_key: str
    title: str
    state: ActivityTaskState
    internal_state: ActivityTaskState | None
    objective_id: str | None
    owner_actor_id: ActorId | None
    created_by_actor_id: ActorId | None
    updated_by_actor_id: ActorId | None
    created_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    latest_summary: str | None
    work_product_ids: tuple[WorkProductId, ...]

    @classmethod
    def from_wire(cls, value: JsonValue) -> ActivityTask:
        payload = exact_object(
            value,
            label="swarm activity task",
            fields=frozenset(
                {
                    "task_id",
                    "task_key",
                    "title",
                    "public_task_state",
                    "task_state",
                    "objective_id",
                    "owner_actor_id",
                    "created_by_actor_id",
                    "updated_by_actor_id",
                    "created_at",
                    "started_at",
                    "completed_at",
                    "latest_summary",
                    "work_product_refs",
                }
            ),
        )
        internal_state = optional_text(payload, "task_state")
        owner_actor_id = optional_text(payload, "owner_actor_id")
        created_by_actor_id = optional_text(payload, "created_by_actor_id")
        updated_by_actor_id = optional_text(payload, "updated_by_actor_id")
        return cls(
            task_id=TaskId(required_text(payload, "task_id")),
            task_key=required_text(payload, "task_key"),
            title=required_text(payload, "title"),
            state=ActivityTaskState(required_text(payload, "public_task_state")),
            internal_state=(
                ActivityTaskState(internal_state) if internal_state is not None else None
            ),
            objective_id=optional_text(payload, "objective_id"),
            owner_actor_id=(ActorId(owner_actor_id) if owner_actor_id else None),
            created_by_actor_id=(ActorId(created_by_actor_id) if created_by_actor_id else None),
            updated_by_actor_id=(ActorId(updated_by_actor_id) if updated_by_actor_id else None),
            created_at=optional_datetime(payload, "created_at"),
            started_at=optional_datetime(payload, "started_at"),
            completed_at=optional_datetime(payload, "completed_at"),
            latest_summary=optional_text(payload, "latest_summary"),
            work_product_ids=tuple(
                WorkProductId(item) for item in text_tuple(payload, "work_product_refs")
            ),
        )

    def to_wire(self) -> JsonObject:
        return {
            "task_id": self.task_id,
            "task_key": self.task_key,
            "title": self.title,
            "public_task_state": self.state.value,
            "task_state": (self.internal_state.value if self.internal_state is not None else None),
            "objective_id": self.objective_id,
            "owner_actor_id": self.owner_actor_id,
            "created_by_actor_id": self.created_by_actor_id,
            "updated_by_actor_id": self.updated_by_actor_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
            "latest_summary": self.latest_summary,
            "work_product_refs": list(self.work_product_ids),
        }


@dataclass(frozen=True, slots=True)
class ActivityMessage:
    message_id: MessageId
    event_id: ActivityEventId | None
    actor_id: ActorId | None
    participant_role: str | None
    task_id: TaskId | None
    task_key: str | None
    occurred_at: datetime
    title: str
    body: str
    visibility: ActivityVisibility

    @classmethod
    def from_wire(cls, value: JsonValue) -> ActivityMessage:
        payload = exact_object(
            value,
            label="swarm activity message",
            fields=frozenset(
                {
                    "message_id",
                    "event_id",
                    "actor_id",
                    "participant_role",
                    "task_id",
                    "task_key",
                    "occurred_at",
                    "title",
                    "body",
                    "visibility",
                }
            ),
        )
        event_id = optional_text(payload, "event_id")
        actor_id = optional_text(payload, "actor_id")
        task_id = optional_text(payload, "task_id")
        return cls(
            message_id=MessageId(required_text(payload, "message_id")),
            event_id=ActivityEventId(event_id) if event_id else None,
            actor_id=ActorId(actor_id) if actor_id else None,
            participant_role=optional_text(payload, "participant_role"),
            task_id=TaskId(task_id) if task_id else None,
            task_key=optional_text(payload, "task_key"),
            occurred_at=required_datetime(payload, "occurred_at"),
            title=required_text(payload, "title"),
            body=required_text(payload, "body"),
            visibility=ActivityVisibility(required_text(payload, "visibility")),
        )

    def to_wire(self) -> JsonObject:
        return {
            "message_id": self.message_id,
            "event_id": self.event_id,
            "actor_id": self.actor_id,
            "participant_role": self.participant_role,
            "task_id": self.task_id,
            "task_key": self.task_key,
            "occurred_at": self.occurred_at.isoformat(),
            "title": self.title,
            "body": self.body,
            "visibility": self.visibility.value,
        }


@dataclass(frozen=True, slots=True)
class ActivityEvent:
    event_id: ActivityEventId
    kind: str
    occurred_at: datetime
    actor_id: ActorId | None
    task_id: TaskId | None
    task_key: str | None
    message_id: MessageId | None
    title: str
    summary: str
    state: str | None
    parent_event_id: ActivityEventId | None
    source: str | None

    @classmethod
    def from_wire(cls, value: JsonValue) -> ActivityEvent:
        payload = exact_object(
            value,
            label="swarm activity event",
            fields=frozenset(
                {
                    "event_id",
                    "kind",
                    "occurred_at",
                    "actor_id",
                    "task_id",
                    "task_key",
                    "message_id",
                    "title",
                    "summary",
                    "event_state",
                    "parent_event_id",
                    "source",
                }
            ),
        )
        actor_id = optional_text(payload, "actor_id")
        task_id = optional_text(payload, "task_id")
        message_id = optional_text(payload, "message_id")
        parent_event_id = optional_text(payload, "parent_event_id")
        return cls(
            event_id=ActivityEventId(required_text(payload, "event_id")),
            kind=required_text(payload, "kind"),
            occurred_at=required_datetime(payload, "occurred_at"),
            actor_id=ActorId(actor_id) if actor_id else None,
            task_id=TaskId(task_id) if task_id else None,
            task_key=optional_text(payload, "task_key"),
            message_id=MessageId(message_id) if message_id else None,
            title=required_text(payload, "title"),
            summary=required_text(payload, "summary"),
            state=optional_text(payload, "event_state"),
            parent_event_id=(ActivityEventId(parent_event_id) if parent_event_id else None),
            source=optional_text(payload, "source"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "occurred_at": self.occurred_at.isoformat(),
            "actor_id": self.actor_id,
            "task_id": self.task_id,
            "task_key": self.task_key,
            "message_id": self.message_id,
            "title": self.title,
            "summary": self.summary,
            "event_state": self.state,
            "parent_event_id": self.parent_event_id,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class ActivityWorkProduct:
    work_product_id: WorkProductId
    kind: WorkProductKind
    title: str
    status: WorkProductStatus
    task_id: TaskId | None
    actor_id: ActorId | None

    @classmethod
    def from_wire(cls, value: JsonValue) -> ActivityWorkProduct:
        payload = exact_object(
            value,
            label="swarm activity WorkProduct",
            fields=frozenset(
                {
                    "work_product_id",
                    "kind",
                    "title",
                    "status",
                    "task_id",
                    "actor_id",
                }
            ),
        )
        task_id = optional_text(payload, "task_id")
        actor_id = optional_text(payload, "actor_id")
        return cls(
            work_product_id=WorkProductId(required_text(payload, "work_product_id")),
            kind=WorkProductKind(required_text(payload, "kind")),
            title=required_text(payload, "title"),
            status=WorkProductStatus(required_text(payload, "status")),
            task_id=TaskId(task_id) if task_id else None,
            actor_id=ActorId(actor_id) if actor_id else None,
        )

    def to_wire(self) -> JsonObject:
        return {
            "work_product_id": self.work_product_id,
            "kind": self.kind.value,
            "title": self.title,
            "status": self.status.value,
            "task_id": self.task_id,
            "actor_id": self.actor_id,
        }


__all__ = [
    "ActivityActor",
    "ActivityActorState",
    "ActivityEvent",
    "ActivityMessage",
    "ActivityTask",
    "ActivityTaskState",
    "ActivityVisibility",
    "ActivityWorkProduct",
]
