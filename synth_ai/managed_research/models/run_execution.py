"""Typed Managed Research run execution projection models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime


def _require_mapping(payload: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return payload


def _optional_text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string when provided")
    text = value.strip()
    return text or None


def _require_text(payload: Mapping[str, object], key: str, *, label: str) -> str:
    value = _optional_text(payload, key)
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _require_public_task_state(payload: Mapping[str, object], *, label: str) -> str:
    value = _optional_text(payload, "public_task_state") or _optional_text(payload, "task_state")
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _optional_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer when provided")
    return value


def _require_datetime(payload: Mapping[str, object], key: str, *, label: str) -> datetime:
    value = payload.get(key)
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _optional_datetime(payload: Mapping[str, object], key: str) -> datetime | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{key} must be an ISO-8601 datetime when provided")
    text = value.strip()
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _text_list(payload: Mapping[str, object], key: str) -> list[str]:
    value = payload.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array when provided")
    return [str(item) for item in value if str(item).strip()]


def _object_list(payload: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    value = payload.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array when provided")
    return [_require_mapping(item, label=key) for item in value]


@dataclass(frozen=True, slots=True)
class RunExecutionCursor:
    latest_node_id: str | None = None
    latest_event_seq: int | None = None
    latest_runtime_message_seq: int | None = None
    transcript_cursor: str | None = None
    generated_at: datetime | None = None

    @classmethod
    def from_wire(cls, payload: object) -> RunExecutionCursor:
        mapping = _require_mapping(payload, label="run execution cursor")
        return cls(
            latest_node_id=_optional_text(mapping, "latest_node_id"),
            latest_event_seq=_optional_int(mapping, "latest_event_seq"),
            latest_runtime_message_seq=_optional_int(
                mapping,
                "latest_runtime_message_seq",
            ),
            transcript_cursor=_optional_text(mapping, "transcript_cursor"),
            generated_at=_optional_datetime(mapping, "generated_at"),
        )


@dataclass(frozen=True, slots=True)
class RunExecutionRun:
    public_state: str
    liveness_phase: str | None = None
    terminal_outcome: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    latest_summary: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> RunExecutionRun:
        mapping = _require_mapping(payload, label="run execution run")
        return cls(
            public_state=_require_text(mapping, "public_state", label="run.public_state"),
            liveness_phase=_optional_text(mapping, "liveness_phase"),
            terminal_outcome=_optional_text(mapping, "terminal_outcome"),
            started_at=_optional_datetime(mapping, "started_at"),
            completed_at=_optional_datetime(mapping, "completed_at"),
            latest_summary=_optional_text(mapping, "latest_summary"),
        )


@dataclass(frozen=True, slots=True)
class RunExecutionActor:
    actor_id: str
    role: str
    label: str
    state: str
    task_ids: list[str] = field(default_factory=list)
    task_keys: list[str] = field(default_factory=list)
    spawned_at: datetime | None = None
    completed_at: datetime | None = None
    latest_event_at: datetime | None = None
    latest_summary: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> RunExecutionActor:
        mapping = _require_mapping(payload, label="run execution actor")
        return cls(
            actor_id=_require_text(mapping, "actor_id", label="actor.actor_id"),
            role=_require_text(mapping, "role", label="actor.role"),
            label=_require_text(mapping, "label", label="actor.label"),
            state=_require_text(mapping, "state", label="actor.state"),
            task_ids=_text_list(mapping, "task_ids"),
            task_keys=_text_list(mapping, "task_keys"),
            spawned_at=_optional_datetime(mapping, "spawned_at"),
            completed_at=_optional_datetime(mapping, "completed_at"),
            latest_event_at=_optional_datetime(mapping, "latest_event_at"),
            latest_summary=_optional_text(mapping, "latest_summary"),
        )


@dataclass(frozen=True, slots=True)
class RunExecutionTask:
    task_id: str
    task_key: str
    title: str
    task_state: str
    public_task_state: str
    objective_id: str | None = None
    owner_actor_id: str | None = None
    created_by_actor_id: str | None = None
    updated_by_actor_id: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    latest_summary: str | None = None
    work_product_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> RunExecutionTask:
        mapping = _require_mapping(payload, label="run execution task")
        public_task_state = _require_public_task_state(mapping, label="task.public_task_state")
        return cls(
            task_id=_require_text(mapping, "task_id", label="task.task_id"),
            task_key=_require_text(mapping, "task_key", label="task.task_key"),
            title=_require_text(mapping, "title", label="task.title"),
            task_state=public_task_state,
            public_task_state=public_task_state,
            objective_id=_optional_text(mapping, "objective_id"),
            owner_actor_id=_optional_text(mapping, "owner_actor_id"),
            created_by_actor_id=_optional_text(mapping, "created_by_actor_id"),
            updated_by_actor_id=_optional_text(mapping, "updated_by_actor_id"),
            created_at=_optional_datetime(mapping, "created_at"),
            started_at=_optional_datetime(mapping, "started_at"),
            completed_at=_optional_datetime(mapping, "completed_at"),
            latest_summary=_optional_text(mapping, "latest_summary"),
            work_product_refs=_text_list(mapping, "work_product_refs"),
        )


@dataclass(frozen=True, slots=True)
class RunExecutionMessage:
    message_id: str
    occurred_at: datetime
    title: str
    body: str
    event_id: str | None = None
    actor_id: str | None = None
    participant_role: str | None = None
    task_id: str | None = None
    task_key: str | None = None
    visibility: str = "public"

    @classmethod
    def from_wire(cls, payload: object) -> RunExecutionMessage:
        mapping = _require_mapping(payload, label="run execution message")
        return cls(
            message_id=_require_text(mapping, "message_id", label="message.message_id"),
            event_id=_optional_text(mapping, "event_id"),
            actor_id=_optional_text(mapping, "actor_id"),
            participant_role=_optional_text(mapping, "participant_role"),
            task_id=_optional_text(mapping, "task_id"),
            task_key=_optional_text(mapping, "task_key"),
            occurred_at=_require_datetime(
                mapping,
                "occurred_at",
                label="message.occurred_at",
            ),
            title=_require_text(mapping, "title", label="message.title"),
            body=_require_text(mapping, "body", label="message.body"),
            visibility=_optional_text(mapping, "visibility") or "public",
        )


@dataclass(frozen=True, slots=True)
class RunExecutionEvent:
    event_id: str
    kind: str
    occurred_at: datetime
    title: str
    summary: str
    actor_id: str | None = None
    task_id: str | None = None
    task_key: str | None = None
    message_id: str | None = None
    event_state: str | None = None
    parent_event_id: str | None = None
    source: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> RunExecutionEvent:
        mapping = _require_mapping(payload, label="run execution event")
        return cls(
            event_id=_require_text(mapping, "event_id", label="event.event_id"),
            kind=_require_text(mapping, "kind", label="event.kind"),
            occurred_at=_require_datetime(
                mapping,
                "occurred_at",
                label="event.occurred_at",
            ),
            actor_id=_optional_text(mapping, "actor_id"),
            task_id=_optional_text(mapping, "task_id"),
            task_key=_optional_text(mapping, "task_key"),
            message_id=_optional_text(mapping, "message_id"),
            title=_require_text(mapping, "title", label="event.title"),
            summary=_require_text(mapping, "summary", label="event.summary"),
            event_state=_optional_text(mapping, "event_state"),
            parent_event_id=_optional_text(mapping, "parent_event_id"),
            source=_optional_text(mapping, "source"),
        )


@dataclass(frozen=True, slots=True)
class RunExecutionWorkProductRef:
    work_product_id: str
    kind: str
    title: str
    status: str
    task_id: str | None = None
    actor_id: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> RunExecutionWorkProductRef:
        mapping = _require_mapping(payload, label="run execution work product")
        return cls(
            work_product_id=_require_text(
                mapping,
                "work_product_id",
                label="work_product.work_product_id",
            ),
            kind=_require_text(mapping, "kind", label="work_product.kind"),
            title=_require_text(mapping, "title", label="work_product.title"),
            status=_require_text(mapping, "status", label="work_product.status"),
            task_id=_optional_text(mapping, "task_id"),
            actor_id=_optional_text(mapping, "actor_id"),
        )


@dataclass(frozen=True, slots=True)
class RunExecutionProjection:
    schema_version: str
    project_id: str
    run_id: str
    generated_at: datetime
    run: RunExecutionRun
    cursor: RunExecutionCursor
    actors: list[RunExecutionActor] = field(default_factory=list)
    tasks: list[RunExecutionTask] = field(default_factory=list)
    messages: list[RunExecutionMessage] = field(default_factory=list)
    events: list[RunExecutionEvent] = field(default_factory=list)
    work_products: list[RunExecutionWorkProductRef] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> RunExecutionProjection:
        mapping = _require_mapping(payload, label="run execution projection")
        return cls(
            schema_version=_require_text(
                mapping,
                "schema_version",
                label="projection.schema_version",
            ),
            project_id=_require_text(mapping, "project_id", label="projection.project_id"),
            run_id=_require_text(mapping, "run_id", label="projection.run_id"),
            generated_at=_require_datetime(
                mapping,
                "generated_at",
                label="projection.generated_at",
            ),
            run=RunExecutionRun.from_wire(mapping.get("run")),
            cursor=RunExecutionCursor.from_wire(mapping.get("cursor")),
            actors=[RunExecutionActor.from_wire(item) for item in _object_list(mapping, "actors")],
            tasks=[RunExecutionTask.from_wire(item) for item in _object_list(mapping, "tasks")],
            messages=[
                RunExecutionMessage.from_wire(item) for item in _object_list(mapping, "messages")
            ],
            events=[RunExecutionEvent.from_wire(item) for item in _object_list(mapping, "events")],
            work_products=[
                RunExecutionWorkProductRef.from_wire(item)
                for item in _object_list(mapping, "work_products")
            ],
        )


__all__ = [
    "RunExecutionActor",
    "RunExecutionCursor",
    "RunExecutionEvent",
    "RunExecutionMessage",
    "RunExecutionProjection",
    "RunExecutionRun",
    "RunExecutionTask",
    "RunExecutionWorkProductRef",
]
