"""Typed Synth Tag SDK models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class TagSessionStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"
    ARCHIVED = "archived"


class TagSteeringTarget(StrEnum):
    ACTIVE_RUN = "active_run"
    NEXT_CYCLE = "next_cycle"


class TagSessionControlAction(StrEnum):
    PAUSE = "pause"
    STOP = "stop"
    ARCHIVE = "archive"
    RECONNECT = "reconnect"


class TagSessionKind(StrEnum):
    RESEARCH = "research"
    REVIEW = "review"
    MAINTENANCE = "maintenance"
    FACTORY_OP = "factory_op"


def _mapping(payload: Mapping[str, Any] | dict[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(payload)


def _optional_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object when provided")
    return dict(value)


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _datetime(payload: Mapping[str, Any], key: str) -> datetime:
    value = payload.get(key)
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise ValueError(f"{key} must be an ISO-8601 datetime")


@dataclass(frozen=True)
class TagSessionCreateRequest:
    request: str
    factory_id: str
    effort_id: str
    definition_of_done: str | None = None
    scope_id: str | None = None
    session_kind: TagSessionKind | str = TagSessionKind.RESEARCH
    experiment_id: str | None = None
    candidate_id: str | None = None
    conversation_ref: dict[str, Any] | None = None
    timebox_seconds: int | None = None
    runbook_preset: str | None = None
    host_kind: str | None = None
    worker_pool_id: str | None = None
    local_execution: dict[str, Any] | None = None
    execution_profile: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "request": self.request,
            "session_kind": getattr(self.session_kind, "value", self.session_kind),
        }
        for key, value in (
            ("definition_of_done", self.definition_of_done),
            ("scope_id", self.scope_id),
            ("factory_id", self.factory_id),
            ("effort_id", self.effort_id),
            ("experiment_id", self.experiment_id),
            ("candidate_id", self.candidate_id),
            ("conversation_ref", self.conversation_ref),
            ("timebox_seconds", self.timebox_seconds),
            ("runbook_preset", self.runbook_preset),
            ("host_kind", self.host_kind),
            ("worker_pool_id", self.worker_pool_id),
            ("local_execution", self.local_execution),
            ("execution_profile", self.execution_profile),
            ("metadata", self.metadata),
        ):
            if value not in (None, {}, []):
                payload[key] = value
        return payload


@dataclass(frozen=True)
class TagMessageRequest:
    message: str
    steering_target: TagSteeringTarget | str = TagSteeringTarget.ACTIVE_RUN
    metadata: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "message": self.message,
            "steering_target": getattr(self.steering_target, "value", self.steering_target),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        if self.idempotency_key:
            payload["idempotency_key"] = self.idempotency_key
        return payload


@dataclass(frozen=True)
class TagScope:
    scope_id: str
    org_id: str
    name: str
    status: str
    is_default: bool
    factory_id: str | None
    default_project_id: str | None
    default_launch_profile: dict[str, Any]
    policy: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any] | dict[str, Any]) -> TagScope:
        data = _mapping(payload, label="TagScope")
        return cls(
            scope_id=_required_string(data, "scope_id"),
            org_id=_required_string(data, "org_id"),
            name=_required_string(data, "name"),
            status=_required_string(data, "status"),
            is_default=bool(data.get("is_default")),
            factory_id=_optional_string(data, "factory_id"),
            default_project_id=_optional_string(data, "default_project_id"),
            default_launch_profile=dict(data.get("default_launch_profile") or {}),
            policy=dict(data.get("policy") or {}),
            metadata=dict(data.get("metadata") or {}),
            created_at=_datetime(data, "created_at"),
            updated_at=_datetime(data, "updated_at"),
        )


@dataclass(frozen=True)
class TagSessionReceipt:
    state: str
    run_id: str | None = None
    run_url: str | None = None
    artifact_urls: list[str] = field(default_factory=list)
    artifact_empty_reason: str | None = None
    terminal_outcome: str | None = None
    project_url: str | None = None
    factory_url: str | None = None
    effort_url: str | None = None
    experiment_url: str | None = None
    wiki_urls: list[str] = field(default_factory=list)
    git_urls: list[str] = field(default_factory=list)
    steering_receipts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_wire(
        cls,
        payload: Mapping[str, Any] | dict[str, Any] | None,
    ) -> TagSessionReceipt:
        data = dict(payload or {})
        return cls(
            state=str(data.get("state") or "queued"),
            run_id=_optional_string(data, "run_id"),
            run_url=_optional_string(data, "run_url"),
            artifact_urls=[str(item) for item in data.get("artifact_urls") or []],
            artifact_empty_reason=_optional_string(data, "artifact_empty_reason"),
            terminal_outcome=_optional_string(data, "terminal_outcome"),
            project_url=_optional_string(data, "project_url"),
            factory_url=_optional_string(data, "factory_url"),
            effort_url=_optional_string(data, "effort_url"),
            experiment_url=_optional_string(data, "experiment_url"),
            wiki_urls=[str(item) for item in data.get("wiki_urls") or []],
            git_urls=[str(item) for item in data.get("git_urls") or []],
            steering_receipts=[
                dict(item)
                for item in data.get("steering_receipts") or []
                if isinstance(item, Mapping)
            ],
        )


@dataclass(frozen=True)
class TagSession:
    session_id: str
    org_id: str
    scope_id: str
    status: TagSessionStatus
    session_kind: TagSessionKind
    request: str
    receipt: TagSessionReceipt
    project_id: str | None = None
    run_id: str | None = None
    factory_id: str | None = None
    effort_id: str | None = None
    experiment_id: str | None = None
    candidate_id: str | None = None
    conversation_ref: dict[str, Any] | None = None
    definition_of_done: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any] | dict[str, Any]) -> TagSession:
        data = _mapping(payload, label="TagSession")
        return cls(
            session_id=_required_string(data, "session_id"),
            org_id=_required_string(data, "org_id"),
            scope_id=_required_string(data, "scope_id"),
            status=TagSessionStatus(_required_string(data, "status")),
            session_kind=TagSessionKind(_required_string(data, "session_kind")),
            request=_required_string(data, "request"),
            receipt=TagSessionReceipt.from_wire(data.get("receipt")),
            project_id=_optional_string(data, "project_id"),
            run_id=_optional_string(data, "run_id"),
            factory_id=_optional_string(data, "factory_id"),
            effort_id=_optional_string(data, "effort_id"),
            experiment_id=_optional_string(data, "experiment_id"),
            candidate_id=_optional_string(data, "candidate_id"),
            conversation_ref=_optional_mapping(data, "conversation_ref"),
            definition_of_done=_optional_string(data, "definition_of_done"),
            metadata=dict(data.get("metadata") or {}),
            created_at=_datetime(data, "created_at"),
            updated_at=_datetime(data, "updated_at"),
        )


@dataclass(frozen=True)
class TagTask:
    task_id: str
    session_id: str
    task_kind: str
    body: str
    run_id: str | None = None
    steering_target: TagSteeringTarget | None = None
    definition_of_done: str | None = None
    transport_ref: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any] | dict[str, Any]) -> TagTask:
        data = _mapping(payload, label="TagTask")
        steering_target = _optional_string(data, "steering_target")
        return cls(
            task_id=_required_string(data, "task_id"),
            session_id=_required_string(data, "session_id"),
            task_kind=_required_string(data, "task_kind"),
            body=_required_string(data, "body"),
            run_id=_optional_string(data, "run_id"),
            steering_target=(TagSteeringTarget(steering_target) if steering_target else None),
            definition_of_done=_optional_string(data, "definition_of_done"),
            transport_ref=dict(data.get("transport_ref") or {}),
            metadata=dict(data.get("metadata") or {}),
            created_at=_datetime(data, "created_at"),
        )


@dataclass(frozen=True)
class TagSessionWatch:
    session: TagSession
    messages: tuple[TagTask, ...] = ()

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any] | dict[str, Any]) -> TagSessionWatch:
        data = _mapping(payload, label="TagSessionWatch")
        return cls(
            session=TagSession.from_wire(data.get("session") or {}),
            messages=tuple(TagTask.from_wire(item) for item in data.get("messages") or []),
        )


TagSessionReceiptResponse = TagSessionReceipt

__all__ = [
    "TagMessageRequest",
    "TagScope",
    "TagSession",
    "TagSessionControlAction",
    "TagSessionCreateRequest",
    "TagSessionKind",
    "TagSessionReceipt",
    "TagSessionReceiptResponse",
    "TagSessionStatus",
    "TagSessionWatch",
    "TagSteeringTarget",
    "TagTask",
]
