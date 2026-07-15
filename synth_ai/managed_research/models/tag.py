"""Typed Synth Tag SDK models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, cast


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


class TagArtifactKind(StrEnum):
    RUN_RECORD = "run_record"
    RUN_SUMMARY = "run_summary"
    WORK_PRODUCT = "work_product"


class TagChampionEventAction(StrEnum):
    PROMOTE = "promote"
    NO_LIFT = "no_lift"
    ROLLBACK = "rollback"


def _mapping(payload: object, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(cast(Mapping[str, Any], payload))


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


def _optional_datetime(payload: Mapping[str, Any], key: str) -> datetime | None:
    if payload.get(key) is None:
        return None
    return _datetime(payload, key)


def _optional_float(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    return float(value) if value is not None else None


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

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | dict[str, Any]) -> TagScope:
        return cls.from_wire(payload)


@dataclass(frozen=True)
class TagSteeringWikiLink:
    changeset_id: str
    proposal_id: str
    url: str

    @classmethod
    def from_wire(cls, payload: object) -> TagSteeringWikiLink:
        data = _mapping(payload, label="TagSteeringWikiLink")
        return cls(
            changeset_id=_required_string(data, "changeset_id"),
            proposal_id=_required_string(data, "proposal_id"),
            url=_required_string(data, "url"),
        )

    @classmethod
    def from_payload(cls, payload: object) -> TagSteeringWikiLink:
        return cls.from_wire(payload)


@dataclass(frozen=True)
class TagSteeringReceipt:
    schema_version: str = "tag_steering_receipt.v1"
    receipt_id: str | None = None
    session_id: str | None = None
    org_id: str | None = None
    factory_id: str | None = None
    effort_id: str | None = None
    project_id: str | None = None
    experiment_id: str | None = None
    candidate_id: str | None = None
    run_id: str | None = None
    steering_target: TagSteeringTarget | None = None
    message: str | None = None
    idempotency_key: str | None = None
    transport_ref: dict[str, Any] = field(default_factory=dict)
    accepted_at: datetime | None = None
    wiki: TagSteeringWikiLink | None = None

    @classmethod
    def from_wire(cls, payload: object) -> TagSteeringReceipt:
        data = _mapping(payload, label="TagSteeringReceipt")
        steering_target = _optional_string(data, "steering_target")
        wiki = data.get("wiki")
        return cls(
            schema_version=str(data.get("schema_version") or "tag_steering_receipt.v1"),
            receipt_id=_optional_string(data, "receipt_id"),
            session_id=_optional_string(data, "session_id"),
            org_id=_optional_string(data, "org_id"),
            factory_id=_optional_string(data, "factory_id"),
            effort_id=_optional_string(data, "effort_id"),
            project_id=_optional_string(data, "project_id"),
            experiment_id=_optional_string(data, "experiment_id"),
            candidate_id=_optional_string(data, "candidate_id"),
            run_id=_optional_string(data, "run_id"),
            steering_target=(TagSteeringTarget(steering_target) if steering_target else None),
            message=_optional_string(data, "message"),
            idempotency_key=_optional_string(data, "idempotency_key"),
            transport_ref=dict(data.get("transport_ref") or {}),
            accepted_at=_optional_datetime(data, "accepted_at"),
            wiki=TagSteeringWikiLink.from_wire(wiki) if wiki is not None else None,
        )

    @classmethod
    def from_payload(cls, payload: object) -> TagSteeringReceipt:
        return cls.from_wire(payload)


@dataclass(frozen=True)
class TagArtifactLink:
    kind: TagArtifactKind
    url: str
    schema_version: str = "tag_artifact_link.v1"
    artifact_id: str | None = None
    work_product_id: str | None = None
    title: str | None = None
    status: str | None = None
    readiness: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> TagArtifactLink:
        data = _mapping(payload, label="TagArtifactLink")
        return cls(
            schema_version=str(data.get("schema_version") or "tag_artifact_link.v1"),
            kind=TagArtifactKind(_required_string(data, "kind")),
            url=_required_string(data, "url"),
            artifact_id=_optional_string(data, "artifact_id"),
            work_product_id=_optional_string(data, "work_product_id"),
            title=_optional_string(data, "title"),
            status=_optional_string(data, "status"),
            readiness=_optional_string(data, "readiness"),
        )

    @classmethod
    def from_payload(cls, payload: object) -> TagArtifactLink:
        return cls.from_wire(payload)


@dataclass(frozen=True)
class TagSessionReceipt:
    state: str
    schema_version: str = "tag_receipt.v1"
    run_id: str | None = None
    run_url: str | None = None
    artifact_urls: list[str] = field(default_factory=list)
    artifacts: list[TagArtifactLink] = field(default_factory=list)
    artifact_empty_reason: str | None = None
    terminal_outcome: str | None = None
    project_url: str | None = None
    factory_url: str | None = None
    effort_url: str | None = None
    experiment_url: str | None = None
    wiki_urls: list[str] = field(default_factory=list)
    git_urls: list[str] = field(default_factory=list)
    steering_receipts: list[TagSteeringReceipt] = field(default_factory=list)

    @classmethod
    def from_wire(
        cls,
        payload: Mapping[str, Any] | dict[str, Any] | None,
    ) -> TagSessionReceipt:
        data = dict(payload or {})
        return cls(
            state=str(data.get("state") or "queued"),
            schema_version=str(data.get("schema_version") or "tag_receipt.v1"),
            run_id=_optional_string(data, "run_id"),
            run_url=_optional_string(data, "run_url"),
            artifact_urls=[str(item) for item in data.get("artifact_urls") or []],
            artifacts=[TagArtifactLink.from_wire(item) for item in data.get("artifacts") or []],
            artifact_empty_reason=_optional_string(data, "artifact_empty_reason"),
            terminal_outcome=_optional_string(data, "terminal_outcome"),
            project_url=_optional_string(data, "project_url"),
            factory_url=_optional_string(data, "factory_url"),
            effort_url=_optional_string(data, "effort_url"),
            experiment_url=_optional_string(data, "experiment_url"),
            wiki_urls=[str(item) for item in data.get("wiki_urls") or []],
            git_urls=[str(item) for item in data.get("git_urls") or []],
            steering_receipts=[
                TagSteeringReceipt.from_wire(item) for item in data.get("steering_receipts") or []
            ],
        )

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any] | dict[str, Any] | None,
    ) -> TagSessionReceipt:
        return cls.from_wire(payload)


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
    graduation_proposal: dict[str, Any] | None = None
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
            graduation_proposal=_optional_mapping(data, "graduation_proposal"),
            definition_of_done=_optional_string(data, "definition_of_done"),
            metadata=dict(data.get("metadata") or {}),
            created_at=_datetime(data, "created_at"),
            updated_at=_datetime(data, "updated_at"),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | dict[str, Any]) -> TagSession:
        return cls.from_wire(payload)


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

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | dict[str, Any]) -> TagTask:
        return cls.from_wire(payload)


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

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | dict[str, Any]) -> TagSessionWatch:
        return cls.from_wire(payload)


@dataclass(frozen=True)
class TagFactoryChampion:
    candidate_id: str
    git_sha: str
    work_product_id: str | None = None
    held_out_score: float | None = None
    selected_at: datetime | None = None

    @classmethod
    def from_wire(cls, payload: object) -> TagFactoryChampion:
        data = _mapping(payload, label="TagFactoryChampion")
        return cls(
            candidate_id=_required_string(data, "candidate_id"),
            git_sha=_required_string(data, "git_sha"),
            work_product_id=_optional_string(data, "work_product_id"),
            held_out_score=_optional_float(data, "held_out_score"),
            selected_at=_optional_datetime(data, "selected_at"),
        )

    @classmethod
    def from_payload(cls, payload: object) -> TagFactoryChampion:
        return cls.from_wire(payload)


@dataclass(frozen=True)
class TagFactoryChampionEvent:
    action: TagChampionEventAction
    at: datetime
    candidate_id: str | None = None
    git_sha: str | None = None
    held_out_score: float | None = None
    reason: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> TagFactoryChampionEvent:
        data = _mapping(payload, label="TagFactoryChampionEvent")
        return cls(
            action=TagChampionEventAction(_required_string(data, "action")),
            at=_datetime(data, "at"),
            candidate_id=_optional_string(data, "candidate_id"),
            git_sha=_optional_string(data, "git_sha"),
            held_out_score=_optional_float(data, "held_out_score"),
            reason=_optional_string(data, "reason"),
        )

    @classmethod
    def from_payload(cls, payload: object) -> TagFactoryChampionEvent:
        return cls.from_wire(payload)


@dataclass(frozen=True)
class TagFactoryCandidateCounts:
    total: int
    graded: int
    pending: int

    @classmethod
    def from_wire(cls, payload: object) -> TagFactoryCandidateCounts:
        data = _mapping(payload, label="TagFactoryCandidateCounts")
        return cls(
            total=int(data.get("total") or 0),
            graded=int(data.get("graded") or 0),
            pending=int(data.get("pending") or 0),
        )

    @classmethod
    def from_payload(cls, payload: object) -> TagFactoryCandidateCounts:
        return cls.from_wire(payload)


@dataclass(frozen=True)
class TagFactoryContext:
    factory_id: str
    candidates: TagFactoryCandidateCounts
    as_of: datetime
    schema_version: str = "tag_factory_context.v1"
    effort_id: str | None = None
    experiment_ref: str | None = None
    champion: TagFactoryChampion | None = None
    last_champion_event: TagFactoryChampionEvent | None = None

    @classmethod
    def from_wire(cls, payload: object) -> TagFactoryContext:
        data = _mapping(payload, label="TagFactoryContext")
        champion = data.get("champion")
        event = data.get("last_champion_event")
        return cls(
            schema_version=str(data.get("schema_version") or "tag_factory_context.v1"),
            factory_id=_required_string(data, "factory_id"),
            effort_id=_optional_string(data, "effort_id"),
            experiment_ref=_optional_string(data, "experiment_ref"),
            champion=(TagFactoryChampion.from_wire(champion) if champion is not None else None),
            last_champion_event=(
                TagFactoryChampionEvent.from_wire(event) if event is not None else None
            ),
            candidates=TagFactoryCandidateCounts.from_wire(data.get("candidates") or {}),
            as_of=_datetime(data, "as_of"),
        )

    @classmethod
    def from_payload(cls, payload: object) -> TagFactoryContext:
        return cls.from_wire(payload)


TagSessionReceiptResponse = TagSessionReceipt

__all__ = [
    "TagArtifactKind",
    "TagArtifactLink",
    "TagChampionEventAction",
    "TagFactoryCandidateCounts",
    "TagFactoryChampion",
    "TagFactoryChampionEvent",
    "TagFactoryContext",
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
    "TagSteeringReceipt",
    "TagSteeringWikiLink",
    "TagTask",
]
