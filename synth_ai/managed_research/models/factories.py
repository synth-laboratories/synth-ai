"""Typed Factory and Effort models mirrored from the backend contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from synth_ai.managed_research.models.run_state import (
    _optional_bool,
    _optional_object_dict,
    _optional_string,
    _require_mapping,
    _require_string,
)


class FactoryKind(StrEnum):
    CUSTOMER = "customer"
    INTERNAL = "internal"
    OPEN_RESEARCH = "open_research"


class FactoryLifecycleState(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class EffortStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    WAITING = "waiting"
    BLOCKED = "blocked"
    READY_FOR_REVIEW = "ready_for_review"
    ARCHIVED_REFERENCE = "archived_reference"


class EffortType(StrEnum):
    RESEARCH = "research"
    MAINTENANCE = "maintenance"
    EVAL_FACTORY = "eval_factory"
    OPTIMIZER = "optimizer"
    OPEN_RESEARCH = "open_research"


def _optional_datetime(payload: Mapping[str, object], key: str) -> datetime | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        return datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    raise ValueError(f"{key} must be null, a datetime, or an ISO-8601 string")


def _optional_enum(enum_type: type[StrEnum], value: object, *, field_name: str) -> StrEnum | None:
    if value is None:
        return None
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        return enum_type(normalized)
    raise ValueError(f"{field_name} must be a string when provided")


def _enum_value(value: StrEnum | str | None) -> str | None:
    if value is None:
        return None
    return value.value if isinstance(value, StrEnum) else str(value)


def _optional_payload_mapping(value: Mapping[str, Any] | dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(value)


def _policy_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    to_wire = getattr(value, "to_wire", None)
    if callable(to_wire):
        return dict(to_wire())
    return dict(value)


@dataclass(frozen=True)
class FactoryCreateRequest:
    name: str
    description: str | None = None
    kind: FactoryKind | str = FactoryKind.CUSTOMER
    status: FactoryLifecycleState | str = FactoryLifecycleState.ACTIVE
    budget_policy: BudgetPolicy | dict[str, Any] = field(default_factory=dict)
    cap_policy: CapPolicy | dict[str, Any] = field(default_factory=dict)
    publication_policy: PublicationPolicy | dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "kind": _enum_value(self.kind),
            "status": _enum_value(self.status),
            "budget_policy": _policy_payload(self.budget_policy),
            "cap_policy": _policy_payload(self.cap_policy),
            "publication_policy": _policy_payload(self.publication_policy),
            "metadata": _policy_payload(self.metadata),
        }
        if self.description is not None:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True)
class FactoryPatchRequest:
    name: str | None = None
    description: str | None = None
    status: FactoryLifecycleState | str | None = None
    budget_policy: BudgetPolicy | dict[str, Any] | None = None
    cap_policy: CapPolicy | dict[str, Any] | None = None
    publication_policy: PublicationPolicy | dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in (
            ("name", self.name),
            ("description", self.description),
        ):
            if value is not None:
                payload[key] = value
        if self.status is not None:
            payload["status"] = _enum_value(self.status)
        for key, value in (
            ("budget_policy", self.budget_policy),
            ("cap_policy", self.cap_policy),
            ("publication_policy", self.publication_policy),
            ("metadata", self.metadata),
        ):
            if value is not None:
                payload[key] = _policy_payload(value)
        return payload


@dataclass(frozen=True)
class RecurrencePolicy:
    cadence: str | None = None
    timezone: str | None = None
    max_active_runs: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        for key, value in (
            ("cadence", self.cadence),
            ("timezone", self.timezone),
            ("max_active_runs", self.max_active_runs),
        ):
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class PublicationPolicy:
    visibility: str | None = None
    publish_reports: bool | None = None
    publish_work_products: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        for key, value in (
            ("visibility", self.visibility),
            ("publish_reports", self.publish_reports),
            ("publish_work_products", self.publish_work_products),
        ):
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class BudgetPolicy:
    limit: float | None = None
    currency: str | None = None
    period: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        for key, value in (
            ("limit", self.limit),
            ("currency", self.currency),
            ("period", self.period),
        ):
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class CapPolicy:
    max_active_efforts: int | None = None
    max_active_runs: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        for key, value in (
            ("max_active_efforts", self.max_active_efforts),
            ("max_active_runs", self.max_active_runs),
        ):
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class Factory:
    factory_id: str
    org_id: str
    name: str
    kind: FactoryKind
    status: FactoryLifecycleState
    description: str | None = None
    budget_policy: dict[str, object] = field(default_factory=dict)
    cap_policy: dict[str, object] = field(default_factory=dict)
    publication_policy: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    created_by_user_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> Factory:
        mapping = _require_mapping(payload, label="factory")
        return cls(
            factory_id=_require_string(mapping, "factory_id", label="factory.factory_id"),
            org_id=_require_string(mapping, "org_id", label="factory.org_id"),
            name=_require_string(mapping, "name", label="factory.name"),
            description=_optional_string(mapping, "description"),
            kind=FactoryKind(_require_string(mapping, "kind", label="factory.kind")),
            status=FactoryLifecycleState(
                _require_string(mapping, "status", label="factory.status")
            ),
            budget_policy=_optional_object_dict(
                mapping.get("budget_policy"),
                label="factory.budget_policy",
            ),
            cap_policy=_optional_object_dict(mapping.get("cap_policy"), label="factory.cap_policy"),
            publication_policy=_optional_object_dict(
                mapping.get("publication_policy"),
                label="factory.publication_policy",
            ),
            metadata=_optional_object_dict(mapping.get("metadata"), label="factory.metadata"),
            created_by_user_id=_optional_string(mapping, "created_by_user_id"),
            created_at=_optional_datetime(mapping, "created_at"),
            updated_at=_optional_datetime(mapping, "updated_at"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class EffortCreateRequest:
    factory_id: str
    project_id: str
    name: str
    hypothesis_or_topic: str | None = None
    status: EffortStatus | str = EffortStatus.ACTIVE
    effort_type: EffortType | str = EffortType.RESEARCH
    recurrence_policy: RecurrencePolicy | dict[str, Any] = field(default_factory=dict)
    next_wake_at: datetime | str | None = None
    latest_run_id: str | None = None
    latest_report_id: str | None = None
    latest_work_product_id: str | None = None
    decision_needed: bool = False
    decision_note: str | None = None
    budget_policy: BudgetPolicy | dict[str, Any] = field(default_factory=dict)
    publication_policy: PublicationPolicy | dict[str, Any] = field(default_factory=dict)
    actor_notes: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "factory_id": self.factory_id,
            "project_id": self.project_id,
            "name": self.name,
            "status": _enum_value(self.status),
            "effort_type": _enum_value(self.effort_type),
            "recurrence_policy": _policy_payload(self.recurrence_policy),
            "decision_needed": self.decision_needed,
            "budget_policy": _policy_payload(self.budget_policy),
            "publication_policy": _policy_payload(self.publication_policy),
            "actor_notes": _policy_payload(self.actor_notes),
            "metadata": _policy_payload(self.metadata),
        }
        for key, value in (
            ("hypothesis_or_topic", self.hypothesis_or_topic),
            ("latest_run_id", self.latest_run_id),
            ("latest_report_id", self.latest_report_id),
            ("latest_work_product_id", self.latest_work_product_id),
            ("decision_note", self.decision_note),
        ):
            if value is not None:
                payload[key] = value
        if self.next_wake_at is not None:
            payload["next_wake_at"] = (
                self.next_wake_at.isoformat()
                if isinstance(self.next_wake_at, datetime)
                else self.next_wake_at
            )
        return payload


@dataclass(frozen=True)
class EffortPatchRequest:
    name: str | None = None
    hypothesis_or_topic: str | None = None
    status: EffortStatus | str | None = None
    effort_type: EffortType | str | None = None
    recurrence_policy: RecurrencePolicy | dict[str, Any] | None = None
    next_wake_at: datetime | str | None = None
    latest_run_id: str | None = None
    latest_report_id: str | None = None
    latest_work_product_id: str | None = None
    decision_needed: bool | None = None
    decision_note: str | None = None
    budget_policy: BudgetPolicy | dict[str, Any] | None = None
    publication_policy: PublicationPolicy | dict[str, Any] | None = None
    actor_notes: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in (
            ("name", self.name),
            ("hypothesis_or_topic", self.hypothesis_or_topic),
            ("latest_run_id", self.latest_run_id),
            ("latest_report_id", self.latest_report_id),
            ("latest_work_product_id", self.latest_work_product_id),
            ("decision_needed", self.decision_needed),
            ("decision_note", self.decision_note),
        ):
            if value is not None:
                payload[key] = value
        if self.status is not None:
            payload["status"] = _enum_value(self.status)
        if self.effort_type is not None:
            payload["effort_type"] = _enum_value(self.effort_type)
        for key, value in (
            ("recurrence_policy", self.recurrence_policy),
            ("budget_policy", self.budget_policy),
            ("publication_policy", self.publication_policy),
            ("actor_notes", self.actor_notes),
            ("metadata", self.metadata),
        ):
            if value is not None:
                payload[key] = _policy_payload(value)
        if self.next_wake_at is not None:
            payload["next_wake_at"] = (
                self.next_wake_at.isoformat()
                if isinstance(self.next_wake_at, datetime)
                else self.next_wake_at
            )
        return payload


@dataclass(frozen=True)
class Effort:
    effort_id: str
    org_id: str
    factory_id: str
    project_id: str
    name: str
    status: EffortStatus
    effort_type: EffortType
    hypothesis_or_topic: str | None = None
    recurrence_policy: dict[str, object] = field(default_factory=dict)
    next_wake_at: datetime | None = None
    latest_run_id: str | None = None
    latest_report_id: str | None = None
    latest_work_product_id: str | None = None
    decision_needed: bool = False
    decision_note: str | None = None
    budget_policy: dict[str, object] = field(default_factory=dict)
    publication_policy: dict[str, object] = field(default_factory=dict)
    actor_notes: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    created_by_user_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> Effort:
        mapping = _require_mapping(payload, label="effort")
        return cls(
            effort_id=_require_string(mapping, "effort_id", label="effort.effort_id"),
            org_id=_require_string(mapping, "org_id", label="effort.org_id"),
            factory_id=_require_string(mapping, "factory_id", label="effort.factory_id"),
            project_id=_require_string(mapping, "project_id", label="effort.project_id"),
            name=_require_string(mapping, "name", label="effort.name"),
            hypothesis_or_topic=_optional_string(mapping, "hypothesis_or_topic"),
            status=EffortStatus(_require_string(mapping, "status", label="effort.status")),
            effort_type=EffortType(
                _require_string(mapping, "effort_type", label="effort.effort_type")
            ),
            recurrence_policy=_optional_object_dict(
                mapping.get("recurrence_policy"),
                label="effort.recurrence_policy",
            ),
            next_wake_at=_optional_datetime(mapping, "next_wake_at"),
            latest_run_id=_optional_string(mapping, "latest_run_id"),
            latest_report_id=_optional_string(mapping, "latest_report_id"),
            latest_work_product_id=_optional_string(mapping, "latest_work_product_id"),
            decision_needed=bool(_optional_bool(mapping, "decision_needed")),
            decision_note=_optional_string(mapping, "decision_note"),
            budget_policy=_optional_object_dict(
                mapping.get("budget_policy"),
                label="effort.budget_policy",
            ),
            publication_policy=_optional_object_dict(
                mapping.get("publication_policy"),
                label="effort.publication_policy",
            ),
            actor_notes=_optional_object_dict(
                mapping.get("actor_notes"), label="effort.actor_notes"
            ),
            metadata=_optional_object_dict(mapping.get("metadata"), label="effort.metadata"),
            created_by_user_id=_optional_string(mapping, "created_by_user_id"),
            created_at=_optional_datetime(mapping, "created_at"),
            updated_at=_optional_datetime(mapping, "updated_at"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryProjectSummary:
    project_id: str
    name: str
    archived: bool = False
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryProjectSummary:
        mapping = _require_mapping(payload, label="factory project summary")
        return cls(
            project_id=_require_string(mapping, "project_id", label="project.project_id"),
            name=_require_string(mapping, "name", label="project.name"),
            archived=bool(_optional_bool(mapping, "archived")),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryRunSummary:
    run_id: str
    project_id: str
    public_state: str
    effort_id: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryRunSummary:
        mapping = _require_mapping(payload, label="factory run summary")
        return cls(
            run_id=_require_string(mapping, "run_id", label="run.run_id"),
            project_id=_require_string(mapping, "project_id", label="run.project_id"),
            effort_id=_optional_string(mapping, "effort_id"),
            public_state=_require_string(mapping, "public_state", label="run.public_state"),
            created_at=_optional_datetime(mapping, "created_at"),
            started_at=_optional_datetime(mapping, "started_at"),
            finished_at=_optional_datetime(mapping, "finished_at"),
            updated_at=_optional_datetime(mapping, "updated_at"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryReportSummary:
    report_id: str
    project_id: str
    run_id: str
    mode: str
    title: str
    effort_id: str | None = None
    task_id: str | None = None
    created_at: datetime | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryReportSummary:
        mapping = _require_mapping(payload, label="factory report summary")
        return cls(
            report_id=_require_string(mapping, "report_id", label="report.report_id"),
            project_id=_require_string(mapping, "project_id", label="report.project_id"),
            run_id=_require_string(mapping, "run_id", label="report.run_id"),
            effort_id=_optional_string(mapping, "effort_id"),
            task_id=_optional_string(mapping, "task_id"),
            mode=_require_string(mapping, "mode", label="report.mode"),
            title=_require_string(mapping, "title", label="report.title"),
            created_at=_optional_datetime(mapping, "created_at"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryWorkProductSummary:
    work_product_id: str
    project_id: str
    run_id: str
    kind: str
    title: str
    status: str
    readiness: str
    effort_id: str | None = None
    artifact_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryWorkProductSummary:
        mapping = _require_mapping(payload, label="factory work product summary")
        return cls(
            work_product_id=_require_string(
                mapping,
                "work_product_id",
                label="work_product.work_product_id",
            ),
            project_id=_require_string(mapping, "project_id", label="work_product.project_id"),
            run_id=_require_string(mapping, "run_id", label="work_product.run_id"),
            effort_id=_optional_string(mapping, "effort_id"),
            kind=_require_string(mapping, "kind", label="work_product.kind"),
            title=_require_string(mapping, "title", label="work_product.title"),
            status=_require_string(mapping, "status", label="work_product.status"),
            readiness=_require_string(mapping, "readiness", label="work_product.readiness"),
            artifact_id=_optional_string(mapping, "artifact_id"),
            created_at=_optional_datetime(mapping, "created_at"),
            updated_at=_optional_datetime(mapping, "updated_at"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryStatus:
    factory: Factory
    projects: tuple[FactoryProjectSummary, ...] = ()
    efforts: tuple[Effort, ...] = ()
    efforts_by_status: dict[str, int] = field(default_factory=dict)
    latest_runs: tuple[FactoryRunSummary, ...] = ()
    latest_reports: tuple[FactoryReportSummary, ...] = ()
    latest_work_products: tuple[FactoryWorkProductSummary, ...] = ()
    open_decisions: tuple[Effort, ...] = ()
    paused_or_waiting: tuple[Effort, ...] = ()
    next_wake_at: datetime | None = None
    publication_states: dict[str, object] = field(default_factory=dict)
    costs_limits: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryStatus:
        mapping = _require_mapping(payload, label="factory status")
        efforts_by_status_raw = mapping.get("efforts_by_status")
        if efforts_by_status_raw is not None and not isinstance(efforts_by_status_raw, Mapping):
            raise ValueError("factory status efforts_by_status must be an object")
        efforts_by_status = {
            str(key): int(value) for key, value in dict(efforts_by_status_raw or {}).items()
        }
        return cls(
            factory=Factory.from_wire(mapping.get("factory")),
            projects=tuple(
                FactoryProjectSummary.from_wire(item)
                for item in list(mapping.get("projects") or [])
            ),
            efforts=tuple(Effort.from_wire(item) for item in list(mapping.get("efforts") or [])),
            efforts_by_status=efforts_by_status,
            latest_runs=tuple(
                FactoryRunSummary.from_wire(item) for item in list(mapping.get("latest_runs") or [])
            ),
            latest_reports=tuple(
                FactoryReportSummary.from_wire(item)
                for item in list(mapping.get("latest_reports") or [])
            ),
            latest_work_products=tuple(
                FactoryWorkProductSummary.from_wire(item)
                for item in list(mapping.get("latest_work_products") or [])
            ),
            open_decisions=tuple(
                Effort.from_wire(item) for item in list(mapping.get("open_decisions") or [])
            ),
            paused_or_waiting=tuple(
                Effort.from_wire(item) for item in list(mapping.get("paused_or_waiting") or [])
            ),
            next_wake_at=_optional_datetime(mapping, "next_wake_at"),
            publication_states=_optional_object_dict(
                mapping.get("publication_states"),
                label="factory status publication_states",
            ),
            costs_limits=_optional_object_dict(
                mapping.get("costs_limits"),
                label="factory status costs_limits",
            ),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryWakeDueRequest:
    launch_request: dict[str, Any] | None = None
    limit: int = 10
    allow_overlap: bool = False
    dry_run: bool = False
    continue_on_error: bool = True

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "limit": self.limit,
            "allow_overlap": self.allow_overlap,
            "dry_run": self.dry_run,
            "continue_on_error": self.continue_on_error,
        }
        if self.launch_request is not None:
            payload["launch_request"] = dict(self.launch_request)
        return payload


@dataclass(frozen=True)
class FactoryWakeDueEffort:
    effort_id: str
    project_id: str
    status: str
    reason: str | None = None
    run_id: str | None = None
    next_wake_at: datetime | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryWakeDueEffort:
        mapping = _require_mapping(payload, label="factory wake due effort")
        return cls(
            effort_id=_require_string(mapping, "effort_id", label="wake.effort_id"),
            project_id=_require_string(mapping, "project_id", label="wake.project_id"),
            status=_require_string(mapping, "status", label="wake.status"),
            reason=_optional_string(mapping, "reason"),
            run_id=_optional_string(mapping, "run_id"),
            next_wake_at=_optional_datetime(mapping, "next_wake_at"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryWakeDueResult:
    factory_id: str
    evaluated_at: datetime | None
    dry_run: bool
    launched: int
    skipped: int
    failed: int
    efforts: tuple[FactoryWakeDueEffort, ...] = ()
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryWakeDueResult:
        mapping = _require_mapping(payload, label="factory wake due result")
        return cls(
            factory_id=_require_string(mapping, "factory_id", label="wake.factory_id"),
            evaluated_at=_optional_datetime(mapping, "evaluated_at"),
            dry_run=bool(_optional_bool(mapping, "dry_run")),
            launched=int(mapping.get("launched") or 0),
            skipped=int(mapping.get("skipped") or 0),
            failed=int(mapping.get("failed") or 0),
            efforts=tuple(
                FactoryWakeDueEffort.from_wire(item)
                for item in list(mapping.get("efforts") or [])
            ),
            raw=dict(mapping),
        )


def factory_create_payload(
    request: FactoryCreateRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, FactoryCreateRequest):
        return request.to_wire()
    return dict(request)


def factory_patch_payload(
    request: FactoryPatchRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, FactoryPatchRequest):
        return request.to_wire()
    return dict(request)


def factory_wake_due_payload(
    request: FactoryWakeDueRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, FactoryWakeDueRequest):
        return request.to_wire()
    return dict(request)


def effort_create_payload(
    request: EffortCreateRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, EffortCreateRequest):
        return request.to_wire()
    return dict(request)


def effort_patch_payload(
    request: EffortPatchRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, EffortPatchRequest):
        return request.to_wire()
    return dict(request)


FACTORY_KIND_VALUES = tuple(item.value for item in FactoryKind)
FACTORY_LIFECYCLE_STATE_VALUES = tuple(item.value for item in FactoryLifecycleState)
EFFORT_STATUS_VALUES = tuple(item.value for item in EffortStatus)
EFFORT_TYPE_VALUES = tuple(item.value for item in EffortType)


__all__ = [
    "EFFORT_STATUS_VALUES",
    "EFFORT_TYPE_VALUES",
    "FACTORY_KIND_VALUES",
    "FACTORY_LIFECYCLE_STATE_VALUES",
    "Effort",
    "EffortCreateRequest",
    "EffortPatchRequest",
    "EffortStatus",
    "EffortType",
    "BudgetPolicy",
    "CapPolicy",
    "Factory",
    "FactoryCreateRequest",
    "FactoryKind",
    "FactoryLifecycleState",
    "FactoryPatchRequest",
    "FactoryProjectSummary",
    "FactoryReportSummary",
    "FactoryRunSummary",
    "FactoryStatus",
    "FactoryWakeDueEffort",
    "FactoryWakeDueRequest",
    "FactoryWakeDueResult",
    "FactoryWorkProductSummary",
    "PublicationPolicy",
    "RecurrencePolicy",
    "effort_create_payload",
    "effort_patch_payload",
    "factory_create_payload",
    "factory_patch_payload",
    "factory_wake_due_payload",
]
