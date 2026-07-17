"""Typed Factory and Effort models mirrored from the backend contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from synth_ai.managed_research.models.run_state import (
    _int_value,
    _optional_bool,
    _optional_object_dict,
    _optional_object_tuple,
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


class FactoryProjectRole(StrEnum):
    CANONICAL = "canonical"
    AUXILIARY = "auxiliary"
    ARCHIVED_REFERENCE = "archived_reference"


class FactoryProjectStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class FactoryIdeaStatus(StrEnum):
    OPEN = "open"
    PROMOTED = "promoted"
    PAUSED = "paused"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class FactoryIdeaSource(StrEnum):
    HUMAN = "human"
    SERAPH = "seraph"
    GARDENER = "gardener"
    ARCHITECT = "architect"
    WORKER = "worker"
    RUN = "run"
    EXTERNAL = "external"


class FactoryActorRole(StrEnum):
    ORCHESTRATOR = "orchestrator"
    SERAPH = "seraph"
    GARDENER = "gardener"
    ARCHITECT = "architect"
    WORKER = "worker"
    REVIEWER = "reviewer"


class FactoryActorOutputKind(StrEnum):
    SERAPH_BRIEF = "seraph_brief"
    GARDENER_DIGEST = "gardener_digest"
    ARCHITECT_FEED_HEALTH = "architect_feed_health"
    FAILURE_TAXONOMY = "failure_taxonomy"
    FINDING_REPORT = "finding_report"
    DECISION_BRIEF = "decision_brief"
    SUCCESS_MEASUREMENT_CARD = "success_measurement_card"


class FactoryActorOutputStatus(StrEnum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    ACCEPTED = "accepted"
    PUBLISHED = "published"
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
    EVAL_FACTORY = "eval_factory"
    OPTIMIZER = "optimizer"
    OPEN_RESEARCH = "open_research"


class FactoryRunKind(StrEnum):
    RESEARCH = "research"
    MAINTENANCE = "maintenance"


class FactoryCandidateGradingStatus(StrEnum):
    PENDING = "pending"
    GRADED = "graded"
    FAILED = "failed"
    TIMEOUT = "timeout"


class FactoryChampionEventAction(StrEnum):
    PROMOTE = "promote"
    ROLLBACK = "rollback"
    NO_LIFT = "no_lift"


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


def _optional_float(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a number when provided")
    return float(value)


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


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    raise ValueError("value must be a list when provided")


@dataclass(frozen=True)
class FactoryCreateRequest:
    name: str
    description: str | None = None
    kind: FactoryKind | str = FactoryKind.CUSTOMER
    status: FactoryLifecycleState | str = FactoryLifecycleState.ACTIVE
    budget_policy: BudgetPolicy | dict[str, Any] = field(default_factory=dict)
    cap_policy: CapPolicy | dict[str, Any] = field(default_factory=dict)
    homeostasis_policy: dict[str, Any] = field(default_factory=dict)
    publication_policy: PublicationPolicy | dict[str, Any] = field(default_factory=dict)
    authorization_policy: AuthorizationPolicy | dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "kind": _enum_value(self.kind),
            "status": _enum_value(self.status),
            "budget_policy": _policy_payload(self.budget_policy),
            "cap_policy": _policy_payload(self.cap_policy),
            "homeostasis_policy": _policy_payload(self.homeostasis_policy),
            "publication_policy": _policy_payload(self.publication_policy),
            "authorization_policy": _policy_payload(self.authorization_policy),
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
    homeostasis_policy: dict[str, Any] | None = None
    publication_policy: PublicationPolicy | dict[str, Any] | None = None
    authorization_policy: AuthorizationPolicy | dict[str, Any] | None = None
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
            ("homeostasis_policy", self.homeostasis_policy),
            ("publication_policy", self.publication_policy),
            ("authorization_policy", self.authorization_policy),
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
    trigger: str | None = None
    event_triggers: tuple[str | dict[str, Any], ...] = ()
    event_scope: str | None = None
    cooldown_seconds: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        for key, value in (
            ("cadence", self.cadence),
            ("timezone", self.timezone),
            ("max_active_runs", self.max_active_runs),
            ("trigger", self.trigger),
            ("event_scope", self.event_scope),
            ("cooldown_seconds", self.cooldown_seconds),
        ):
            if value is not None:
                payload[key] = value
        if self.event_triggers:
            payload["event_triggers"] = [
                dict(item) if isinstance(item, Mapping) else str(item)
                for item in self.event_triggers
            ]
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
class AuthorizationPolicy:
    authorized_project_ids: tuple[str, ...] = ()
    denied_project_ids: tuple[str, ...] = ()
    allowed_factory_project_roles: tuple[str, ...] = ()
    enabled: bool | None = None
    requires_audit_trail: bool | None = None
    disclosure_rules: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        if self.authorized_project_ids:
            payload["authorized_project_ids"] = list(self.authorized_project_ids)
        if self.denied_project_ids:
            payload["denied_project_ids"] = list(self.denied_project_ids)
        if self.allowed_factory_project_roles:
            payload["allowed_factory_project_roles"] = list(self.allowed_factory_project_roles)
        if self.enabled is not None:
            payload["enabled"] = self.enabled
        if self.requires_audit_trail is not None:
            payload["requires_audit_trail"] = self.requires_audit_trail
        if self.disclosure_rules:
            payload["disclosure_rules"] = dict(self.disclosure_rules)
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
    homeostasis_policy: dict[str, object] = field(default_factory=dict)
    publication_policy: dict[str, object] = field(default_factory=dict)
    authorization_policy: dict[str, object] = field(default_factory=dict)
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
            homeostasis_policy=_optional_object_dict(
                mapping.get("homeostasis_policy"),
                label="factory.homeostasis_policy",
            ),
            publication_policy=_optional_object_dict(
                mapping.get("publication_policy"),
                label="factory.publication_policy",
            ),
            authorization_policy=_optional_object_dict(
                mapping.get("authorization_policy"),
                label="factory.authorization_policy",
            ),
            metadata=_optional_object_dict(mapping.get("metadata"), label="factory.metadata"),
            created_by_user_id=_optional_string(mapping, "created_by_user_id"),
            created_at=_optional_datetime(mapping, "created_at"),
            updated_at=_optional_datetime(mapping, "updated_at"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryProjectLinkRequest:
    project_id: str
    role: FactoryProjectRole | str = FactoryProjectRole.CANONICAL
    status: FactoryProjectStatus | str = FactoryProjectStatus.ACTIVE
    display_name: str | None = None
    description: str | None = None
    workspace_policy: dict[str, Any] = field(default_factory=dict)
    resource_bindings: dict[str, Any] = field(default_factory=dict)
    feed_health: dict[str, Any] = field(default_factory=dict)
    default_launch_profile: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "project_id": self.project_id,
            "role": _enum_value(self.role),
            "status": _enum_value(self.status),
            "workspace_policy": _policy_payload(self.workspace_policy),
            "resource_bindings": _policy_payload(self.resource_bindings),
            "feed_health": _policy_payload(self.feed_health),
            "default_launch_profile": _policy_payload(self.default_launch_profile),
            "metadata": _policy_payload(self.metadata),
        }
        for key, value in (
            ("display_name", self.display_name),
            ("description", self.description),
        ):
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class FactoryProjectPatchRequest:
    role: FactoryProjectRole | str | None = None
    status: FactoryProjectStatus | str | None = None
    display_name: str | None = None
    description: str | None = None
    workspace_policy: dict[str, Any] | None = None
    resource_bindings: dict[str, Any] | None = None
    feed_health: dict[str, Any] | None = None
    default_launch_profile: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in (
            ("role", self.role),
            ("status", self.status),
        ):
            if value is not None:
                payload[key] = _enum_value(value)
        for key, value in (
            ("display_name", self.display_name),
            ("description", self.description),
        ):
            if value is not None:
                payload[key] = value
        for key, value in (
            ("workspace_policy", self.workspace_policy),
            ("resource_bindings", self.resource_bindings),
            ("feed_health", self.feed_health),
            ("default_launch_profile", self.default_launch_profile),
            ("metadata", self.metadata),
        ):
            if value is not None:
                payload[key] = _policy_payload(value)
        return payload


@dataclass(frozen=True)
class EffortCreateRequest:
    factory_id: str
    project_id: str
    name: str
    allow_implicit_project_link: bool = True
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
    authorization_policy: AuthorizationPolicy | dict[str, Any] = field(default_factory=dict)
    actor_notes: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "factory_id": self.factory_id,
            "project_id": self.project_id,
            "name": self.name,
            "allow_implicit_project_link": self.allow_implicit_project_link,
            "status": _enum_value(self.status),
            "effort_type": _enum_value(self.effort_type),
            "recurrence_policy": _policy_payload(self.recurrence_policy),
            "decision_needed": self.decision_needed,
            "budget_policy": _policy_payload(self.budget_policy),
            "publication_policy": _policy_payload(self.publication_policy),
            "authorization_policy": _policy_payload(self.authorization_policy),
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
    authorization_policy: AuthorizationPolicy | dict[str, Any] | None = None
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
            ("authorization_policy", self.authorization_policy),
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
class EffortFromRunsRequest:
    project_id: str
    name: str
    run_ids: tuple[str, ...] | list[str]
    factory_id: str | None = None

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "project_id": self.project_id,
            "name": self.name,
            "run_ids": list(self.run_ids),
        }
        if self.factory_id is not None:
            payload["factory_id"] = self.factory_id
        return payload


@dataclass(frozen=True)
class GraduationProposal:
    """Gardener-authored proposal to graduate related Runs into a persistent Effort."""

    proposal_id: str
    project_id: str
    suggested_name: str
    run_ids: tuple[str, ...]
    factory_id: str | None = None
    hypothesis_or_topic: str | None = None
    effort_type: EffortType | None = None
    rationale: str | None = None
    confidence: float | None = None
    created_at: datetime | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> GraduationProposal:
        mapping = _require_mapping(payload, label="graduation proposal")
        effort_type_value = _optional_string(mapping, "effort_type")
        return cls(
            proposal_id=_require_string(
                mapping, "proposal_id", label="graduation_proposal.proposal_id"
            ),
            project_id=_require_string(
                mapping, "project_id", label="graduation_proposal.project_id"
            ),
            suggested_name=_require_string(
                mapping, "suggested_name", label="graduation_proposal.suggested_name"
            ),
            run_ids=_string_tuple(mapping.get("run_ids")),
            factory_id=_optional_string(mapping, "factory_id"),
            hypothesis_or_topic=_optional_string(mapping, "hypothesis_or_topic"),
            effort_type=EffortType(effort_type_value) if effort_type_value is not None else None,
            rationale=_optional_string(mapping, "rationale"),
            confidence=_optional_float(mapping, "confidence"),
            created_at=_optional_datetime(mapping, "created_at"),
            raw=dict(mapping),
        )


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
    authorization_policy: dict[str, object] = field(default_factory=dict)
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
            authorization_policy=_optional_object_dict(
                mapping.get("authorization_policy"),
                label="effort.authorization_policy",
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
class FactoryProjectLink:
    factory_project_id: str
    org_id: str
    factory_id: str
    project_id: str
    role: FactoryProjectRole
    status: FactoryProjectStatus
    display_name: str | None = None
    description: str | None = None
    workspace_policy: dict[str, object] = field(default_factory=dict)
    resource_bindings: dict[str, object] = field(default_factory=dict)
    feed_health: dict[str, object] = field(default_factory=dict)
    default_launch_profile: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    created_by_user_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    project: FactoryProjectSummary | None = None
    effort_count: int = 0
    latest_run_id: str | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryProjectLink:
        mapping = _require_mapping(payload, label="factory project link")
        project_payload = mapping.get("project")
        return cls(
            factory_project_id=_require_string(
                mapping,
                "factory_project_id",
                label="factory_project.factory_project_id",
            ),
            org_id=_require_string(mapping, "org_id", label="factory_project.org_id"),
            factory_id=_require_string(
                mapping,
                "factory_id",
                label="factory_project.factory_id",
            ),
            project_id=_require_string(
                mapping,
                "project_id",
                label="factory_project.project_id",
            ),
            role=FactoryProjectRole(_require_string(mapping, "role", label="factory_project.role")),
            status=FactoryProjectStatus(
                _require_string(mapping, "status", label="factory_project.status")
            ),
            display_name=_optional_string(mapping, "display_name"),
            description=_optional_string(mapping, "description"),
            workspace_policy=_optional_object_dict(
                mapping.get("workspace_policy"),
                label="factory_project.workspace_policy",
            ),
            resource_bindings=_optional_object_dict(
                mapping.get("resource_bindings"),
                label="factory_project.resource_bindings",
            ),
            feed_health=_optional_object_dict(
                mapping.get("feed_health"),
                label="factory_project.feed_health",
            ),
            default_launch_profile=_optional_object_dict(
                mapping.get("default_launch_profile"),
                label="factory_project.default_launch_profile",
            ),
            metadata=_optional_object_dict(
                mapping.get("metadata"),
                label="factory_project.metadata",
            ),
            created_by_user_id=_optional_string(mapping, "created_by_user_id"),
            created_at=_optional_datetime(mapping, "created_at"),
            updated_at=_optional_datetime(mapping, "updated_at"),
            project=(
                FactoryProjectSummary.from_wire(project_payload)
                if project_payload is not None
                else None
            ),
            effort_count=int(mapping.get("effort_count") or 0),
            latest_run_id=_optional_string(mapping, "latest_run_id"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryWorkspace:
    factory_id: str
    project: FactoryProjectLink | None = None
    canonical_project: FactoryProjectLink | None = None
    projects: tuple[FactoryProjectLink, ...] = ()
    workspace_context: dict[str, object] = field(default_factory=dict)
    workspace_readiness: dict[str, object] = field(default_factory=dict)
    workspace_policy: dict[str, object] = field(default_factory=dict)
    resource_bindings: dict[str, object] = field(default_factory=dict)
    feed_health: dict[str, object] = field(default_factory=dict)
    default_launch_profile: dict[str, object] = field(default_factory=dict)
    resource_bindings_by_project: dict[str, object] = field(default_factory=dict)
    feeds: dict[str, object] = field(default_factory=dict)
    default_launch_profiles: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryWorkspace:
        mapping = _require_mapping(payload, label="factory workspace")
        project_payload = mapping.get("project")
        canonical_project_payload = mapping.get("canonical_project")
        return cls(
            factory_id=_require_string(mapping, "factory_id", label="workspace.factory_id"),
            project=(
                FactoryProjectLink.from_wire(project_payload)
                if project_payload is not None
                else None
            ),
            canonical_project=(
                FactoryProjectLink.from_wire(canonical_project_payload)
                if canonical_project_payload is not None
                else None
            ),
            projects=tuple(
                FactoryProjectLink.from_wire(item) for item in list(mapping.get("projects") or [])
            ),
            workspace_context=_optional_object_dict(
                mapping.get("workspace_context"),
                label="workspace.workspace_context",
            ),
            workspace_readiness=_optional_object_dict(
                mapping.get("workspace_readiness"),
                label="workspace.workspace_readiness",
            ),
            workspace_policy=_optional_object_dict(
                mapping.get("workspace_policy"),
                label="workspace.workspace_policy",
            ),
            resource_bindings=_optional_object_dict(
                mapping.get("resource_bindings"),
                label="workspace.resource_bindings",
            ),
            feed_health=_optional_object_dict(
                mapping.get("feed_health"),
                label="workspace.feed_health",
            ),
            default_launch_profile=_optional_object_dict(
                mapping.get("default_launch_profile"),
                label="workspace.default_launch_profile",
            ),
            resource_bindings_by_project=_optional_object_dict(
                mapping.get("resource_bindings_by_project"),
                label="workspace.resource_bindings_by_project",
            ),
            feeds=_optional_object_dict(mapping.get("feeds"), label="workspace.feeds"),
            default_launch_profiles=_optional_object_dict(
                mapping.get("default_launch_profiles"),
                label="workspace.default_launch_profiles",
            ),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryIdeaCreateRequest:
    title: str
    body: str | None = None
    status: FactoryIdeaStatus | str = FactoryIdeaStatus.OPEN
    source: FactoryIdeaSource | str = FactoryIdeaSource.HUMAN
    project_id: str | None = None
    effort_id: str | None = None
    run_id: str | None = None
    priority: str | None = None
    tags: tuple[str, ...] = ()
    promotion_target: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "title": self.title,
            "status": _enum_value(self.status),
            "source": _enum_value(self.source),
            "tags": list(self.tags),
            "promotion_target": _policy_payload(self.promotion_target),
            "metadata": _policy_payload(self.metadata),
        }
        for key, value in (
            ("body", self.body),
            ("project_id", self.project_id),
            ("effort_id", self.effort_id),
            ("run_id", self.run_id),
            ("priority", self.priority),
        ):
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class FactoryIdeaPatchRequest:
    title: str | None = None
    body: str | None = None
    status: FactoryIdeaStatus | str | None = None
    source: FactoryIdeaSource | str | None = None
    project_id: str | None = None
    effort_id: str | None = None
    run_id: str | None = None
    priority: str | None = None
    tags: tuple[str, ...] | None = None
    promotion_target: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in (
            ("title", self.title),
            ("body", self.body),
            ("project_id", self.project_id),
            ("effort_id", self.effort_id),
            ("run_id", self.run_id),
            ("priority", self.priority),
        ):
            if value is not None:
                payload[key] = value
        if self.status is not None:
            payload["status"] = _enum_value(self.status)
        if self.source is not None:
            payload["source"] = _enum_value(self.source)
        if self.tags is not None:
            payload["tags"] = list(self.tags)
        for key, value in (
            ("promotion_target", self.promotion_target),
            ("metadata", self.metadata),
        ):
            if value is not None:
                payload[key] = _policy_payload(value)
        return payload


@dataclass(frozen=True)
class FactoryIdea:
    idea_id: str
    org_id: str
    factory_id: str
    title: str
    status: FactoryIdeaStatus
    source: FactoryIdeaSource
    body: str | None = None
    project_id: str | None = None
    effort_id: str | None = None
    run_id: str | None = None
    priority: str | None = None
    tags: tuple[str, ...] = ()
    promotion_target: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    created_by_user_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryIdea:
        mapping = _require_mapping(payload, label="factory idea")
        return cls(
            idea_id=_require_string(mapping, "idea_id", label="idea.idea_id"),
            org_id=_require_string(mapping, "org_id", label="idea.org_id"),
            factory_id=_require_string(mapping, "factory_id", label="idea.factory_id"),
            project_id=_optional_string(mapping, "project_id"),
            effort_id=_optional_string(mapping, "effort_id"),
            run_id=_optional_string(mapping, "run_id"),
            title=_require_string(mapping, "title", label="idea.title"),
            body=_optional_string(mapping, "body"),
            status=FactoryIdeaStatus(_require_string(mapping, "status", label="idea.status")),
            source=FactoryIdeaSource(_require_string(mapping, "source", label="idea.source")),
            priority=_optional_string(mapping, "priority"),
            tags=_string_tuple(mapping.get("tags")),
            promotion_target=_optional_object_dict(
                mapping.get("promotion_target"),
                label="idea.promotion_target",
            ),
            metadata=_optional_object_dict(mapping.get("metadata"), label="idea.metadata"),
            created_by_user_id=_optional_string(mapping, "created_by_user_id"),
            created_at=_optional_datetime(mapping, "created_at"),
            updated_at=_optional_datetime(mapping, "updated_at"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryActorOutputCreateRequest:
    actor_role: FactoryActorRole | str
    kind: FactoryActorOutputKind | str
    title: str
    summary: str | None = None
    status: FactoryActorOutputStatus | str = FactoryActorOutputStatus.DRAFT
    project_id: str | None = None
    effort_id: str | None = None
    run_id: str | None = None
    report_id: str | None = None
    work_product_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "actor_role": _enum_value(self.actor_role),
            "kind": _enum_value(self.kind),
            "title": self.title,
            "status": _enum_value(self.status),
            "payload": _policy_payload(self.payload),
            "metadata": _policy_payload(self.metadata),
        }
        for key, value in (
            ("summary", self.summary),
            ("project_id", self.project_id),
            ("effort_id", self.effort_id),
            ("run_id", self.run_id),
            ("report_id", self.report_id),
            ("work_product_id", self.work_product_id),
        ):
            if value is not None:
                wire[key] = value
        return wire


@dataclass(frozen=True)
class FactoryActorOutputPatchRequest:
    actor_role: FactoryActorRole | str | None = None
    kind: FactoryActorOutputKind | str | None = None
    title: str | None = None
    summary: str | None = None
    status: FactoryActorOutputStatus | str | None = None
    project_id: str | None = None
    effort_id: str | None = None
    run_id: str | None = None
    report_id: str | None = None
    work_product_id: str | None = None
    payload: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {}
        for key, value in (
            ("title", self.title),
            ("summary", self.summary),
            ("project_id", self.project_id),
            ("effort_id", self.effort_id),
            ("run_id", self.run_id),
            ("report_id", self.report_id),
            ("work_product_id", self.work_product_id),
        ):
            if value is not None:
                wire[key] = value
        if self.actor_role is not None:
            wire["actor_role"] = _enum_value(self.actor_role)
        if self.kind is not None:
            wire["kind"] = _enum_value(self.kind)
        if self.status is not None:
            wire["status"] = _enum_value(self.status)
        for key, value in (
            ("payload", self.payload),
            ("metadata", self.metadata),
        ):
            if value is not None:
                wire[key] = _policy_payload(value)
        return wire


@dataclass(frozen=True)
class FactoryActorOutput:
    actor_output_id: str
    org_id: str
    factory_id: str
    actor_role: FactoryActorRole
    kind: FactoryActorOutputKind
    status: FactoryActorOutputStatus
    title: str
    project_id: str | None = None
    effort_id: str | None = None
    run_id: str | None = None
    report_id: str | None = None
    work_product_id: str | None = None
    summary: str | None = None
    payload: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    created_by_user_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryActorOutput:
        mapping = _require_mapping(payload, label="factory actor output")
        return cls(
            actor_output_id=_require_string(
                mapping,
                "actor_output_id",
                label="actor_output.actor_output_id",
            ),
            org_id=_require_string(mapping, "org_id", label="actor_output.org_id"),
            factory_id=_require_string(
                mapping,
                "factory_id",
                label="actor_output.factory_id",
            ),
            project_id=_optional_string(mapping, "project_id"),
            effort_id=_optional_string(mapping, "effort_id"),
            run_id=_optional_string(mapping, "run_id"),
            report_id=_optional_string(mapping, "report_id"),
            work_product_id=_optional_string(mapping, "work_product_id"),
            actor_role=FactoryActorRole(
                _require_string(mapping, "actor_role", label="actor_output.actor_role")
            ),
            kind=FactoryActorOutputKind(
                _require_string(mapping, "kind", label="actor_output.kind")
            ),
            status=FactoryActorOutputStatus(
                _require_string(mapping, "status", label="actor_output.status")
            ),
            title=_require_string(mapping, "title", label="actor_output.title"),
            summary=_optional_string(mapping, "summary"),
            payload=_optional_object_dict(
                mapping.get("payload"),
                label="actor_output.payload",
            ),
            metadata=_optional_object_dict(
                mapping.get("metadata"),
                label="actor_output.metadata",
            ),
            created_by_user_id=_optional_string(mapping, "created_by_user_id"),
            created_at=_optional_datetime(mapping, "created_at"),
            updated_at=_optional_datetime(mapping, "updated_at"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryRunSummary:
    run_id: str
    project_id: str
    public_state: str
    effort_id: str | None = None
    run_kind: FactoryRunKind = FactoryRunKind.RESEARCH
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
            run_kind=FactoryRunKind(
                _optional_string(mapping, "run_kind") or FactoryRunKind.RESEARCH
            ),
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
    summary: str | None = None
    effort_id: str | None = None
    artifact_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
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
            summary=_optional_string(mapping, "summary"),
            status=_require_string(mapping, "status", label="work_product.status"),
            readiness=_require_string(mapping, "readiness", label="work_product.readiness"),
            artifact_id=_optional_string(mapping, "artifact_id"),
            metadata=_optional_object_dict(mapping.get("metadata"), label="work_product.metadata"),
            created_at=_optional_datetime(mapping, "created_at"),
            updated_at=_optional_datetime(mapping, "updated_at"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryMaintenanceAction:
    vital: str
    action_key: str
    action: str
    owner: str | None = None
    status: str | None = None
    reason: str | None = None
    band: dict[str, object] = field(default_factory=dict)
    observed: dict[str, object] = field(default_factory=dict)
    wake: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryMaintenanceAction:
        mapping = _require_mapping(payload, label="factory maintenance action")
        vital = _require_string(mapping, "vital", label="factory_action.vital")
        action = _require_string(mapping, "action", label="factory_action.action")
        owner = _optional_string(mapping, "owner")
        action_key = _optional_string(mapping, "action_key")
        if action_key is None:
            action_key = f"{vital}:{owner or 'unowned'}:{action}"
        return cls(
            vital=vital,
            action_key=action_key,
            action=action,
            owner=owner,
            status=_optional_string(mapping, "status"),
            reason=_optional_string(mapping, "reason"),
            band=_optional_object_dict(
                mapping.get("band"),
                label="factory_action.band",
            ),
            observed=_optional_object_dict(
                mapping.get("observed"),
                label="factory_action.observed",
            ),
            wake=_optional_string(mapping, "wake"),
            metadata=_optional_object_dict(
                mapping.get("metadata"),
                label="factory_action.metadata",
            ),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryHealth:
    schema_version: str
    status: str
    health_score: float | None = None
    threshold: float | None = None
    evaluated_at: datetime | None = None
    policy: dict[str, object] = field(default_factory=dict)
    vitals: dict[str, object] = field(default_factory=dict)
    triggers: dict[str, object] = field(default_factory=dict)
    recommended_actions: tuple[FactoryMaintenanceAction, ...] = ()
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryHealth:
        mapping = _require_mapping(payload, label="factory health")
        return cls(
            schema_version=_require_string(
                mapping,
                "schema_version",
                label="factory_health.schema_version",
            ),
            status=_require_string(mapping, "status", label="factory_health.status"),
            health_score=_optional_float(mapping, "health_score"),
            threshold=_optional_float(mapping, "threshold"),
            evaluated_at=_optional_datetime(mapping, "evaluated_at"),
            policy=_optional_object_dict(
                mapping.get("policy"),
                label="factory_health.policy",
            ),
            vitals=_optional_object_dict(
                mapping.get("vitals"),
                label="factory_health.vitals",
            ),
            triggers=_optional_object_dict(
                mapping.get("triggers"),
                label="factory_health.triggers",
            ),
            recommended_actions=tuple(
                FactoryMaintenanceAction.from_wire(item)
                for item in list(mapping.get("recommended_actions") or [])
            ),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryControlLoopFlags:
    schema_version: str
    service_type: str
    environment: str
    scheduler_enabled: bool
    reactor_enabled: bool
    flag_names: dict[str, str]
    observed_at: datetime

    @classmethod
    def from_wire(cls, payload: object) -> FactoryControlLoopFlags:
        mapping = _require_mapping(payload, label="factory control-loop flags")
        schema_version = _require_string(
            mapping,
            "schema_version",
            label="factory control-loop flags.schema_version",
        )
        if schema_version != "factory_control_loop_flags.v1":
            raise ValueError(
                "factory control-loop flags schema must be factory_control_loop_flags.v1"
            )
        observed_at = _optional_datetime(mapping, "observed_at")
        if observed_at is None:
            raise ValueError("factory control-loop flags observed_at is required")
        raw_flag_names = _require_mapping(
            mapping.get("flag_names"), label="factory control-loop flags.flag_names"
        )
        return cls(
            schema_version=schema_version,
            service_type=_require_string(
                mapping,
                "service_type",
                label="factory control-loop flags.service_type",
            ),
            environment=_require_string(
                mapping,
                "environment",
                label="factory control-loop flags.environment",
            ),
            scheduler_enabled=bool(_optional_bool(mapping, "scheduler_enabled")),
            reactor_enabled=bool(_optional_bool(mapping, "reactor_enabled")),
            flag_names={str(key): str(value) for key, value in raw_flag_names.items()},
            observed_at=observed_at,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "service_type": self.service_type,
            "environment": self.environment,
            "scheduler_enabled": self.scheduler_enabled,
            "reactor_enabled": self.reactor_enabled,
            "flag_names": dict(self.flag_names),
            "observed_at": self.observed_at.isoformat(),
        }


@dataclass(frozen=True)
class FactoryReactorReceipt:
    effort_id: str
    run_id: str
    observed_at: datetime
    terminal_outcome: str | None = None
    failure_class: str | None = None
    action: str | None = None
    review_actor_keys: tuple[str, ...] = ()

    @classmethod
    def from_wire(cls, payload: object) -> FactoryReactorReceipt:
        mapping = _require_mapping(payload, label="factory reactor receipt")
        observed_at = _optional_datetime(mapping, "observed_at")
        if observed_at is None:
            raise ValueError("factory reactor receipt observed_at is required")
        return cls(
            effort_id=_require_string(
                mapping, "effort_id", label="factory reactor receipt.effort_id"
            ),
            run_id=_require_string(mapping, "run_id", label="factory reactor receipt.run_id"),
            terminal_outcome=_optional_string(mapping, "terminal_outcome"),
            failure_class=_optional_string(mapping, "failure_class"),
            action=_optional_string(mapping, "action"),
            review_actor_keys=_string_tuple(mapping.get("review_actor_keys")),
            observed_at=observed_at,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "effort_id": self.effort_id,
            "run_id": self.run_id,
            "terminal_outcome": self.terminal_outcome,
            "failure_class": self.failure_class,
            "action": self.action,
            "review_actor_keys": list(self.review_actor_keys),
            "observed_at": self.observed_at.isoformat(),
        }


@dataclass(frozen=True)
class FactoryReactorStatus:
    kind: str
    owner: str
    state: str
    processed_terminal_runs: int
    efforts_with_receipts: int
    last_observed_at: datetime | None = None
    last_receipt: FactoryReactorReceipt | None = None

    @classmethod
    def from_wire(cls, payload: object) -> FactoryReactorStatus:
        mapping = _require_mapping(payload, label="factory reactor status")
        return cls(
            kind=_require_string(mapping, "kind", label="factory reactor status.kind"),
            owner=_require_string(mapping, "owner", label="factory reactor status.owner"),
            state=_require_string(mapping, "state", label="factory reactor status.state"),
            processed_terminal_runs=int(mapping.get("processed_terminal_runs") or 0),
            efforts_with_receipts=int(mapping.get("efforts_with_receipts") or 0),
            last_observed_at=_optional_datetime(mapping, "last_observed_at"),
            last_receipt=(
                FactoryReactorReceipt.from_wire(mapping.get("last_receipt"))
                if mapping.get("last_receipt") is not None
                else None
            ),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "owner": self.owner,
            "state": self.state,
            "last_observed_at": (
                self.last_observed_at.isoformat() if self.last_observed_at is not None else None
            ),
            "last_receipt": (
                self.last_receipt.to_wire() if self.last_receipt is not None else None
            ),
            "processed_terminal_runs": self.processed_terminal_runs,
            "efforts_with_receipts": self.efforts_with_receipts,
        }


@dataclass(frozen=True)
class FactoryRuntimeStatus:
    kind: str
    owner: str
    mode: str
    state: str
    enabled: bool
    observed_at: datetime
    reactor: FactoryReactorStatus
    control_loop_flags: FactoryControlLoopFlags
    last_observed_at: datetime | None = None
    last_cycle: dict[str, object] = field(default_factory=dict)
    next_event: dict[str, object] = field(default_factory=dict)
    schedule: tuple[dict[str, object], ...] = ()
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryRuntimeStatus:
        mapping = _require_mapping(payload, label="factory runtime status")
        observed_at = _optional_datetime(mapping, "observed_at")
        if observed_at is None:
            raise ValueError("factory runtime status observed_at is required")
        raw_schedule = mapping.get("schedule") or []
        if not isinstance(raw_schedule, list):
            raise ValueError("factory runtime status schedule must be a list")
        return cls(
            kind=_require_string(mapping, "kind", label="factory runtime status.kind"),
            owner=_require_string(mapping, "owner", label="factory runtime status.owner"),
            mode=_require_string(mapping, "mode", label="factory runtime status.mode"),
            state=_require_string(mapping, "state", label="factory runtime status.state"),
            enabled=bool(_optional_bool(mapping, "enabled")),
            observed_at=observed_at,
            last_observed_at=_optional_datetime(mapping, "last_observed_at"),
            last_cycle=_optional_object_dict(
                mapping.get("last_cycle"), label="factory runtime status.last_cycle"
            ),
            next_event=_optional_object_dict(
                mapping.get("next_event"), label="factory runtime status.next_event"
            ),
            schedule=tuple(
                _optional_object_dict(item, label="factory runtime status.schedule item")
                for item in raw_schedule
            ),
            reactor=FactoryReactorStatus.from_wire(mapping.get("reactor")),
            control_loop_flags=FactoryControlLoopFlags.from_wire(mapping.get("control_loop_flags")),
            raw=dict(mapping),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "owner": self.owner,
            "mode": self.mode,
            "state": self.state,
            "enabled": self.enabled,
            "observed_at": self.observed_at.isoformat(),
            "last_observed_at": (
                self.last_observed_at.isoformat() if self.last_observed_at is not None else None
            ),
            "last_cycle": dict(self.last_cycle),
            "next_event": dict(self.next_event),
            "schedule": [dict(item) for item in self.schedule],
            "reactor": self.reactor.to_wire(),
            "control_loop_flags": self.control_loop_flags.to_wire(),
        }


@dataclass(frozen=True)
class ExperimentBundle:
    """Typed top-level contract for an owner-assembled experiment bundle."""

    experiment_id: str
    project_id: str
    schema_version: str
    run_ids: tuple[str, ...] = ()
    experiment: dict[str, object] = field(default_factory=dict)
    candidate: dict[str, object] = field(default_factory=dict)
    executions: tuple[dict[str, object], ...] = ()
    evaluations: tuple[dict[str, object], ...] = ()
    trace_index: tuple[dict[str, object], ...] = ()
    economics: dict[str, object] = field(default_factory=dict)
    decisions: dict[str, object] = field(default_factory=dict)
    provenance: dict[str, object] = field(default_factory=dict)
    workspace_layout: dict[str, object] = field(default_factory=dict)
    integrity: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)

    @property
    def accepted_cycle(self) -> bool:
        return self.integrity.get("accepted_cycle") is True

    @classmethod
    def from_wire(cls, payload: object) -> ExperimentBundle:
        mapping = _require_mapping(payload, label="experiment bundle")
        return cls(
            experiment_id=_require_string(mapping, "experiment_id", label="experiment bundle"),
            project_id=_require_string(mapping, "project_id", label="experiment bundle"),
            schema_version=_require_string(mapping, "schema_version", label="experiment bundle"),
            run_ids=tuple(str(item) for item in list(mapping.get("run_ids") or [])),
            experiment=_optional_object_dict(
                mapping.get("experiment"), label="experiment bundle experiment"
            ),
            candidate=_optional_object_dict(
                mapping.get("candidate"), label="experiment bundle candidate"
            ),
            executions=tuple(
                _optional_object_dict(item, label="experiment bundle execution")
                for item in list(mapping.get("executions") or [])
            ),
            evaluations=tuple(
                _optional_object_dict(item, label="experiment bundle evaluation")
                for item in list(mapping.get("evaluations") or [])
            ),
            trace_index=tuple(
                _optional_object_dict(item, label="experiment bundle trace")
                for item in list(mapping.get("trace_index") or [])
            ),
            economics=_optional_object_dict(
                mapping.get("economics"), label="experiment bundle economics"
            ),
            decisions=_optional_object_dict(
                mapping.get("decisions"), label="experiment bundle decisions"
            ),
            provenance=_optional_object_dict(
                mapping.get("provenance"), label="experiment bundle provenance"
            ),
            workspace_layout=_optional_object_dict(
                mapping.get("workspace_layout"), label="experiment bundle workspace_layout"
            ),
            integrity=_optional_object_dict(
                mapping.get("integrity"), label="experiment bundle integrity"
            ),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class ExperimentHistory:
    project_id: str
    schema_version: str
    bundles: tuple[ExperimentBundle, ...] = ()
    accepted_cycles: int = 0
    incomplete_cycles: int = 0
    missing_evidence_alerts: tuple[dict[str, object], ...] = ()

    @classmethod
    def from_wire(cls, payload: object) -> ExperimentHistory:
        mapping = _require_mapping(payload, label="experiment history")
        return cls(
            project_id=_require_string(mapping, "project_id", label="experiment history"),
            schema_version=_require_string(mapping, "schema_version", label="experiment history"),
            bundles=tuple(
                ExperimentBundle.from_wire(item)
                for item in _optional_object_tuple(
                    mapping.get("bundles"), label="experiment history bundles"
                )
            ),
            accepted_cycles=_int_value(
                mapping, "accepted_cycles", label="experiment history accepted_cycles"
            ),
            incomplete_cycles=_int_value(
                mapping, "incomplete_cycles", label="experiment history incomplete_cycles"
            ),
            missing_evidence_alerts=_optional_object_tuple(
                mapping.get("missing_evidence_alerts"),
                label="experiment history missing_evidence_alerts",
            ),
        )


@dataclass(frozen=True)
class ExperimentComparison:
    project_id: str
    schema_version: str
    experiment_ids: tuple[str, ...]
    comparable: bool
    comparison_dimensions: dict[str, object] = field(default_factory=dict)
    rows: tuple[dict[str, object], ...] = ()
    not_comparable_reasons: tuple[str, ...] = ()

    @classmethod
    def from_wire(cls, payload: object) -> ExperimentComparison:
        mapping = _require_mapping(payload, label="experiment comparison")
        return cls(
            project_id=_require_string(mapping, "project_id", label="experiment comparison"),
            schema_version=_require_string(
                mapping, "schema_version", label="experiment comparison"
            ),
            experiment_ids=_string_tuple(mapping.get("experiment_ids")),
            comparable=bool(mapping.get("comparable")),
            comparison_dimensions=_optional_object_dict(
                mapping.get("comparison_dimensions"),
                label="experiment comparison dimensions",
            ),
            rows=_optional_object_tuple(
                mapping.get("rows"),
                label="experiment comparison rows",
            ),
            not_comparable_reasons=_string_tuple(mapping.get("not_comparable_reasons")),
        )


@dataclass(frozen=True)
class FactoryOperatingWindow:
    schema_version: str
    evaluated_at: datetime
    window_started_at: datetime
    window_days: int
    required_cycles: int
    terminal_attempts: int
    observed_cycles: int
    rejected_cycles: int
    remaining_cycles: int
    status: str
    first_cycle_at: datetime | None = None
    latest_cycle_at: datetime | None = None
    cycle_run_ids: tuple[str, ...] = ()
    rejected_cycle_run_ids: tuple[str, ...] = ()
    cycle_evidence: tuple[dict[str, object], ...] = ()

    @classmethod
    def from_wire(cls, payload: object) -> FactoryOperatingWindow:
        mapping = _require_mapping(payload, label="factory operating window")
        evaluated_at = _optional_datetime(mapping, "evaluated_at")
        window_started_at = _optional_datetime(mapping, "window_started_at")
        if evaluated_at is None or window_started_at is None:
            raise ValueError("factory operating window timestamps are required")
        return cls(
            schema_version=_require_string(
                mapping, "schema_version", label="factory operating window"
            ),
            evaluated_at=evaluated_at,
            window_started_at=window_started_at,
            window_days=(
                _int_value(mapping, "window_days", label="factory operating window window_days")
                or 30
            ),
            required_cycles=(
                _int_value(
                    mapping,
                    "required_cycles",
                    label="factory operating window required_cycles",
                )
                or 12
            ),
            terminal_attempts=_int_value(
                mapping,
                "terminal_attempts",
                label="factory operating window terminal_attempts",
            ),
            observed_cycles=_int_value(
                mapping,
                "observed_cycles",
                label="factory operating window observed_cycles",
            ),
            rejected_cycles=_int_value(
                mapping,
                "rejected_cycles",
                label="factory operating window rejected_cycles",
            ),
            remaining_cycles=_int_value(
                mapping,
                "remaining_cycles",
                label="factory operating window remaining_cycles",
            ),
            status=_require_string(mapping, "status", label="factory operating window"),
            first_cycle_at=_optional_datetime(mapping, "first_cycle_at"),
            latest_cycle_at=_optional_datetime(mapping, "latest_cycle_at"),
            cycle_run_ids=_string_tuple(mapping.get("cycle_run_ids")),
            rejected_cycle_run_ids=_string_tuple(mapping.get("rejected_cycle_run_ids")),
            cycle_evidence=_optional_object_tuple(
                mapping.get("cycle_evidence"),
                label="factory operating window cycle_evidence",
            ),
        )


@dataclass(frozen=True)
class FactoryStatus:
    factory: Factory
    projects: tuple[FactoryProjectSummary, ...] = ()
    linked_projects: tuple[FactoryProjectLink, ...] = ()
    workspace: FactoryWorkspace | None = None
    efforts: tuple[Effort, ...] = ()
    ideas: tuple[FactoryIdea, ...] = ()
    actor_outputs: tuple[FactoryActorOutput, ...] = ()
    efforts_by_status: dict[str, int] = field(default_factory=dict)
    latest_runs: tuple[FactoryRunSummary, ...] = ()
    latest_reports: tuple[FactoryReportSummary, ...] = ()
    latest_work_products: tuple[FactoryWorkProductSummary, ...] = ()
    open_decisions: tuple[Effort, ...] = ()
    paused_or_waiting: tuple[Effort, ...] = ()
    next_wake_at: datetime | None = None
    runtime: dict[str, object] = field(default_factory=dict)
    publication_states: dict[str, object] = field(default_factory=dict)
    costs_limits: dict[str, object] = field(default_factory=dict)
    factory_health: FactoryHealth | None = None
    proof_readiness: dict[str, object] = field(default_factory=dict)
    public_visuals: dict[str, object] = field(default_factory=dict)
    experiment_observability: ExperimentBundle | None = None
    judgment_state: dict[str, object] = field(default_factory=dict)
    operating_window: FactoryOperatingWindow | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @property
    def typed_runtime(self) -> FactoryRuntimeStatus:
        """Return the typed backend scheduler/reactor owner-route payload."""

        return FactoryRuntimeStatus.from_wire(self.runtime)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryStatus:
        mapping = _require_mapping(payload, label="factory status")
        experiment_payload = mapping.get("experiment_observability")
        experiment_observability = (
            ExperimentBundle.from_wire(experiment_payload)
            if isinstance(experiment_payload, Mapping)
            and bool(experiment_payload.get("experiment_id"))
            else None
        )
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
            linked_projects=tuple(
                FactoryProjectLink.from_wire(item)
                for item in list(mapping.get("linked_projects") or [])
            ),
            workspace=(
                FactoryWorkspace.from_wire(mapping.get("workspace"))
                if mapping.get("workspace") is not None
                else None
            ),
            efforts=tuple(Effort.from_wire(item) for item in list(mapping.get("efforts") or [])),
            ideas=tuple(FactoryIdea.from_wire(item) for item in list(mapping.get("ideas") or [])),
            actor_outputs=tuple(
                FactoryActorOutput.from_wire(item)
                for item in list(mapping.get("actor_outputs") or [])
            ),
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
            runtime=_optional_object_dict(
                mapping.get("runtime"),
                label="factory status runtime",
            ),
            publication_states=_optional_object_dict(
                mapping.get("publication_states"),
                label="factory status publication_states",
            ),
            costs_limits=_optional_object_dict(
                mapping.get("costs_limits"),
                label="factory status costs_limits",
            ),
            factory_health=(
                FactoryHealth.from_wire(mapping.get("factory_health"))
                if mapping.get("factory_health") is not None
                else None
            ),
            proof_readiness=_optional_object_dict(
                mapping.get("proof_readiness"),
                label="factory status proof_readiness",
            ),
            public_visuals=_optional_object_dict(
                mapping.get("public_visuals"),
                label="factory status public_visuals",
            ),
            experiment_observability=experiment_observability,
            judgment_state=_optional_object_dict(
                mapping.get("judgment_state"),
                label="factory status judgment_state",
            ),
            operating_window=(
                FactoryOperatingWindow.from_wire(mapping.get("operating_window"))
                if mapping.get("operating_window") is not None
                else None
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
    confirmed_preview_id: str | None = None
    confirmed_preview_token: str | None = field(default=None, repr=False)

    @classmethod
    def from_contract_wire(cls, payload: object) -> FactoryWakeDueRequest:
        """Parse the exact token-bound contract returned by a wake preview."""
        mapping = _require_mapping(payload, label="factory wake request contract")
        allowed_keys = {
            "launch_request",
            "limit",
            "allow_overlap",
            "continue_on_error",
        }
        unknown_keys = sorted(str(key) for key in set(mapping) - allowed_keys)
        if unknown_keys:
            raise ValueError(
                "factory wake request contract contains unknown fields: " + ", ".join(unknown_keys)
            )
        contract = cls.from_wire(mapping)
        canonical_contract = contract.to_contract_wire()
        replayed_fields = {str(key): canonical_contract[str(key)] for key in mapping}
        if dict(mapping) != replayed_fields:
            raise ValueError("factory wake request contract must be replayed exactly as returned")
        return contract

    @classmethod
    def from_wire(cls, payload: object) -> FactoryWakeDueRequest:
        mapping = _require_mapping(payload, label="factory wake request contract")
        launch_request = mapping.get("launch_request")
        if launch_request is not None and not isinstance(launch_request, Mapping):
            raise ValueError("factory wake launch_request must be an object")
        limit_value = mapping.get("limit")
        if limit_value is not None and (
            isinstance(limit_value, bool) or not isinstance(limit_value, int)
        ):
            raise ValueError("factory wake limit must be an integer")
        limit = 10 if limit_value is None else limit_value
        if not 1 <= limit <= 100:
            raise ValueError("factory wake limit must be between 1 and 100")
        continue_on_error = _optional_bool(mapping, "continue_on_error")
        return cls(
            launch_request=(
                _optional_object_dict(launch_request, label="factory wake launch_request")
                if launch_request is not None
                else None
            ),
            limit=limit,
            allow_overlap=bool(_optional_bool(mapping, "allow_overlap")),
            dry_run=bool(_optional_bool(mapping, "dry_run")),
            continue_on_error=(True if continue_on_error is None else continue_on_error),
            confirmed_preview_id=_optional_string(mapping, "confirmed_preview_id"),
            confirmed_preview_token=_optional_string(mapping, "confirmed_preview_token"),
        )

    def to_contract_wire(self) -> dict[str, Any]:
        """Return the token-bound request fields, excluding transport controls."""
        payload: dict[str, Any] = {
            "launch_request": (
                dict(self.launch_request) if self.launch_request is not None else None
            ),
            "limit": self.limit,
            "allow_overlap": self.allow_overlap,
            "continue_on_error": self.continue_on_error,
        }
        return payload

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "limit": self.limit,
            "allow_overlap": self.allow_overlap,
            "dry_run": self.dry_run,
            "continue_on_error": self.continue_on_error,
        }
        if self.launch_request is not None:
            payload["launch_request"] = dict(self.launch_request)
        if self.confirmed_preview_id is not None:
            payload["confirmed_preview_id"] = self.confirmed_preview_id
        if self.confirmed_preview_token is not None:
            payload["confirmed_preview_token"] = self.confirmed_preview_token
        return payload


@dataclass(frozen=True)
class FactoryWakeDueEffort:
    effort_id: str
    project_id: str
    status: str
    reason: str | None = None
    run_id: str | None = None
    run_kind: FactoryRunKind | None = None
    next_wake_at: datetime | None = None
    launch_preview: dict[str, object] = field(default_factory=dict)
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
            run_kind=_optional_enum(
                FactoryRunKind,
                mapping.get("run_kind"),
                field_name="wake.run_kind",
            ),
            next_wake_at=_optional_datetime(mapping, "next_wake_at"),
            launch_preview=_optional_object_dict(
                mapping.get("launch_preview"),
                label="wake.launch_preview",
            ),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryWakeDueResult:
    factory_id: str
    evaluated_at: datetime | None
    dry_run: bool
    launched: int = 0
    skipped: int = 0
    failed: int = 0
    efforts: tuple[FactoryWakeDueEffort, ...] = ()
    raw: dict[str, object] = field(default_factory=dict, repr=False)
    preview_id: str | None = None
    preview_token: str | None = field(default=None, repr=False)
    preview_expires_at: datetime | None = None
    confirmed_preview_id: str | None = None
    confirmation_required: bool = False
    request_contract: FactoryWakeDueRequest | None = None
    receipt_id: str | None = None
    executed_at: datetime | None = None
    ready: int = 0

    @classmethod
    def from_wire(cls, payload: object) -> FactoryWakeDueResult:
        mapping = _require_mapping(payload, label="factory wake due result")
        return cls(
            factory_id=_require_string(mapping, "factory_id", label="wake.factory_id"),
            evaluated_at=_optional_datetime(mapping, "evaluated_at"),
            dry_run=bool(_optional_bool(mapping, "dry_run")),
            preview_id=_optional_string(mapping, "preview_id"),
            preview_token=_optional_string(mapping, "preview_token"),
            preview_expires_at=_optional_datetime(mapping, "preview_expires_at"),
            confirmed_preview_id=_optional_string(mapping, "confirmed_preview_id"),
            confirmation_required=bool(_optional_bool(mapping, "confirmation_required")),
            request_contract=(
                FactoryWakeDueRequest.from_contract_wire(mapping.get("request_contract"))
                if mapping.get("request_contract") is not None
                else None
            ),
            receipt_id=_optional_string(mapping, "receipt_id"),
            executed_at=_optional_datetime(mapping, "executed_at"),
            ready=int(mapping.get("ready") or 0),
            launched=int(mapping.get("launched") or 0),
            skipped=int(mapping.get("skipped") or 0),
            failed=int(mapping.get("failed") or 0),
            efforts=tuple(
                FactoryWakeDueEffort.from_wire(item) for item in list(mapping.get("efforts") or [])
            ),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryCandidateGradingRequest:
    """Benchmark-owned grading record submitted for one immutable candidate."""

    grading: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        return {"grading": dict(self.grading)}


@dataclass(frozen=True)
class FactoryChampionSelectRequest:
    baseline_score: float
    effort_id: str | None = None

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"baseline_score": float(self.baseline_score)}
        if self.effort_id is not None:
            payload["effort_id"] = self.effort_id
        return payload


@dataclass(frozen=True)
class FactoryChampionRollbackRequest:
    to_candidate_id: str
    reason: str
    effort_id: str | None = None

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "to_candidate_id": self.to_candidate_id,
            "reason": self.reason,
        }
        if self.effort_id is not None:
            payload["effort_id"] = self.effort_id
        return payload


@dataclass(frozen=True)
class FactoryCandidate:
    candidate_id: str
    org_id: str
    factory_id: str
    candidate_key: str
    git_remote: str
    git_sha: str
    entrypoint: str
    execution_contract_version: str
    grading_status: FactoryCandidateGradingStatus
    published_at: datetime
    created_at: datetime
    updated_at: datetime
    project_id: str | None = None
    effort_id: str | None = None
    run_id: str | None = None
    work_product_id: str | None = None
    parent_candidate_id: str | None = None
    grading_target: dict[str, object] = field(default_factory=dict)
    artifact_ids: tuple[object, ...] = ()
    grading: dict[str, object] = field(default_factory=dict)
    held_out_score: float | None = None
    baseline_score: float | None = None
    graded_at: datetime | None = None
    is_champion: bool = False
    champion_selected_at: datetime | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryCandidate:
        mapping = _require_mapping(payload, label="factory candidate")
        published_at = _optional_datetime(mapping, "published_at")
        created_at = _optional_datetime(mapping, "created_at")
        updated_at = _optional_datetime(mapping, "updated_at")
        if published_at is None or created_at is None or updated_at is None:
            raise ValueError("factory candidate requires published_at, created_at, and updated_at")
        return cls(
            candidate_id=_require_string(mapping, "candidate_id", label="candidate.candidate_id"),
            org_id=_require_string(mapping, "org_id", label="candidate.org_id"),
            factory_id=_require_string(mapping, "factory_id", label="candidate.factory_id"),
            project_id=_optional_string(mapping, "project_id"),
            effort_id=_optional_string(mapping, "effort_id"),
            run_id=_optional_string(mapping, "run_id"),
            work_product_id=_optional_string(mapping, "work_product_id"),
            candidate_key=_require_string(
                mapping, "candidate_key", label="candidate.candidate_key"
            ),
            git_remote=_require_string(mapping, "git_remote", label="candidate.git_remote"),
            git_sha=_require_string(mapping, "git_sha", label="candidate.git_sha"),
            entrypoint=_require_string(mapping, "entrypoint", label="candidate.entrypoint"),
            execution_contract_version=_require_string(
                mapping,
                "execution_contract_version",
                label="candidate.execution_contract_version",
            ),
            parent_candidate_id=_optional_string(mapping, "parent_candidate_id"),
            grading_target=_optional_object_dict(
                mapping.get("grading_target"), label="candidate.grading_target"
            ),
            artifact_ids=tuple(mapping.get("artifact_ids") or ()),
            grading_status=FactoryCandidateGradingStatus(
                _require_string(mapping, "grading_status", label="candidate.grading_status")
            ),
            grading=_optional_object_dict(mapping.get("grading"), label="candidate.grading"),
            held_out_score=_optional_float(mapping, "held_out_score"),
            baseline_score=_optional_float(mapping, "baseline_score"),
            graded_at=_optional_datetime(mapping, "graded_at"),
            is_champion=bool(_optional_bool(mapping, "is_champion")),
            champion_selected_at=_optional_datetime(mapping, "champion_selected_at"),
            published_at=published_at,
            created_at=created_at,
            updated_at=updated_at,
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryChampionEvent:
    event_id: str
    org_id: str
    factory_id: str
    action: FactoryChampionEventAction
    created_at: datetime
    effort_id: str | None = None
    candidate_id: str | None = None
    git_sha: str | None = None
    held_out_score: float | None = None
    baseline_score: float | None = None
    reason: str | None = None
    payload: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryChampionEvent:
        mapping = _require_mapping(payload, label="factory champion event")
        created_at = _optional_datetime(mapping, "created_at")
        if created_at is None:
            raise ValueError("factory champion event requires created_at")
        return cls(
            event_id=_require_string(mapping, "event_id", label="event.event_id"),
            org_id=_require_string(mapping, "org_id", label="event.org_id"),
            factory_id=_require_string(mapping, "factory_id", label="event.factory_id"),
            effort_id=_optional_string(mapping, "effort_id"),
            candidate_id=_optional_string(mapping, "candidate_id"),
            action=FactoryChampionEventAction(
                _require_string(mapping, "action", label="event.action")
            ),
            git_sha=_optional_string(mapping, "git_sha"),
            held_out_score=_optional_float(mapping, "held_out_score"),
            baseline_score=_optional_float(mapping, "baseline_score"),
            reason=_optional_string(mapping, "reason"),
            payload=_optional_object_dict(mapping.get("payload"), label="event.payload"),
            created_at=created_at,
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryChampionDecision:
    action: FactoryChampionEventAction
    reason: str
    changed: bool
    champion: FactoryCandidate | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryChampionDecision:
        mapping = _require_mapping(payload, label="factory champion decision")
        champion = mapping.get("champion")
        return cls(
            action=FactoryChampionEventAction(
                _require_string(mapping, "action", label="decision.action")
            ),
            reason=_require_string(mapping, "reason", label="decision.reason"),
            changed=bool(_optional_bool(mapping, "changed")),
            champion=FactoryCandidate.from_wire(champion) if champion is not None else None,
            raw=dict(mapping),
        )


# ---------------------------------------------------------------------------
# Result — the public product noun a Factory produces (report, prompt, policy,
# dataset, model, artifact, code change). A Result projects over a WorkProduct;
# evaluation and current-best selection are optional metadata. Legacy candidate
# and champion models above remain for the compatibility window.
# ---------------------------------------------------------------------------


class FactoryResultEvaluationStatus(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class FactoryResultEvaluation:
    """Optional benchmark-owned grading conserved from the candidate record."""

    status: FactoryResultEvaluationStatus = FactoryResultEvaluationStatus.PENDING
    evaluator: str | None = None
    objective: str | None = None
    score: float | None = None
    baseline_score: float | None = None
    verdict: str | None = None
    record: dict[str, object] = field(default_factory=dict)
    evaluated_at: datetime | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryResultEvaluation:
        mapping = _require_mapping(payload, label="factory result evaluation")
        status_value = _optional_string(mapping, "status") or "pending"
        return cls(
            status=FactoryResultEvaluationStatus(status_value),
            evaluator=_optional_string(mapping, "evaluator"),
            objective=_optional_string(mapping, "objective"),
            score=_optional_float(mapping, "score"),
            baseline_score=_optional_float(mapping, "baseline_score"),
            verdict=_optional_string(mapping, "verdict"),
            record=_optional_object_dict(mapping.get("record"), label="evaluation.record"),
            evaluated_at=_optional_datetime(mapping, "evaluated_at"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryResultSelection:
    """Optional current-best selection state for a named objective/scope."""

    is_current_best: bool = False
    scope: str | None = None
    selected_at: datetime | None = None
    reason: str | None = None
    event_id: str | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryResultSelection:
        mapping = _require_mapping(payload, label="factory result selection")
        return cls(
            is_current_best=bool(_optional_bool(mapping, "is_current_best")),
            scope=_optional_string(mapping, "scope"),
            selected_at=_optional_datetime(mapping, "selected_at"),
            reason=_optional_string(mapping, "reason"),
            event_id=_optional_string(mapping, "event_id"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryResultCompatibility:
    """Legacy candidate identity for a candidate-backed Result."""

    candidate_id: str | None = None
    candidate_key: str | None = None
    git_remote: str | None = None
    git_sha: str | None = None
    entrypoint: str | None = None
    is_legacy_projection: bool = False
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryResultCompatibility:
        mapping = _require_mapping(payload, label="factory result compatibility")
        return cls(
            candidate_id=_optional_string(mapping, "candidate_id"),
            candidate_key=_optional_string(mapping, "candidate_key"),
            git_remote=_optional_string(mapping, "git_remote"),
            git_sha=_optional_string(mapping, "git_sha"),
            entrypoint=_optional_string(mapping, "entrypoint"),
            is_legacy_projection=bool(_optional_bool(mapping, "is_legacy_projection")),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryResult:
    result_id: str
    org_id: str
    factory_id: str
    kind: str
    title: str
    status: str
    readiness: str
    created_at: datetime
    updated_at: datetime
    project_id: str | None = None
    effort_id: str | None = None
    run_id: str | None = None
    subtype_kind: str | None = None
    summary: str | None = None
    work_product_id: str | None = None
    content_url: str | None = None
    artifact_id: str | None = None
    artifact_links: tuple[object, ...] = ()
    evaluation: FactoryResultEvaluation | None = None
    selection: FactoryResultSelection | None = None
    compatibility: FactoryResultCompatibility | None = None
    blocker: dict[str, object] | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryResult:
        mapping = _require_mapping(payload, label="factory result")
        created_at = _optional_datetime(mapping, "created_at")
        updated_at = _optional_datetime(mapping, "updated_at")
        if created_at is None or updated_at is None:
            raise ValueError("factory result requires created_at and updated_at")
        evaluation = mapping.get("evaluation")
        selection = mapping.get("selection")
        compatibility = mapping.get("compatibility")
        blocker = mapping.get("blocker")
        return cls(
            result_id=_require_string(mapping, "result_id", label="result.result_id"),
            org_id=_require_string(mapping, "org_id", label="result.org_id"),
            factory_id=_require_string(mapping, "factory_id", label="result.factory_id"),
            project_id=_optional_string(mapping, "project_id"),
            effort_id=_optional_string(mapping, "effort_id"),
            run_id=_optional_string(mapping, "run_id"),
            kind=_require_string(mapping, "kind", label="result.kind"),
            subtype_kind=_optional_string(mapping, "subtype_kind"),
            title=_require_string(mapping, "title", label="result.title"),
            summary=_optional_string(mapping, "summary"),
            status=_require_string(mapping, "status", label="result.status"),
            readiness=_require_string(mapping, "readiness", label="result.readiness"),
            work_product_id=_optional_string(mapping, "work_product_id"),
            content_url=_optional_string(mapping, "content_url"),
            artifact_id=_optional_string(mapping, "artifact_id"),
            artifact_links=tuple(mapping.get("artifact_links") or ()),
            evaluation=(
                FactoryResultEvaluation.from_wire(evaluation)
                if evaluation is not None
                else None
            ),
            selection=(
                FactoryResultSelection.from_wire(selection)
                if selection is not None
                else None
            ),
            compatibility=(
                FactoryResultCompatibility.from_wire(compatibility)
                if compatibility is not None
                else None
            ),
            blocker=(
                _optional_object_dict(blocker, label="result.blocker")
                if blocker is not None
                else None
            ),
            metadata=_optional_object_dict(mapping.get("metadata"), label="result.metadata"),
            created_at=created_at,
            updated_at=updated_at,
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryResultSelectionEvent:
    event_id: str
    org_id: str
    factory_id: str
    action: FactoryChampionEventAction
    created_at: datetime
    effort_id: str | None = None
    result_id: str | None = None
    candidate_id: str | None = None
    scope: str | None = None
    score: float | None = None
    baseline_score: float | None = None
    reason: str | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryResultSelectionEvent:
        mapping = _require_mapping(payload, label="factory result selection event")
        created_at = _optional_datetime(mapping, "created_at")
        if created_at is None:
            raise ValueError("factory result selection event requires created_at")
        return cls(
            event_id=_require_string(mapping, "event_id", label="event.event_id"),
            org_id=_require_string(mapping, "org_id", label="event.org_id"),
            factory_id=_require_string(mapping, "factory_id", label="event.factory_id"),
            effort_id=_optional_string(mapping, "effort_id"),
            result_id=_optional_string(mapping, "result_id"),
            candidate_id=_optional_string(mapping, "candidate_id"),
            action=FactoryChampionEventAction(
                _require_string(mapping, "action", label="event.action")
            ),
            scope=_optional_string(mapping, "scope"),
            score=_optional_float(mapping, "score"),
            baseline_score=_optional_float(mapping, "baseline_score"),
            reason=_optional_string(mapping, "reason"),
            created_at=created_at,
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryResultSelectionDecision:
    action: FactoryChampionEventAction
    reason: str
    changed: bool
    result: FactoryResult | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryResultSelectionDecision:
        mapping = _require_mapping(payload, label="factory result selection decision")
        result = mapping.get("result")
        return cls(
            action=FactoryChampionEventAction(
                _require_string(mapping, "action", label="decision.action")
            ),
            reason=_require_string(mapping, "reason", label="decision.reason"),
            changed=bool(_optional_bool(mapping, "changed")),
            result=FactoryResult.from_wire(result) if result is not None else None,
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryResultEvaluateRequest:
    """Benchmark-owned grading record submitted for one Result (WorkProduct id)."""

    evaluation: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        return {"evaluation": dict(self.evaluation)}


@dataclass(frozen=True)
class FactoryResultSelectRequest:
    result_id: str
    reason: str
    scope: str | None = None
    effort_id: str | None = None

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "result_id": self.result_id,
            "reason": self.reason,
        }
        if self.scope is not None:
            payload["scope"] = self.scope
        if self.effort_id is not None:
            payload["effort_id"] = self.effort_id
        return payload


@dataclass(frozen=True)
class FactoryResultRestoreRequest:
    result_id: str
    reason: str
    scope: str | None = None
    effort_id: str | None = None

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "result_id": self.result_id,
            "reason": self.reason,
        }
        if self.scope is not None:
            payload["scope"] = self.scope
        if self.effort_id is not None:
            payload["effort_id"] = self.effort_id
        return payload


def factory_result_evaluate_payload(
    request: FactoryResultEvaluateRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, FactoryResultEvaluateRequest):
        return request.to_wire()
    return dict(request)


def factory_result_select_payload(
    request: FactoryResultSelectRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, FactoryResultSelectRequest):
        return request.to_wire()
    return dict(request)


def factory_result_restore_payload(
    request: FactoryResultRestoreRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, FactoryResultRestoreRequest):
        return request.to_wire()
    return dict(request)


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


def factory_project_link_payload(
    request: FactoryProjectLinkRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, FactoryProjectLinkRequest):
        return request.to_wire()
    return dict(request)


def factory_project_patch_payload(
    request: FactoryProjectPatchRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, FactoryProjectPatchRequest):
        return request.to_wire()
    return dict(request)


def factory_wake_due_payload(
    request: FactoryWakeDueRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, FactoryWakeDueRequest):
        return request.to_wire()
    return dict(request)


def factory_candidate_grading_payload(
    request: FactoryCandidateGradingRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, FactoryCandidateGradingRequest):
        return request.to_wire()
    return dict(request)


def factory_champion_select_payload(
    request: FactoryChampionSelectRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, FactoryChampionSelectRequest):
        return request.to_wire()
    return dict(request)


def factory_champion_rollback_payload(
    request: FactoryChampionRollbackRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, FactoryChampionRollbackRequest):
        return request.to_wire()
    return dict(request)


def factory_idea_create_payload(
    request: FactoryIdeaCreateRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, FactoryIdeaCreateRequest):
        return request.to_wire()
    return dict(request)


def factory_idea_patch_payload(
    request: FactoryIdeaPatchRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, FactoryIdeaPatchRequest):
        return request.to_wire()
    return dict(request)


def factory_actor_output_create_payload(
    request: FactoryActorOutputCreateRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, FactoryActorOutputCreateRequest):
        return request.to_wire()
    return dict(request)


def factory_actor_output_patch_payload(
    request: FactoryActorOutputPatchRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, FactoryActorOutputPatchRequest):
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


def effort_from_runs_payload(
    request: EffortFromRunsRequest | Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(request, EffortFromRunsRequest):
        return request.to_wire()
    return dict(request)


FACTORY_KIND_VALUES = tuple(item.value for item in FactoryKind)
FACTORY_LIFECYCLE_STATE_VALUES = tuple(item.value for item in FactoryLifecycleState)
FACTORY_PROJECT_ROLE_VALUES = tuple(item.value for item in FactoryProjectRole)
FACTORY_PROJECT_STATUS_VALUES = tuple(item.value for item in FactoryProjectStatus)
FACTORY_IDEA_STATUS_VALUES = tuple(item.value for item in FactoryIdeaStatus)
FACTORY_IDEA_SOURCE_VALUES = tuple(item.value for item in FactoryIdeaSource)
FACTORY_ACTOR_ROLE_VALUES = tuple(item.value for item in FactoryActorRole)
FACTORY_ACTOR_OUTPUT_KIND_VALUES = tuple(item.value for item in FactoryActorOutputKind)
FACTORY_ACTOR_OUTPUT_STATUS_VALUES = tuple(item.value for item in FactoryActorOutputStatus)
EFFORT_STATUS_VALUES = tuple(item.value for item in EffortStatus)
EFFORT_TYPE_VALUES = tuple(item.value for item in EffortType)
FACTORY_RUN_KIND_VALUES = tuple(item.value for item in FactoryRunKind)
FACTORY_CANDIDATE_GRADING_STATUS_VALUES = tuple(
    item.value for item in FactoryCandidateGradingStatus
)
FACTORY_CHAMPION_EVENT_ACTION_VALUES = tuple(item.value for item in FactoryChampionEventAction)


__all__ = [
    "EFFORT_STATUS_VALUES",
    "EFFORT_TYPE_VALUES",
    "FACTORY_KIND_VALUES",
    "FACTORY_LIFECYCLE_STATE_VALUES",
    "FACTORY_PROJECT_ROLE_VALUES",
    "FACTORY_PROJECT_STATUS_VALUES",
    "FACTORY_IDEA_STATUS_VALUES",
    "FACTORY_IDEA_SOURCE_VALUES",
    "FACTORY_ACTOR_ROLE_VALUES",
    "FACTORY_ACTOR_OUTPUT_KIND_VALUES",
    "FACTORY_ACTOR_OUTPUT_STATUS_VALUES",
    "FACTORY_RUN_KIND_VALUES",
    "FACTORY_CANDIDATE_GRADING_STATUS_VALUES",
    "FACTORY_CHAMPION_EVENT_ACTION_VALUES",
    "Effort",
    "EffortCreateRequest",
    "EffortFromRunsRequest",
    "EffortPatchRequest",
    "EffortStatus",
    "EffortType",
    "ExperimentBundle",
    "ExperimentComparison",
    "ExperimentHistory",
    "GraduationProposal",
    "BudgetPolicy",
    "CapPolicy",
    "AuthorizationPolicy",
    "Factory",
    "FactoryActorOutput",
    "FactoryActorOutputCreateRequest",
    "FactoryActorOutputKind",
    "FactoryActorOutputPatchRequest",
    "FactoryActorOutputStatus",
    "FactoryActorRole",
    "FactoryCreateRequest",
    "FactoryCandidate",
    "FactoryCandidateGradingRequest",
    "FactoryCandidateGradingStatus",
    "FactoryChampionDecision",
    "FactoryChampionEvent",
    "FactoryChampionEventAction",
    "FactoryChampionRollbackRequest",
    "FactoryChampionSelectRequest",
    "FactoryResult",
    "FactoryResultCompatibility",
    "FactoryResultEvaluateRequest",
    "FactoryResultEvaluation",
    "FactoryResultEvaluationStatus",
    "FactoryResultRestoreRequest",
    "FactoryResultSelectRequest",
    "FactoryResultSelection",
    "FactoryResultSelectionDecision",
    "FactoryResultSelectionEvent",
    "FactoryHealth",
    "FactoryMaintenanceAction",
    "FactoryOperatingWindow",
    "FactoryControlLoopFlags",
    "FactoryReactorReceipt",
    "FactoryReactorStatus",
    "FactoryRuntimeStatus",
    "FactoryIdea",
    "FactoryIdeaCreateRequest",
    "FactoryIdeaPatchRequest",
    "FactoryIdeaSource",
    "FactoryIdeaStatus",
    "FactoryKind",
    "FactoryLifecycleState",
    "FactoryPatchRequest",
    "FactoryProjectLink",
    "FactoryProjectLinkRequest",
    "FactoryProjectPatchRequest",
    "FactoryProjectRole",
    "FactoryProjectStatus",
    "FactoryProjectSummary",
    "FactoryReportSummary",
    "FactoryRunSummary",
    "FactoryRunKind",
    "FactoryStatus",
    "FactoryWakeDueEffort",
    "FactoryWakeDueRequest",
    "FactoryWakeDueResult",
    "FactoryWorkspace",
    "FactoryWorkProductSummary",
    "PublicationPolicy",
    "RecurrencePolicy",
    "effort_create_payload",
    "effort_from_runs_payload",
    "effort_patch_payload",
    "factory_actor_output_create_payload",
    "factory_actor_output_patch_payload",
    "factory_create_payload",
    "factory_candidate_grading_payload",
    "factory_champion_rollback_payload",
    "factory_result_evaluate_payload",
    "factory_result_restore_payload",
    "factory_result_select_payload",
    "factory_champion_select_payload",
    "factory_idea_create_payload",
    "factory_idea_patch_payload",
    "factory_patch_payload",
    "factory_project_link_payload",
    "factory_project_patch_payload",
    "factory_wake_due_payload",
]
