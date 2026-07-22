"""Typed Factory and Effort contracts for the stable Research API.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.research.contracts._wire import (
    object_value,
    optional_bool,
    optional_text,
    required_datetime,
    required_text,
)
from synth_ai.core.research.contracts.common import (
    EffortId,
    FactoryId,
    OrganizationId,
    ProjectId,
    SwarmId,
    require_text,
)
from synth_ai.core.research.contracts.swarms import SwarmSpec


class FactoryKind(StrEnum):
    CUSTOMER = "customer"
    INTERNAL = "internal"
    OPEN_RESEARCH = "open_research"


class FactoryLifecycleState(StrEnum):
    CONFIGURED = "configured"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class FactoryCreateState(StrEnum):
    CONFIGURED = "configured"
    ACTIVE = "active"


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


class EffortRunClass(StrEnum):
    ORDINARY = "ordinary"
    TINKER_SFT = "tinker_sft"


class FactoryTransitionDecision(StrEnum):
    APPLIED = "applied"
    NOOP = "noop"
    PREVIEW_APPLIED = "preview_applied"


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """Stable customer budget controls."""

    limit_usd: float | None = None
    period: str | None = None

    def __post_init__(self) -> None:
        if self.limit_usd is not None and self.limit_usd < 0:
            raise ValueError("limit_usd must be non-negative")
        if self.period is not None:
            require_text(self.period, field_name="period")

    def to_wire(self) -> JsonObject:
        value: JsonObject = {}
        if self.limit_usd is not None:
            value["limit"] = self.limit_usd
            value["currency"] = "USD"
        if self.period is not None:
            value["period"] = self.period
        return value


@dataclass(frozen=True, slots=True)
class FactoryBudgetPolicy:
    """Factory-owned budget envelope and per-swarm admission limits."""

    factory_limit_usd: float
    ordinary_run_limit_usd: float
    ordinary_run_target_usd: float
    tinker_sft_run_limit_usd: float | None = None
    tinker_sft_runs_per_window: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "factory_limit_usd",
            "ordinary_run_limit_usd",
            "ordinary_run_target_usd",
            "tinker_sft_run_limit_usd",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.tinker_sft_runs_per_window is not None and self.tinker_sft_runs_per_window < 0:
            raise ValueError("tinker_sft_runs_per_window must be non-negative")

    def to_wire(self) -> JsonObject:
        value: JsonObject = {
            "factory_limit_usd": self.factory_limit_usd,
            "ordinary_run_limit_usd": self.ordinary_run_limit_usd,
            "ordinary_run_target_usd": self.ordinary_run_target_usd,
        }
        if self.tinker_sft_run_limit_usd is not None:
            value["tinker_sft_run_limit_usd"] = self.tinker_sft_run_limit_usd
        if self.tinker_sft_runs_per_window is not None:
            value["tinker_sft_runs_per_window"] = self.tinker_sft_runs_per_window
        return value


@dataclass(frozen=True, slots=True)
class EffortBudgetPolicy:
    """Admission class and human-approval posture for one Effort."""

    run_class: EffortRunClass
    operator_approved: bool

    def to_wire(self) -> JsonObject:
        return {
            "run_class": self.run_class.value,
            "operator_approved": self.operator_approved,
        }


@dataclass(frozen=True, slots=True)
class CapacityPolicy:
    max_active_efforts: int | None = None
    max_active_swarms: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_active_efforts", self.max_active_efforts),
            ("max_active_swarms", self.max_active_swarms),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")

    def to_wire(self) -> JsonObject:
        value: JsonObject = {}
        if self.max_active_efforts is not None:
            value["max_active_efforts"] = self.max_active_efforts
        if self.max_active_swarms is not None:
            value["max_active_runs"] = self.max_active_swarms
        return value


@dataclass(frozen=True, slots=True)
class EffortRecurrence:
    """Typed recurring launch policy for one Effort."""

    cadence: str | None = None
    timezone: str | None = None
    max_active_swarms: int | None = None
    launch: SwarmSpec | None = None

    def __post_init__(self) -> None:
        if self.cadence is not None:
            require_text(self.cadence, field_name="cadence")
        if self.timezone is not None:
            require_text(self.timezone, field_name="timezone")
        if self.max_active_swarms is not None and self.max_active_swarms < 1:
            raise ValueError("max_active_swarms must be positive")

    def to_wire(self) -> JsonObject:
        value: JsonObject = {}
        if self.cadence is not None:
            value["cadence"] = self.cadence
        if self.timezone is not None:
            value["timezone"] = self.timezone
        if self.max_active_swarms is not None:
            value["max_active_runs"] = self.max_active_swarms
        if self.launch is not None:
            value["launch_request"] = self.launch.to_wire()
        return value


@dataclass(frozen=True, slots=True)
class FactorySpec:
    name: str
    description: str | None = None
    kind: FactoryKind = FactoryKind.CUSTOMER
    state: FactoryCreateState = FactoryCreateState.ACTIVE
    budget: BudgetPolicy | FactoryBudgetPolicy | None = None
    capacity: CapacityPolicy | None = None
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_text(self.name, field_name="factory name")

    def to_wire(self) -> JsonObject:
        value: JsonObject = {
            "name": self.name,
            "kind": self.kind.value,
            "status": self.state.value,
            "budget_policy": self.budget.to_wire() if self.budget is not None else {},
            "cap_policy": self.capacity.to_wire() if self.capacity is not None else {},
            "homeostasis_policy": {},
            "publication_policy": {},
            "authorization_policy": {},
            "metadata": dict(self.metadata),
        }
        if self.description is not None:
            value["description"] = self.description
        return value


@dataclass(frozen=True, slots=True)
class FactoryPatch:
    name: str | None = None
    description: str | None = None
    budget: BudgetPolicy | None = None
    capacity: CapacityPolicy | None = None
    metadata: JsonObject | None = None

    def __post_init__(self) -> None:
        if self.name is not None:
            require_text(self.name, field_name="factory name")
        if all(
            value is None
            for value in (
                self.name,
                self.description,
                self.budget,
                self.capacity,
                self.metadata,
            )
        ):
            raise ValueError("factory patch must change at least one field")

    def to_wire(self) -> JsonObject:
        value: JsonObject = {}
        if self.name is not None:
            value["name"] = self.name
        if self.description is not None:
            value["description"] = self.description
        if self.budget is not None:
            value["budget_policy"] = self.budget.to_wire()
        if self.capacity is not None:
            value["cap_policy"] = self.capacity.to_wire()
        if self.metadata is not None:
            value["metadata"] = dict(self.metadata)
        return value


@dataclass(frozen=True, slots=True)
class FactoryTransition:
    reason: str | None = None
    preview: bool = False

    def __post_init__(self) -> None:
        if self.reason is not None:
            require_text(self.reason, field_name="transition reason")

    def to_wire(self) -> JsonObject:
        value: JsonObject = {"dry_run": self.preview}
        if self.reason is not None:
            value["reason"] = self.reason
        return value


@dataclass(frozen=True, slots=True)
class Factory:
    factory_id: FactoryId
    organization_id: OrganizationId
    name: str
    kind: FactoryKind
    state: FactoryLifecycleState
    created_at: datetime
    updated_at: datetime
    description: str | None = None
    budget_policy: JsonObject = field(default_factory=dict)
    capacity_policy: JsonObject = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: JsonValue) -> Factory:
        value = object_value(payload, operation_id="decode_factory")
        return cls(
            factory_id=FactoryId(required_text(value, "factory_id")),
            organization_id=OrganizationId(required_text(value, "org_id")),
            name=required_text(value, "name"),
            description=optional_text(value, "description"),
            kind=FactoryKind(required_text(value, "kind")),
            state=FactoryLifecycleState(required_text(value, "status")),
            budget_policy=_optional_object(value, "budget_policy", operation_id="decode_factory"),
            capacity_policy=_optional_object(value, "cap_policy", operation_id="decode_factory"),
            metadata=_optional_object(value, "metadata", operation_id="decode_factory"),
            created_at=required_datetime(value, "created_at"),
            updated_at=required_datetime(value, "updated_at"),
        )


@dataclass(frozen=True, slots=True)
class FactoryTransitionResult:
    factory: Factory
    command: str
    decision: FactoryTransitionDecision
    state: FactoryLifecycleState
    detail: str | None = None
    effects: tuple[str, ...] = ()
    woken_effort_count: int = 0

    @classmethod
    def from_wire(cls, payload: JsonValue) -> FactoryTransitionResult:
        value = object_value(payload, operation_id="decode_factory_transition")
        effects = value.get("effects")
        if not isinstance(effects, list) or not all(isinstance(item, str) for item in effects):
            raise ValueError("factory transition effects must be a string array")
        wake_count = value.get("woken_efforts", 0)
        if not isinstance(wake_count, int) or isinstance(wake_count, bool):
            raise ValueError("factory transition woken_efforts must be an integer")
        return cls(
            factory=Factory.from_wire(value.get("factory")),
            command=required_text(value, "command"),
            decision=FactoryTransitionDecision(required_text(value, "decision")),
            detail=optional_text(value, "decision_detail"),
            state=FactoryLifecycleState(required_text(value, "to_status")),
            effects=tuple(effects),
            woken_effort_count=wake_count,
        )


@dataclass(frozen=True, slots=True)
class EffortSpec:
    factory_id: FactoryId
    project_id: ProjectId
    name: str
    hypothesis: str | None = None
    state: EffortStatus = EffortStatus.ACTIVE
    effort_type: EffortType = EffortType.RESEARCH
    recurrence: EffortRecurrence | None = None
    next_wake_at: datetime | None = None
    decision_needed: bool = False
    decision_note: str | None = None
    budget: BudgetPolicy | EffortBudgetPolicy | None = None
    actor_notes: JsonObject = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_text(self.factory_id, field_name="factory_id")
        require_text(self.project_id, field_name="project_id")
        require_text(self.name, field_name="effort name")

    def to_wire(self) -> JsonObject:
        value: JsonObject = {
            "factory_id": self.factory_id,
            "project_id": self.project_id,
            "allow_implicit_project_link": True,
            "name": self.name,
            "status": self.state.value,
            "effort_type": self.effort_type.value,
            "recurrence_policy": self.recurrence.to_wire() if self.recurrence is not None else {},
            "decision_needed": self.decision_needed,
            "budget_policy": self.budget.to_wire() if self.budget is not None else {},
            "publication_policy": {},
            "authorization_policy": {},
            "actor_notes": dict(self.actor_notes),
            "metadata": dict(self.metadata),
        }
        if self.hypothesis is not None:
            value["hypothesis_or_topic"] = self.hypothesis
        if self.next_wake_at is not None:
            value["next_wake_at"] = self.next_wake_at.isoformat()
        if self.decision_note is not None:
            value["decision_note"] = self.decision_note
        return value


@dataclass(frozen=True, slots=True)
class EffortPatch:
    name: str | None = None
    hypothesis: str | None = None
    state: EffortStatus | None = None
    recurrence: EffortRecurrence | None = None
    next_wake_at: datetime | None = None
    decision_needed: bool | None = None
    decision_note: str | None = None
    budget: BudgetPolicy | None = None
    metadata: JsonObject | None = None

    def to_wire(self) -> JsonObject:
        value: JsonObject = {}
        for name, field_value in (
            ("name", self.name),
            ("hypothesis_or_topic", self.hypothesis),
            ("decision_needed", self.decision_needed),
            ("decision_note", self.decision_note),
        ):
            if field_value is not None:
                value[name] = field_value
        if self.state is not None:
            value["status"] = self.state.value
        if self.recurrence is not None:
            value["recurrence_policy"] = self.recurrence.to_wire()
        if self.next_wake_at is not None:
            value["next_wake_at"] = self.next_wake_at.isoformat()
        if self.budget is not None:
            value["budget_policy"] = self.budget.to_wire()
        if self.metadata is not None:
            value["metadata"] = dict(self.metadata)
        if not value:
            raise ValueError("effort patch must change at least one field")
        return value


@dataclass(frozen=True, slots=True)
class Effort:
    effort_id: EffortId
    organization_id: OrganizationId
    factory_id: FactoryId
    project_id: ProjectId
    name: str
    state: EffortStatus
    effort_type: EffortType
    created_at: datetime
    updated_at: datetime
    hypothesis: str | None = None
    recurrence_policy: JsonObject = field(default_factory=dict)
    next_wake_at: datetime | None = None
    latest_swarm_id: SwarmId | None = None
    latest_report_id: str | None = None
    latest_work_product_id: str | None = None
    decision_needed: bool = False
    decision_note: str | None = None
    budget_policy: JsonObject = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: JsonValue) -> Effort:
        value = object_value(payload, operation_id="decode_effort")
        next_wake = value.get("next_wake_at")
        if next_wake is not None and not isinstance(next_wake, str):
            raise ValueError("effort next_wake_at must be an ISO-8601 string or null")
        latest_swarm = optional_text(value, "latest_run_id")
        return cls(
            effort_id=EffortId(required_text(value, "effort_id")),
            organization_id=OrganizationId(required_text(value, "org_id")),
            factory_id=FactoryId(required_text(value, "factory_id")),
            project_id=ProjectId(required_text(value, "project_id")),
            name=required_text(value, "name"),
            hypothesis=optional_text(value, "hypothesis_or_topic"),
            state=EffortStatus(required_text(value, "status")),
            effort_type=EffortType(required_text(value, "effort_type")),
            recurrence_policy=_optional_object(
                value, "recurrence_policy", operation_id="decode_effort"
            ),
            next_wake_at=(
                datetime.fromisoformat(next_wake.replace("Z", "+00:00"))
                if next_wake is not None
                else None
            ),
            latest_swarm_id=SwarmId(latest_swarm) if latest_swarm is not None else None,
            latest_report_id=optional_text(value, "latest_report_id"),
            latest_work_product_id=optional_text(
                value,
                "latest_work_product_id",
            ),
            decision_needed=optional_bool(value, "decision_needed") or False,
            decision_note=optional_text(value, "decision_note"),
            budget_policy=_optional_object(value, "budget_policy", operation_id="decode_effort"),
            metadata=_optional_object(value, "metadata", operation_id="decode_effort"),
            created_at=required_datetime(value, "created_at"),
            updated_at=required_datetime(value, "updated_at"),
        )


def _optional_object(value: JsonObject, name: str, *, operation_id: str) -> JsonObject:
    field_value = value.get(name)
    if field_value is None:
        return {}
    return object_value(field_value, operation_id=f"{operation_id}.{name}")


# Compatibility names remain exact aliases through the declared removal window.
FactoryCreateRequest = FactorySpec
FactoryPatchRequest = FactoryPatch
FactoryTransitionRequest = FactoryTransition
FactoryTransitionResponse = FactoryTransitionResult
EffortCreateRequest = EffortSpec
EffortPatchRequest = EffortPatch


__all__ = [
    "BudgetPolicy",
    "CapacityPolicy",
    "EffortBudgetPolicy",
    "Effort",
    "EffortCreateRequest",
    "EffortPatch",
    "EffortPatchRequest",
    "EffortRecurrence",
    "EffortRunClass",
    "EffortSpec",
    "EffortStatus",
    "EffortType",
    "Factory",
    "FactoryCreateRequest",
    "FactoryCreateState",
    "FactoryBudgetPolicy",
    "FactoryKind",
    "FactoryLifecycleState",
    "FactoryPatch",
    "FactoryPatchRequest",
    "FactorySpec",
    "FactoryTransition",
    "FactoryTransitionDecision",
    "FactoryTransitionRequest",
    "FactoryTransitionResponse",
    "FactoryTransitionResult",
]
