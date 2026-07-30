"""Typed Factory and Effort contracts for the stable Research API.

# See: testing/specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.sdk.research.contracts._wire import (
    object_value,
    optional_bool,
    optional_datetime,
    optional_text,
    required_bool,
    required_datetime,
    required_text,
)
from synth_ai.sdk.research.contracts.common import (
    EffortId,
    FactoryCandidateId,
    FactoryId,
    OrganizationId,
    ProjectId,
    SwarmId,
    require_text,
)
from synth_ai.sdk.research.contracts.swarms import SwarmSpec


class FactoryKind(StrEnum):
    CUSTOMER = "customer"
    INTERNAL = "internal"


class FactoryLifecycleState(StrEnum):
    CONFIGURED = "configured"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class FactoryCreateState(StrEnum):
    CONFIGURED = "configured"
    ACTIVE = "active"


class FactoryResultAuthorityGeneration(StrEnum):
    """Persistence authority selected when a Factory is created."""

    LEGACY = "legacy"
    RESULT_AUTHORITY = "result_authority"


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


class EffortRunClass(StrEnum):
    ORDINARY = "ordinary"
    TINKER_SFT = "tinker_sft"


class FactoryBudgetPeriod(StrEnum):
    """Accounting window for a Factory's aggregate spend envelope."""

    ALL_TIME = "all_time"
    DAILY = "daily"
    LAST_7D = "last_7d"
    MONTHLY = "monthly"


class FactoryTransitionDecision(StrEnum):
    APPLIED = "applied"
    NOOP = "noop"
    PREVIEW_APPLIED = "preview_applied"


class FactoryCandidateGradingStatus(StrEnum):
    PENDING = "pending"
    GRADED = "graded"
    FAILED = "failed"
    TIMEOUT = "timeout"


class FactoryChampionAction(StrEnum):
    PROMOTE = "promote"
    ROLLBACK = "rollback"
    NO_LIFT = "no_lift"


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
    period: FactoryBudgetPeriod | None = None

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
        if self.period is not None and not isinstance(self.period, FactoryBudgetPeriod):
            object.__setattr__(self, "period", FactoryBudgetPeriod(self.period))
        if self.tinker_sft_runs_per_window is not None and self.tinker_sft_runs_per_window < 0:
            raise ValueError("tinker_sft_runs_per_window must be non-negative")

    def to_wire(self) -> JsonObject:
        value: JsonObject = {
            "factory_limit_usd": self.factory_limit_usd,
            "ordinary_run_limit_usd": self.ordinary_run_limit_usd,
            "ordinary_run_target_usd": self.ordinary_run_target_usd,
        }
        if self.period is not None:
            value["period"] = self.period.value
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


_EFFORT_RECURRENCE_DELAY_SECONDS_MAX = 7 * 24 * 60 * 60
_EFFORT_RECURRENCE_INITIAL_FAILURE_BACKOFF_SECONDS_MAX = 24 * 60 * 60
_EFFORT_RECURRENCE_DEFAULT_FAILURE_BACKOFF_SECONDS = 300


def _bounded_recurrence_seconds(
    value: int | None,
    *,
    field_name: str,
    minimum: int,
    maximum: int = _EFFORT_RECURRENCE_DELAY_SECONDS_MAX,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum} seconds")
    return value


@dataclass(frozen=True, slots=True)
class EffortResearchRecurrencePolicy:
    """Research-lane overrides merged over the root recurrence policy."""

    launch: SwarmSpec | None = None
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.launch is not None and not isinstance(self.launch, SwarmSpec):
            raise ValueError("research launch must be SwarmSpec")

    def to_wire(self) -> JsonObject:
        value = dict(self.metadata)
        if self.launch is not None:
            value["launch_request"] = self.launch.to_wire()
        return value


@dataclass(frozen=True, slots=True)
class EffortMaintenanceRecurrencePolicy:
    """Maintenance cadence and launch overrides for one recurring Effort."""

    enabled: bool | None = None
    every_n_research_runs: int | None = None
    launch: SwarmSpec | None = None
    required_stewards: tuple[str, ...] = ()
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.enabled is not None and not isinstance(self.enabled, bool):
            raise ValueError("maintenance enabled must be a boolean")
        if self.every_n_research_runs is not None and (
            isinstance(self.every_n_research_runs, bool)
            or not isinstance(self.every_n_research_runs, int)
            or self.every_n_research_runs < 1
        ):
            raise ValueError("every_n_research_runs must be a positive integer")
        if self.launch is not None and not isinstance(self.launch, SwarmSpec):
            raise ValueError("maintenance launch must be SwarmSpec")
        normalized_stewards = tuple(
            require_text(
                steward,
                field_name=f"required_stewards[{index}]",
            ).lower()
            for index, steward in enumerate(self.required_stewards)
        )
        invalid_stewards = sorted(set(normalized_stewards).difference({"gardener", "adjudicator"}))
        if invalid_stewards:
            raise ValueError("required_stewards must contain only gardener or adjudicator")
        object.__setattr__(self, "required_stewards", normalized_stewards)

    def to_wire(self) -> JsonObject:
        value = dict(self.metadata)
        if self.enabled is not None:
            value["enabled"] = self.enabled
        if self.every_n_research_runs is not None:
            value["every_n_research_runs"] = self.every_n_research_runs
        if self.launch is not None:
            value["launch_request"] = self.launch.to_wire()
        if self.required_stewards:
            value["required_stewards"] = list(self.required_stewards)
        return value


@dataclass(frozen=True, slots=True, init=False)
class EffortRecurrence:
    """Typed recurring launch policy for one Effort.

    ``launch`` is serialized through :class:`SwarmSpec` without changing its
    provider, profile, model, or role bindings. A scalar provider therefore
    remains a strict provider pin; this layer never adds a fallback.
    """

    cadence: str | None = None
    timezone: str | None = None
    max_active_swarms: int | None = None
    trigger: str | None = None
    on_run_complete: bool | None = None
    event_triggers: tuple[str | JsonObject, ...] = ()
    event_scope: str | None = None
    cooldown_seconds: int | None = None
    success_delay_seconds: int | None = None
    failure_backoff_seconds: int | None = None
    failure_backoff_max_seconds: int | None = None
    failure_backoff_multiplier: float | None = None
    enabled: bool = True
    metadata: JsonObject = field(default_factory=dict)
    launch: SwarmSpec | None = None
    research: EffortResearchRecurrencePolicy | None = None
    maintenance: EffortMaintenanceRecurrencePolicy | None = None

    def __init__(
        self,
        cadence: str | None = None,
        timezone: str | None = None,
        max_active_swarms: int | None = None,
        launch: SwarmSpec | None = None,
        trigger: str | None = None,
        on_run_complete: bool | None = None,
        event_triggers: tuple[str | JsonObject, ...] = (),
        event_scope: str | None = None,
        cooldown_seconds: int | None = None,
        success_delay_seconds: int | None = None,
        failure_backoff_seconds: int | None = None,
        failure_backoff_max_seconds: int | None = None,
        failure_backoff_multiplier: float | None = None,
        enabled: bool = True,
        metadata: JsonObject | None = None,
        research: EffortResearchRecurrencePolicy | None = None,
        maintenance: EffortMaintenanceRecurrencePolicy | None = None,
        max_active_runs: int | None = None,
    ) -> None:
        if (
            max_active_swarms is not None
            and max_active_runs is not None
            and max_active_swarms != max_active_runs
        ):
            raise ValueError("max_active_swarms and max_active_runs must match")
        object.__setattr__(
            self,
            "max_active_swarms",
            max_active_swarms if max_active_swarms is not None else max_active_runs,
        )
        object.__setattr__(self, "cadence", cadence)
        object.__setattr__(self, "timezone", timezone)
        object.__setattr__(self, "trigger", trigger)
        object.__setattr__(self, "on_run_complete", on_run_complete)
        object.__setattr__(self, "event_triggers", tuple(event_triggers))
        object.__setattr__(self, "event_scope", event_scope)
        object.__setattr__(self, "cooldown_seconds", cooldown_seconds)
        object.__setattr__(self, "success_delay_seconds", success_delay_seconds)
        object.__setattr__(self, "failure_backoff_seconds", failure_backoff_seconds)
        object.__setattr__(
            self,
            "failure_backoff_max_seconds",
            failure_backoff_max_seconds,
        )
        object.__setattr__(
            self,
            "failure_backoff_multiplier",
            failure_backoff_multiplier,
        )
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "metadata", dict(metadata or {}))
        object.__setattr__(self, "launch", launch)
        object.__setattr__(self, "research", research)
        object.__setattr__(self, "maintenance", maintenance)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.cadence is not None:
            require_text(self.cadence, field_name="cadence")
        if self.timezone is not None:
            require_text(self.timezone, field_name="timezone")
        if self.trigger is not None:
            require_text(self.trigger, field_name="trigger")
        if self.event_scope is not None:
            require_text(self.event_scope, field_name="event_scope")
        if self.max_active_swarms is not None and self.max_active_swarms < 1:
            raise ValueError("max_active_swarms must be positive")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        if self.on_run_complete is not None and not isinstance(self.on_run_complete, bool):
            raise ValueError("on_run_complete must be a boolean")
        if self.trigger == "on_run_complete" and self.on_run_complete is False:
            raise ValueError("on_run_complete cannot be false when trigger is on_run_complete")
        if self.on_run_complete is True and self.trigger not in {
            None,
            "on_run_complete",
        }:
            raise ValueError("on_run_complete cannot be combined with another trigger")
        for index, event_trigger in enumerate(self.event_triggers):
            if isinstance(event_trigger, str):
                require_text(
                    event_trigger,
                    field_name=f"event_triggers[{index}]",
                )
            elif not isinstance(event_trigger, dict):
                raise ValueError(f"event_triggers[{index}] must be a string or JSON object")
        object.__setattr__(
            self,
            "cooldown_seconds",
            _bounded_recurrence_seconds(
                self.cooldown_seconds,
                field_name="cooldown_seconds",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "success_delay_seconds",
            _bounded_recurrence_seconds(
                self.success_delay_seconds,
                field_name="success_delay_seconds",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "failure_backoff_seconds",
            _bounded_recurrence_seconds(
                self.failure_backoff_seconds,
                field_name="failure_backoff_seconds",
                minimum=1,
                maximum=_EFFORT_RECURRENCE_INITIAL_FAILURE_BACKOFF_SECONDS_MAX,
            ),
        )
        object.__setattr__(
            self,
            "failure_backoff_max_seconds",
            _bounded_recurrence_seconds(
                self.failure_backoff_max_seconds,
                field_name="failure_backoff_max_seconds",
                minimum=1,
            ),
        )
        initial_failure_backoff_seconds = (
            self.failure_backoff_seconds
            if self.failure_backoff_seconds is not None
            else _EFFORT_RECURRENCE_DEFAULT_FAILURE_BACKOFF_SECONDS
        )
        if (
            self.failure_backoff_max_seconds is not None
            and self.failure_backoff_max_seconds < initial_failure_backoff_seconds
        ):
            raise ValueError(
                "failure_backoff_max_seconds must be greater than or equal to "
                "failure_backoff_seconds"
            )
        if self.failure_backoff_multiplier is not None:
            if (
                isinstance(self.failure_backoff_multiplier, bool)
                or not isinstance(self.failure_backoff_multiplier, (int, float))
                or not 1.0 <= float(self.failure_backoff_multiplier) <= 16.0
            ):
                raise ValueError("failure_backoff_multiplier must be between 1 and 16")
            object.__setattr__(
                self,
                "failure_backoff_multiplier",
                float(self.failure_backoff_multiplier),
            )
        if self.launch is not None and not isinstance(self.launch, SwarmSpec):
            raise ValueError("launch must be SwarmSpec")
        if self.research is not None and not isinstance(
            self.research,
            EffortResearchRecurrencePolicy,
        ):
            raise ValueError("research must be EffortResearchRecurrencePolicy")
        if self.maintenance is not None and not isinstance(
            self.maintenance,
            EffortMaintenanceRecurrencePolicy,
        ):
            raise ValueError("maintenance must be EffortMaintenanceRecurrencePolicy")

    @property
    def max_active_runs(self) -> int | None:
        """Compatibility spelling for the backend wire field."""
        return self.max_active_swarms

    def to_wire(self) -> JsonObject:
        value: JsonObject = {"enabled": self.enabled}
        if self.cadence is not None:
            value["cadence"] = self.cadence
        if self.timezone is not None:
            value["timezone"] = self.timezone
        if self.max_active_swarms is not None:
            value["max_active_runs"] = self.max_active_swarms
        if self.trigger is not None:
            value["trigger"] = self.trigger
        if self.on_run_complete is not None:
            value["on_run_complete"] = self.on_run_complete
        if self.event_triggers:
            value["event_triggers"] = [
                dict(item) if isinstance(item, dict) else item for item in self.event_triggers
            ]
        if self.event_scope is not None:
            value["event_scope"] = self.event_scope
        if self.cooldown_seconds is not None:
            value["cooldown_seconds"] = self.cooldown_seconds
        if self.success_delay_seconds is not None:
            value["delay_seconds"] = self.success_delay_seconds
        if (
            self.failure_backoff_seconds is not None
            or self.failure_backoff_max_seconds is not None
            or self.failure_backoff_multiplier is not None
        ):
            failure_policy: JsonObject = {"action": "backoff"}
            if self.failure_backoff_seconds is not None:
                failure_policy["backoff_seconds"] = self.failure_backoff_seconds
            if self.failure_backoff_max_seconds is not None:
                failure_policy["max_backoff_seconds"] = self.failure_backoff_max_seconds
            if self.failure_backoff_multiplier is not None:
                failure_policy["multiplier"] = self.failure_backoff_multiplier
            value["failure_policy"] = failure_policy
        if self.metadata:
            value["metadata"] = dict(self.metadata)
        if self.launch is not None:
            value["launch_request"] = self.launch.to_wire()
        if self.research is not None:
            value["research"] = self.research.to_wire()
        if self.maintenance is not None:
            value["maintenance"] = self.maintenance.to_wire()
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
    result_authority_generation: FactoryResultAuthorityGeneration = (
        FactoryResultAuthorityGeneration.LEGACY
    )

    def __post_init__(self) -> None:
        require_text(self.name, field_name="factory name")

    def to_wire(self) -> JsonObject:
        value: JsonObject = {
            "name": self.name,
            "kind": self.kind.value,
            "status": self.state.value,
            "result_authority_generation": self.result_authority_generation.value,
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
    result_authority_generation: FactoryResultAuthorityGeneration = (
        FactoryResultAuthorityGeneration.LEGACY
    )

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
            result_authority_generation=FactoryResultAuthorityGeneration(
                optional_text(value, "result_authority_generation")
                or FactoryResultAuthorityGeneration.LEGACY.value
            ),
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
class FactoryCandidateGradingRequest:
    """Benchmark-owned grading record for one immutable candidate."""

    grading: JsonObject

    def __post_init__(self) -> None:
        status = str(self.grading.get("status") or "").strip().lower()
        if status != FactoryCandidateGradingStatus.GRADED.value:
            return
        evaluation_mode = str(self.grading.get("evaluation_mode") or "").strip().lower()
        allowed_modes = {"live", "mock", "fixture", "offline_fallback"}
        if not evaluation_mode:
            raise ValueError(
                "grading_evaluation_mode_missing: status 'graded' requires evaluation_mode"
            )
        if evaluation_mode not in allowed_modes:
            raise ValueError(
                "grading_evaluation_mode_invalid: evaluation_mode must be one of "
                f"{sorted(allowed_modes)}"
            )

    def to_wire(self) -> JsonObject:
        return {"grading": dict(self.grading)}


@dataclass(frozen=True, slots=True)
class FactoryChampionSelectRequest:
    baseline_score: float
    effort_id: EffortId | None = None

    def to_wire(self) -> JsonObject:
        value: JsonObject = {"baseline_score": float(self.baseline_score)}
        if self.effort_id is not None:
            value["effort_id"] = self.effort_id
        return value


@dataclass(frozen=True, slots=True)
class FactoryChampionRollbackRequest:
    to_candidate_id: FactoryCandidateId
    reason: str
    effort_id: EffortId | None = None

    def __post_init__(self) -> None:
        require_text(self.to_candidate_id, field_name="to_candidate_id")
        require_text(self.reason, field_name="reason")

    def to_wire(self) -> JsonObject:
        value: JsonObject = {
            "to_candidate_id": self.to_candidate_id,
            "reason": self.reason,
        }
        if self.effort_id is not None:
            value["effort_id"] = self.effort_id
        return value


@dataclass(frozen=True, slots=True)
class FactoryCandidate:
    candidate_id: FactoryCandidateId
    organization_id: OrganizationId
    factory_id: FactoryId
    candidate_key: str
    git_remote: str
    git_sha: str
    entrypoint: str
    execution_contract_version: str
    grading_status: FactoryCandidateGradingStatus
    published_at: datetime
    created_at: datetime
    updated_at: datetime
    project_id: ProjectId | None = None
    effort_id: EffortId | None = None
    swarm_id: SwarmId | None = None
    work_product_id: str | None = None
    parent_candidate_id: FactoryCandidateId | None = None
    grading_target: JsonObject = field(default_factory=dict)
    artifact_ids: tuple[JsonValue, ...] = ()
    grading: JsonObject = field(default_factory=dict)
    held_out_score: float | None = None
    baseline_score: float | None = None
    graded_at: datetime | None = None
    is_champion: bool = False
    champion_selected_at: datetime | None = None
    raw: JsonObject = field(default_factory=dict)

    @property
    def run_id(self) -> SwarmId | None:
        """Temporary wire-name compatibility alias."""
        return self.swarm_id

    @classmethod
    def from_wire(cls, payload: JsonValue) -> FactoryCandidate:
        value = object_value(payload, operation_id="decode_factory_candidate")
        artifact_ids = value.get("artifact_ids")
        if artifact_ids is None:
            artifact_ids = []
        if not isinstance(artifact_ids, list):
            raise ValueError("factory candidate artifact_ids must be an array")
        project_id = optional_text(value, "project_id")
        effort_id = optional_text(value, "effort_id")
        swarm_id = optional_text(value, "run_id")
        parent_candidate_id = optional_text(value, "parent_candidate_id")
        return cls(
            candidate_id=FactoryCandidateId(required_text(value, "candidate_id")),
            organization_id=OrganizationId(required_text(value, "org_id")),
            factory_id=FactoryId(required_text(value, "factory_id")),
            project_id=ProjectId(project_id) if project_id is not None else None,
            effort_id=EffortId(effort_id) if effort_id is not None else None,
            swarm_id=SwarmId(swarm_id) if swarm_id is not None else None,
            work_product_id=optional_text(value, "work_product_id"),
            candidate_key=required_text(value, "candidate_key"),
            git_remote=required_text(value, "git_remote"),
            git_sha=required_text(value, "git_sha"),
            entrypoint=required_text(value, "entrypoint"),
            execution_contract_version=required_text(value, "execution_contract_version"),
            parent_candidate_id=(
                FactoryCandidateId(parent_candidate_id) if parent_candidate_id is not None else None
            ),
            grading_target=_optional_object(
                value,
                "grading_target",
                operation_id="decode_factory_candidate",
            ),
            artifact_ids=tuple(artifact_ids),
            grading_status=FactoryCandidateGradingStatus(required_text(value, "grading_status")),
            grading=_optional_object(
                value,
                "grading",
                operation_id="decode_factory_candidate",
            ),
            held_out_score=_optional_number(value, "held_out_score"),
            baseline_score=_optional_number(value, "baseline_score"),
            graded_at=optional_datetime(value, "graded_at"),
            is_champion=optional_bool(value, "is_champion"),
            champion_selected_at=optional_datetime(value, "champion_selected_at"),
            published_at=required_datetime(value, "published_at"),
            created_at=required_datetime(value, "created_at"),
            updated_at=required_datetime(value, "updated_at"),
            raw=dict(value),
        )


@dataclass(frozen=True, slots=True)
class FactoryChampionEvent:
    event_id: str
    organization_id: OrganizationId
    factory_id: FactoryId
    action: FactoryChampionAction
    created_at: datetime
    effort_id: EffortId | None = None
    candidate_id: FactoryCandidateId | None = None
    git_sha: str | None = None
    held_out_score: float | None = None
    baseline_score: float | None = None
    reason: str | None = None
    payload: JsonObject = field(default_factory=dict)
    raw: JsonObject = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: JsonValue) -> FactoryChampionEvent:
        value = object_value(payload, operation_id="decode_factory_champion_event")
        effort_id = optional_text(value, "effort_id")
        candidate_id = optional_text(value, "candidate_id")
        return cls(
            event_id=required_text(value, "event_id"),
            organization_id=OrganizationId(required_text(value, "org_id")),
            factory_id=FactoryId(required_text(value, "factory_id")),
            effort_id=EffortId(effort_id) if effort_id is not None else None,
            candidate_id=(FactoryCandidateId(candidate_id) if candidate_id is not None else None),
            action=FactoryChampionAction(required_text(value, "action")),
            git_sha=optional_text(value, "git_sha"),
            held_out_score=_optional_number(value, "held_out_score"),
            baseline_score=_optional_number(value, "baseline_score"),
            reason=optional_text(value, "reason"),
            payload=_optional_object(
                value,
                "payload",
                operation_id="decode_factory_champion_event",
            ),
            created_at=required_datetime(value, "created_at"),
            raw=dict(value),
        )


@dataclass(frozen=True, slots=True)
class FactoryChampionDecision:
    action: FactoryChampionAction
    reason: str
    changed: bool
    champion: FactoryCandidate | None = None
    raw: JsonObject = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: JsonValue) -> FactoryChampionDecision:
        value = object_value(payload, operation_id="decode_factory_champion_decision")
        champion = value.get("champion")
        return cls(
            action=FactoryChampionAction(required_text(value, "action")),
            reason=required_text(value, "reason"),
            changed=required_bool(value, "changed"),
            champion=FactoryCandidate.from_wire(champion) if champion is not None else None,
            raw=dict(value),
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


def _optional_number(value: JsonObject, name: str) -> float | None:
    field_value = value.get(name)
    if field_value is None:
        return None
    if isinstance(field_value, bool) or not isinstance(field_value, (int, float)):
        raise ValueError(f"{name} must be a number when provided")
    return float(field_value)


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
    "EffortMaintenanceRecurrencePolicy",
    "EffortResearchRecurrencePolicy",
    "EffortRunClass",
    "EffortSpec",
    "EffortStatus",
    "EffortType",
    "Factory",
    "FactoryCandidate",
    "FactoryCandidateGradingRequest",
    "FactoryCandidateGradingStatus",
    "FactoryChampionAction",
    "FactoryChampionDecision",
    "FactoryChampionEvent",
    "FactoryChampionRollbackRequest",
    "FactoryChampionSelectRequest",
    "FactoryCreateRequest",
    "FactoryCreateState",
    "FactoryBudgetPeriod",
    "FactoryBudgetPolicy",
    "FactoryKind",
    "FactoryLifecycleState",
    "FactoryResultAuthorityGeneration",
    "FactoryPatch",
    "FactoryPatchRequest",
    "FactorySpec",
    "FactoryTransition",
    "FactoryTransitionDecision",
    "FactoryTransitionRequest",
    "FactoryTransitionResponse",
    "FactoryTransitionResult",
]
