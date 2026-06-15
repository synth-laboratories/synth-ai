"""Typed public run-state models mirrored from the backend contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from synth_ai.managed_research.models.smr_host_kinds import SmrHostKind, coerce_smr_host_kind
from synth_ai.managed_research.models.smr_network_topology import (
    SmrNetworkTopology,
    coerce_smr_network_topology,
)
from synth_ai.managed_research.models.smr_providers import (
    ActorResourceCapability,
    ProviderBinding,
    UsageLimit,
    coerce_provider_bindings,
    coerce_usage_limit,
)
from synth_ai.managed_research.models.smr_roles import SmrRoleBindings
from synth_ai.managed_research.models.smr_work_modes import SmrWorkMode, coerce_smr_work_mode


class RunState(StrEnum):
    UNKNOWN = "unknown"
    QUEUED = "queued"
    PLANNING = "planning"
    RUNNING = "running"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    REVIEWER_REQUIRED = "reviewer_required"
    BLOCKED = "blocked"
    PAUSED = "paused"
    FINALIZING = "finalizing"
    DONE = "done"
    FAILED = "failed"
    STOPPED = "stopped"
    CANCELED = "canceled"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_RUN_STATES


ManagedResearchRunState = RunState

_TERMINAL_RUN_STATES = frozenset(
    {
        RunState.DONE,
        RunState.FAILED,
        RunState.STOPPED,
        RunState.CANCELED,
    }
)


class ManagedResearchRunTerminalOutcome(StrEnum):
    DONE = "done"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"
    CANCELED = "canceled"
    TIMED_OUT = "timed_out"


class ManagedResearchRunLivenessPhase(StrEnum):
    ACCEPTED = "accepted"
    BOOTSTRAPPING = "bootstrapping"
    QUEUED = "queued"
    WAITING = "waiting"
    PLANNING = "planning"
    WORKING = "working"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    BLOCKED = "blocked"
    PAUSED = "paused"
    ADMITTED = "admitted"
    RUNTIME_INTENT_PENDING = "runtime_intent_pending"
    STARTUP_PENDING = "startup_pending"
    STARTUP_BLOCKED = "startup_blocked"
    PARTICIPANT_REQUESTED = "participant_requested"
    PARTICIPANT_STARTING = "participant_starting"
    PARTICIPANT_LIVE = "participant_live"
    TASK_RUNNING = "task_running"
    RUNTIME_REGISTERED = "runtime_registered"
    DEGRADED = "degraded"
    FINALIZING = "finalizing"
    TERMINAL = "terminal"
    UNKNOWN = "unknown"


def _require_mapping(payload: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return payload


def _optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string when provided")
    normalized = value.strip()
    return normalized or None


def _require_string(payload: Mapping[str, object], key: str, *, label: str) -> str:
    value = _optional_string(payload, key)
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _optional_bool(payload: Mapping[str, object], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean when provided")
    return value


def _optional_object_dict(payload: object, *, label: str = "metadata") -> dict[str, object]:
    if payload is None:
        return {}
    return dict(_require_mapping(payload, label=label))


def _optional_object_tuple(payload: object, *, label: str) -> tuple[dict[str, object], ...]:
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be an array when provided")
    return tuple(dict(_require_mapping(item, label=f"{label}[]")) for item in payload)


def _parse_state(value: str | None) -> RunState:
    if not value:
        return RunState.UNKNOWN
    try:
        return RunState(value)
    except ValueError:
        return RunState.UNKNOWN


def _parse_terminal_outcome(
    value: str | None,
) -> ManagedResearchRunTerminalOutcome | None:
    if not value:
        return None
    return ManagedResearchRunTerminalOutcome(value)


def _parse_liveness_phase(value: str | None) -> ManagedResearchRunLivenessPhase:
    if not value:
        return ManagedResearchRunLivenessPhase.UNKNOWN
    try:
        return ManagedResearchRunLivenessPhase(value)
    except ValueError:
        return ManagedResearchRunLivenessPhase.UNKNOWN


def _parse_host_kind(value: str | None, *, field_name: str) -> SmrHostKind | None:
    return coerce_smr_host_kind(value, field_name=field_name)


def _parse_work_mode(value: str | None, *, field_name: str) -> SmrWorkMode | None:
    return coerce_smr_work_mode(value, field_name=field_name)


def _parse_provider_bindings(payload: object) -> tuple[ProviderBinding, ...]:
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise ValueError("run.providers must be an array when provided")
    return coerce_provider_bindings(payload, field_name="run.providers")


def _parse_actor_resource_capabilities(payload: object) -> frozenset[ActorResourceCapability]:
    if payload is None:
        return frozenset()
    if not isinstance(payload, list):
        raise ValueError("run.capabilities must be an array when provided")
    capabilities: set[ActorResourceCapability] = set()
    for index, value in enumerate(payload):
        if not isinstance(value, str):
            raise ValueError(f"run.capabilities[{index}] must be a string")
        capabilities.add(ActorResourceCapability(value))
    return frozenset(capabilities)


def _parse_usage_limit(payload: object) -> UsageLimit | None:
    if payload is None:
        return None
    return coerce_usage_limit(_require_mapping(payload, label="run.limit"), field_name="run.limit")


@dataclass(frozen=True)
class ManagedResearchRun:
    run_id: str
    project_id: str
    public_state: RunState
    runbook: str | None = None
    project_alias: str | None = None
    project_kind: str | None = None
    terminal_outcome: ManagedResearchRunTerminalOutcome | None = None
    liveness_phase: ManagedResearchRunLivenessPhase = ManagedResearchRunLivenessPhase.UNKNOWN
    current_phase: str = "unknown"
    phase_history: tuple[dict[str, object], ...] = field(default_factory=tuple)
    status_reason: str | None = None
    projection_authority: str = "backend_public_run_state_projection.v1"
    work_completed: bool = False
    completion_classifier: str | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)
    host_kind: SmrHostKind | None = None
    resolved_host_kind: SmrHostKind | None = None
    work_mode: SmrWorkMode | None = None
    resolved_work_mode: SmrWorkMode | None = None
    network_topology: SmrNetworkTopology | None = None
    network_surfaces: dict[str, object] = field(default_factory=dict)
    providers: tuple[ProviderBinding, ...] = field(default_factory=tuple)
    capabilities: frozenset[ActorResourceCapability] = field(default_factory=frozenset)
    limit: UsageLimit | None = None
    roles: SmrRoleBindings | None = None
    kickoff_contract: dict[str, object] = field(default_factory=dict)
    execution_contract: dict[str, object] = field(default_factory=dict)
    latest_summary_artifact_id: str | None = None
    stop_reason: str | None = None
    stop_reason_message: str | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchRun:
        mapping = _require_mapping(payload, label="managed research run")
        roles: SmrRoleBindings | None = None
        if mapping.get("roles") is not None:
            try:
                roles = SmrRoleBindings.from_wire(mapping.get("roles"))
            except ValueError:
                roles = None
        return cls(
            run_id=_require_string(mapping, "run_id", label="run.run_id"),
            project_id=_require_string(mapping, "project_id", label="run.project_id"),
            runbook=_optional_string(mapping, "runbook"),
            project_alias=_optional_string(mapping, "project_alias"),
            project_kind=_optional_string(mapping, "project_kind"),
            public_state=_parse_state(
                _require_string(mapping, "public_state", label="run.public_state")
            ),
            terminal_outcome=_parse_terminal_outcome(_optional_string(mapping, "terminal_outcome")),
            liveness_phase=_parse_liveness_phase(_optional_string(mapping, "liveness_phase")),
            current_phase=_optional_string(mapping, "current_phase") or "unknown",
            phase_history=_optional_object_tuple(
                mapping.get("phase_history"),
                label="run.phase_history",
            ),
            status_reason=_optional_string(mapping, "status_reason"),
            projection_authority=(
                _optional_string(mapping, "projection_authority")
                or "backend_public_run_state_projection.v1"
            ),
            work_completed=bool(_optional_bool(mapping, "work_completed")),
            completion_classifier=_optional_string(mapping, "completion_classifier"),
            diagnostics=_optional_object_dict(
                mapping.get("diagnostics"),
                label="run.diagnostics",
            ),
            host_kind=_parse_host_kind(
                _optional_string(mapping, "host_kind"),
                field_name="run.host_kind",
            ),
            resolved_host_kind=_parse_host_kind(
                _optional_string(mapping, "resolved_host_kind"),
                field_name="run.resolved_host_kind",
            ),
            work_mode=_parse_work_mode(
                _optional_string(mapping, "work_mode"),
                field_name="run.work_mode",
            ),
            resolved_work_mode=_parse_work_mode(
                _optional_string(mapping, "resolved_work_mode"),
                field_name="run.resolved_work_mode",
            ),
            network_topology=coerce_smr_network_topology(
                _optional_string(mapping, "network_topology"),
                field_name="run.network_topology",
            ),
            network_surfaces=_optional_object_dict(
                mapping.get("network_surfaces"),
                label="run.network_surfaces",
            ),
            providers=_parse_provider_bindings(mapping.get("providers")),
            capabilities=_parse_actor_resource_capabilities(mapping.get("capabilities")),
            limit=_parse_usage_limit(mapping.get("limit")),
            roles=roles,
            kickoff_contract=_optional_object_dict(
                mapping.get("kickoff_contract"),
                label="run.kickoff_contract",
            ),
            execution_contract=_optional_object_dict(
                mapping.get("execution_contract"),
                label="run.execution_contract",
            ),
            latest_summary_artifact_id=_optional_string(mapping, "latest_summary_artifact_id"),
            stop_reason=_optional_string(mapping, "stop_reason"),
            stop_reason_message=_optional_string(mapping, "stop_reason_message"),
            raw=dict(mapping),
        )


__all__ = [
    "ManagedResearchRun",
    "ManagedResearchRunLivenessPhase",
    "RunState",
    "ManagedResearchRunState",
    "ManagedResearchRunTerminalOutcome",
]
