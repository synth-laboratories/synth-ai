"""Public Managed Research failure vocabulary and wire contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ActorFailureReason(StrEnum):
    ACTOR_STARTUP_RETRY_BUDGET_EXHAUSTED = "actor_startup_retry_budget_exhausted"
    TASK_BLOCKED = "task_blocked"
    TASK_FAILED = "task_failed"
    RUN_FAILED = "run_failed"
    ORCHESTRATOR_SESSION_START_FAILED = "orchestrator_session_start_failed"
    ORCHESTRATOR_SESSION_FAILED = "orchestrator_session_failed"
    TASK_CLAIM_MATERIALIZATION_FAILED = "task_claim_materialization_failed"
    CLAIMED_ACTOR_MISSING_TASK_BINDING = "claimed_actor_missing_task_binding"
    WORKER_HOST_CLAIMED_NON_TASK_ACTOR = "worker_host_claimed_non_task_actor"
    WORKER_HOST_ACTOR_RUN_MISSING = "worker_host_actor_run_missing"
    WORKER_HOST_CLAIMED_ACTOR_TASK_MISSING = "worker_host_claimed_actor_task_missing"
    TASK_ALREADY_TERMINAL = "task_already_terminal"
    DB_POOL_PRESSURE = "db_pool_pressure"
    CONTROL_PLANE_PRESSURE = "control_plane_pressure"
    PARTICIPANT_START_BACKPRESSURE = "participant_start_backpressure"
    PARTICIPANT_SYNC_DEFERRED_UNCONSUMED = "participant_sync_deferred_unconsumed"
    TERMINAL_TRANSITION_DEFERRED_UNCONSUMED = "terminal_transition_deferred_unconsumed"
    PARTICIPANT_LIVE_CAPACITY_EXHAUSTED = "participant_live_capacity_exhausted"
    PARTICIPANT_MEMORY_CAPACITY_EXHAUSTED = "participant_memory_capacity_exhausted"
    EXTERNAL_IO_DEFERRED = "external_io_deferred"
    OPENCODE_ENDPOINT_UNREACHABLE = "opencode_endpoint_unreachable"
    OPENCODE_ENDPOINT_REFUSED = "opencode_endpoint_refused"
    OPENCODE_ENDPOINT_LOST = "opencode_endpoint_lost"
    OPENCODE_ENDPOINT_HEALTH_TIMEOUT = "opencode_endpoint_health_timeout"
    OPENCODE_CONTAINER_OOM = "opencode_container_oom"
    OPENCODE_EVENT_STREAM_ROUTE_UNREACHABLE = "opencode_event_stream_route_unreachable"
    OPENCODE_EVENT_STREAM_ENDPOINT_REFUSED = "opencode_event_stream_endpoint_refused"
    OPENCODE_EVENT_STREAM_READ_TIMEOUT = "opencode_event_stream_read_timeout"
    OPENCODE_EVENT_STREAM_URL_ERROR = "opencode_event_stream_url_error"
    OPENCODE_EVENT_STREAM_RECONNECT_GRACE_EXHAUSTED = "opencode_event_stream_reconnect_grace_exhausted"
    PARTICIPANT_SESSION_MISSING = "participant_session_missing"
    PARTICIPANT_SESSION_START_FAILED = "participant_session_start_failed"
    PARTICIPANT_LIVE_UNBOUND = "participant_live_unbound"
    PARTICIPANT_LIVE_HANDLE_MISSING = "participant_live_handle_missing"
    PARTICIPANT_BINDING_FAILED = "participant_binding_failed"
    CLAIMED_TASK_WITHOUT_PARTICIPANT = "claimed_task_without_participant"
    STALE_CLAIM_REAPED = "stale_claim_reaped"
    RUNTIME_INTENT_BACKLOG = "runtime_intent_backlog"
    UNKNOWN_UNHEALTHY_EXPIRED = "unknown_unhealthy_expired"
    RUNTIME_NO_PRODUCTIVE_PROGRESS = "runtime_no_productive_progress"


class ManagedResearchFailureFamily(StrEnum):
    UNKNOWN = "unknown"
    VALIDATION = "validation"
    BOOTSTRAP = "bootstrap"
    DISPATCH = "dispatch"
    RUNTIME = "runtime"
    RUNTIME_RECOVERY = "runtime_recovery"
    TERMINALIZATION = "terminalization"
    CLEANUP = "cleanup"
    INFRASTRUCTURE = "infrastructure"
    USAGE_CAPTURE = "usage_capture"
    PARTICIPANT_START = "participant_start"
    ORCHESTRATOR_CYCLE = "orchestrator_cycle"
    ORCHESTRATOR_PROVIDER = "orchestrator_provider"
    ORCHESTRATOR_CODEX = "orchestrator_codex"


class ManagedResearchFailureSeverity(StrEnum):
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class ManagedResearchFailureScope(StrEnum):
    ACTOR = "actor"
    TASK = "task"
    RUN = "run"


@dataclass(frozen=True)
class ManagedResearchFailureClassification:
    family: ManagedResearchFailureFamily
    code: str
    severity: ManagedResearchFailureSeverity
    retryable: bool
    scope: ManagedResearchFailureScope | None = None
    operator_action: str | None = None
    detail: str | None = None
    actor_id: str | None = None
    actor_key: str | None = None
    task_id: str | None = None
    task_key: str | None = None
    recorded_at: str | None = None
    family_raw: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchFailureClassification:
        if not isinstance(payload, dict):
            raise ValueError("run failure classification must be an object")
        family = payload.get("family")
        code = payload.get("code")
        severity = payload.get("severity")
        retryable = payload.get("retryable")
        if not isinstance(family, str) or not isinstance(code, str):
            raise ValueError("run failure classification requires family and code")
        if not isinstance(severity, str) or not isinstance(retryable, bool):
            raise ValueError("run failure classification requires severity and retryable")
        try:
            parsed_family = ManagedResearchFailureFamily(family)
        except ValueError:
            parsed_family = ManagedResearchFailureFamily.UNKNOWN
        return cls(
            family=parsed_family,
            code=code,
            severity=ManagedResearchFailureSeverity(severity),
            retryable=retryable,
            scope=(
                ManagedResearchFailureScope(payload["scope"])
                if isinstance(payload.get("scope"), str)
                else None
            ),
            operator_action=payload.get("operator_action") if isinstance(payload.get("operator_action"), str) else None,
            detail=payload.get("detail") if isinstance(payload.get("detail"), str) else None,
            actor_id=payload.get("actor_id") if isinstance(payload.get("actor_id"), str) else None,
            actor_key=payload.get("actor_key") if isinstance(payload.get("actor_key"), str) else None,
            task_id=payload.get("task_id") if isinstance(payload.get("task_id"), str) else None,
            task_key=payload.get("task_key") if isinstance(payload.get("task_key"), str) else None,
            recorded_at=payload.get("recorded_at") if isinstance(payload.get("recorded_at"), str) else None,
            family_raw=family if parsed_family is ManagedResearchFailureFamily.UNKNOWN else None,
            raw=dict(payload),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                **self.raw,
                "family": self.family.value,
                "code": self.code,
                "severity": self.severity.value,
                "retryable": self.retryable,
                "scope": self.scope.value if self.scope is not None else None,
                "operator_action": self.operator_action,
                "detail": self.detail,
                "actor_id": self.actor_id,
                "actor_key": self.actor_key,
                "task_id": self.task_id,
                "task_key": self.task_key,
                "recorded_at": self.recorded_at,
                "family_raw": self.family_raw,
            }.items()
            if value is not None
        }


__all__ = [
    "ActorFailureReason",
    "ManagedResearchFailureClassification",
    "ManagedResearchFailureFamily",
    "ManagedResearchFailureScope",
    "ManagedResearchFailureSeverity",
]
