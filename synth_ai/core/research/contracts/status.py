"""Strict swarm status projection contracts.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.research.contracts._wire import (
    array_value,
    object_value,
    optional_text,
    required_bool,
    required_datetime,
    required_text,
)
from synth_ai.core.research.contracts.common import ProjectId, SwarmId, require_text


def _exact(value: JsonValue, *, label: str, fields: frozenset[str]) -> JsonObject:
    payload = object_value(value, operation_id=label)
    missing = fields - payload.keys()
    extra = payload.keys() - fields
    if missing or extra:
        raise ValueError(
            f"{label} fields drifted: missing={sorted(missing)!r} extra={sorted(extra)!r}"
        )
    return payload


def _optional_datetime(payload: JsonObject, name: str) -> datetime | None:
    if payload.get(name) is None:
        return None
    return required_datetime(payload, name)


def _non_negative_int(payload: JsonObject, name: str) -> int:
    value = payload.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _int_map(value: JsonValue, *, label: str) -> dict[str, int]:
    payload = object_value(value, operation_id=label)
    result: dict[str, int] = {}
    for key, item in payload.items():
        if type(item) is not int or item < 0:
            raise ValueError(f"{label}.{key} must be a non-negative integer")
        result[str(key)] = item
    return result


class SwarmStatusIssueKind(StrEnum):
    BLOCKER = "blocker"
    FAILURE = "failure"
    PROVIDER_INCIDENT = "provider_incident"
    RESOURCE_INCIDENT = "resource_incident"
    PARTICIPANT_START_FAILURE = "participant_start_failure"
    RECOVERY = "recovery"
    RUNTIME_INVARIANT = "runtime_invariant"


@dataclass(frozen=True, slots=True)
class SwarmStatusState:
    public_state: str
    reason: str | None = None
    blocker_code: str | None = None

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmStatusState:
        payload = _exact(
            value,
            label="swarm status state",
            fields=frozenset({"public_state", "reason", "blocker_code"}),
        )
        return cls(
            require_text(required_text(payload, "public_state"), field_name="public_state"),
            optional_text(payload, "reason"),
            optional_text(payload, "blocker_code"),
        )


@dataclass(frozen=True, slots=True)
class SwarmStatusLiveness:
    phase: str
    task_counts: dict[str, int]
    participant_queue_counts: dict[str, int]

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmStatusLiveness:
        payload = _exact(
            value,
            label="swarm status liveness",
            fields=frozenset({"phase", "task_counts", "participant_queue_counts"}),
        )
        return cls(
            require_text(required_text(payload, "phase"), field_name="phase"),
            _int_map(payload["task_counts"], label="task_counts"),
            _int_map(
                payload["participant_queue_counts"],
                label="participant_queue_counts",
            ),
        )


@dataclass(frozen=True, slots=True)
class SwarmStatusTerminal:
    run_state: str | None
    run_state_is_terminal: bool
    contract_status: str | None
    contract_reason: str | None
    outcome: str | None

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmStatusTerminal:
        payload = _exact(
            value,
            label="swarm status terminal",
            fields=frozenset(
                {
                    "run_state",
                    "run_state_is_terminal",
                    "contract_status",
                    "contract_reason",
                    "outcome",
                }
            ),
        )
        return cls(
            optional_text(payload, "run_state"),
            required_bool(payload, "run_state_is_terminal"),
            optional_text(payload, "contract_status"),
            optional_text(payload, "contract_reason"),
            optional_text(payload, "outcome"),
        )


@dataclass(frozen=True, slots=True)
class SwarmStatusFinalization:
    status: str | None
    proof_status: str | None
    proof_satisfied: bool
    blocker_code: str | None
    reason: str | None

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmStatusFinalization:
        payload = _exact(
            value,
            label="swarm status finalization",
            fields=frozenset(
                {
                    "status",
                    "proof_status",
                    "proof_satisfied",
                    "blocker_code",
                    "reason",
                }
            ),
        )
        return cls(
            optional_text(payload, "status"),
            optional_text(payload, "proof_status"),
            required_bool(payload, "proof_satisfied"),
            optional_text(payload, "blocker_code"),
            optional_text(payload, "reason"),
        )


@dataclass(frozen=True, slots=True)
class SwarmStatusRecovery:
    blocked: bool
    codes: tuple[str, ...]
    incident_count: int
    latest_recorded_at: datetime | None

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmStatusRecovery:
        payload = _exact(
            value,
            label="swarm status recovery",
            fields=frozenset({"blocked", "codes", "incident_count", "latest_recorded_at"}),
        )
        codes = tuple(
            required_text({"recovery_code": item}, "recovery_code")
            for item in array_value(payload["codes"], operation_id="recovery.codes")
        )
        return cls(
            required_bool(payload, "blocked"),
            codes,
            _non_negative_int(payload, "incident_count"),
            _optional_datetime(payload, "latest_recorded_at"),
        )


@dataclass(frozen=True, slots=True)
class SwarmStatusProgress:
    last_progress_at: datetime | None
    last_progress_kind: str | None

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmStatusProgress:
        payload = _exact(
            value,
            label="swarm status progress",
            fields=frozenset({"last_progress_at", "last_progress_kind"}),
        )
        return cls(
            _optional_datetime(payload, "last_progress_at"),
            optional_text(payload, "last_progress_kind"),
        )


@dataclass(frozen=True, slots=True)
class SwarmStatusIssue:
    kind: SwarmStatusIssueKind
    code: str
    source: str | None = None
    status: str | None = None
    severity: str | None = None
    summary: str | None = None
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmStatusIssue:
        payload = _exact(
            value,
            label="swarm status issue",
            fields=frozenset(
                {
                    "kind",
                    "code",
                    "source",
                    "status",
                    "severity",
                    "summary",
                    "first_seen_at",
                    "last_seen_at",
                }
            ),
        )
        return cls(
            SwarmStatusIssueKind(required_text(payload, "kind")),
            require_text(required_text(payload, "code"), field_name="issue code"),
            optional_text(payload, "source"),
            optional_text(payload, "status"),
            optional_text(payload, "severity"),
            optional_text(payload, "summary"),
            _optional_datetime(payload, "first_seen_at"),
            _optional_datetime(payload, "last_seen_at"),
        )


@dataclass(frozen=True, slots=True)
class SwarmStatusInvariant:
    code: str
    source: str | None = None
    message: str | None = None

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmStatusInvariant:
        payload = _exact(
            value,
            label="swarm status invariant",
            fields=frozenset({"code", "source", "message"}),
        )
        return cls(
            require_text(required_text(payload, "code"), field_name="invariant code"),
            optional_text(payload, "source"),
            optional_text(payload, "message"),
        )


@dataclass(frozen=True, slots=True)
class SwarmStatusIssues:
    issues: tuple[SwarmStatusIssue, ...]
    invariants: tuple[SwarmStatusInvariant, ...]

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmStatusIssues:
        payload = _exact(
            value,
            label="swarm status issues",
            fields=frozenset({"issues", "invariants"}),
        )
        return cls(
            tuple(
                SwarmStatusIssue.from_wire(item)
                for item in array_value(payload["issues"], operation_id="issues")
            ),
            tuple(
                SwarmStatusInvariant.from_wire(item)
                for item in array_value(payload["invariants"], operation_id="invariants")
            ),
        )


@dataclass(frozen=True, slots=True)
class SwarmStatusFailure:
    code: str | None
    source: str | None
    classification: str | None
    terminal: bool

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmStatusFailure:
        payload = _exact(
            value,
            label="swarm status failure",
            fields=frozenset({"code", "source", "classification", "terminal"}),
        )
        return cls(
            optional_text(payload, "code"),
            optional_text(payload, "source"),
            optional_text(payload, "classification"),
            required_bool(payload, "terminal"),
        )


@dataclass(frozen=True, slots=True)
class SwarmStatusFreshness:
    projection_authority: str
    last_authoritative_update_at: datetime
    generated_at: datetime

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmStatusFreshness:
        payload = _exact(
            value,
            label="swarm status freshness",
            fields=frozenset(
                {
                    "projection_authority",
                    "last_authoritative_update_at",
                    "generated_at",
                }
            ),
        )
        authority = required_text(payload, "projection_authority")
        if authority != "smr_run_status_projection.v1":
            raise ValueError(f"unsupported status projection authority {authority!r}")
        return cls(
            authority,
            required_datetime(payload, "last_authoritative_update_at"),
            required_datetime(payload, "generated_at"),
        )


@dataclass(frozen=True, slots=True)
class SwarmStatus:
    """Cheap authoritative swarm status without actors/tasks/messages."""

    swarm_id: SwarmId
    project_id: ProjectId
    state: SwarmStatusState
    liveness: SwarmStatusLiveness
    terminal: SwarmStatusTerminal
    finalization: SwarmStatusFinalization
    recovery: SwarmStatusRecovery
    progress: SwarmStatusProgress
    issues: SwarmStatusIssues
    failure: SwarmStatusFailure
    freshness: SwarmStatusFreshness
    schema_version: int = 1

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmStatus:
        payload = _exact(
            value,
            label="retrieve_swarm_status",
            fields=frozenset(
                {
                    "schema_version",
                    "run_id",
                    "project_id",
                    "state",
                    "liveness",
                    "terminal",
                    "finalization",
                    "recovery",
                    "progress",
                    "issues",
                    "failure",
                    "freshness",
                }
            ),
        )
        schema_version = payload["schema_version"]
        if schema_version != 1:
            raise ValueError(f"unsupported swarm status schema_version {schema_version!r}")
        return cls(
            SwarmId(required_text(payload, "run_id")),
            ProjectId(required_text(payload, "project_id")),
            SwarmStatusState.from_wire(payload["state"]),
            SwarmStatusLiveness.from_wire(payload["liveness"]),
            SwarmStatusTerminal.from_wire(payload["terminal"]),
            SwarmStatusFinalization.from_wire(payload["finalization"]),
            SwarmStatusRecovery.from_wire(payload["recovery"]),
            SwarmStatusProgress.from_wire(payload["progress"]),
            SwarmStatusIssues.from_wire(payload["issues"]),
            SwarmStatusFailure.from_wire(payload["failure"]),
            SwarmStatusFreshness.from_wire(payload["freshness"]),
            1,
        )

    def to_wire(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "run_id": self.swarm_id,
            "project_id": self.project_id,
            "state": {
                "public_state": self.state.public_state,
                "reason": self.state.reason,
                "blocker_code": self.state.blocker_code,
            },
            "liveness": {
                "phase": self.liveness.phase,
                "task_counts": dict(self.liveness.task_counts),
                "participant_queue_counts": dict(self.liveness.participant_queue_counts),
            },
            "terminal": {
                "run_state": self.terminal.run_state,
                "run_state_is_terminal": self.terminal.run_state_is_terminal,
                "contract_status": self.terminal.contract_status,
                "contract_reason": self.terminal.contract_reason,
                "outcome": self.terminal.outcome,
            },
            "finalization": {
                "status": self.finalization.status,
                "proof_status": self.finalization.proof_status,
                "proof_satisfied": self.finalization.proof_satisfied,
                "blocker_code": self.finalization.blocker_code,
                "reason": self.finalization.reason,
            },
            "recovery": {
                "blocked": self.recovery.blocked,
                "codes": list(self.recovery.codes),
                "incident_count": self.recovery.incident_count,
                "latest_recorded_at": (
                    None
                    if self.recovery.latest_recorded_at is None
                    else self.recovery.latest_recorded_at.isoformat()
                ),
            },
            "progress": {
                "last_progress_at": (
                    None
                    if self.progress.last_progress_at is None
                    else self.progress.last_progress_at.isoformat()
                ),
                "last_progress_kind": self.progress.last_progress_kind,
            },
            "issues": {
                "issues": [
                    {
                        "kind": issue.kind.value,
                        "code": issue.code,
                        "source": issue.source,
                        "status": issue.status,
                        "severity": issue.severity,
                        "summary": issue.summary,
                        "first_seen_at": (
                            None if issue.first_seen_at is None else issue.first_seen_at.isoformat()
                        ),
                        "last_seen_at": (
                            None if issue.last_seen_at is None else issue.last_seen_at.isoformat()
                        ),
                    }
                    for issue in self.issues.issues
                ],
                "invariants": [
                    {
                        "code": item.code,
                        "source": item.source,
                        "message": item.message,
                    }
                    for item in self.issues.invariants
                ],
            },
            "failure": {
                "code": self.failure.code,
                "source": self.failure.source,
                "classification": self.failure.classification,
                "terminal": self.failure.terminal,
            },
            "freshness": {
                "projection_authority": self.freshness.projection_authority,
                "last_authoritative_update_at": (
                    self.freshness.last_authoritative_update_at.isoformat()
                ),
                "generated_at": self.freshness.generated_at.isoformat(),
            },
        }


__all__ = [
    "SwarmStatus",
    "SwarmStatusFailure",
    "SwarmStatusFinalization",
    "SwarmStatusFreshness",
    "SwarmStatusInvariant",
    "SwarmStatusIssue",
    "SwarmStatusIssueKind",
    "SwarmStatusIssues",
    "SwarmStatusLiveness",
    "SwarmStatusProgress",
    "SwarmStatusRecovery",
    "SwarmStatusState",
    "SwarmStatusTerminal",
]
