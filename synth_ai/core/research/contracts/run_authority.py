"""Typed task and execution-turn reads from backend runtime authority."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} keys must be strings")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _optional_mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    return _mapping(value, field_name=field_name)


def _optional_mapping_or_none(value: object, *, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _mapping(value, field_name=field_name)


def _text(value: object, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _optional_text(value: object) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _optional_datetime(value: object, *, field_name: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    raise ValueError(f"{field_name} must be an ISO-8601 datetime when provided")


def _mapping_tuple(value: object, *, field_name: str) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return tuple(
        _mapping(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value)
    )


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return tuple(_text(item, field_name=field_name) for item in value)


def _mapping_group(
    value: object,
    *,
    field_name: str,
) -> dict[str, tuple[dict[str, Any], ...]]:
    mapping = _mapping(value, field_name=field_name)
    return {
        str(key): _mapping_tuple(item, field_name=f"{field_name}.{key}")
        for key, item in mapping.items()
    }


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


@dataclass(frozen=True, slots=True)
class ManagedResearchRunTask:
    """Field-for-field mirror of the backend ``SmrRunTaskResponse`` DTO."""

    task_id: str
    run_id: str
    org_id: str
    project_id: str
    task_key: str
    kind: str
    public_task_state: str
    execution_owner: str
    created_at: datetime
    updated_at: datetime
    task_state: str | None = None
    depends_on_task_keys: tuple[str, ...] = ()
    retry_of: str | None = None
    input: dict[str, Any] = field(default_factory=dict)
    agent_goal_assignment: dict[str, Any] | None = None
    output: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    task_dispatch: dict[str, Any] = field(default_factory=dict)
    worker_pool: str | None = None
    claimed_by: str | None = None
    lease_expires_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchRunTask:
        mapping = _mapping(payload, field_name="run task")
        raw_dependencies = mapping.get("depends_on_task_keys")
        if raw_dependencies is None:
            dependencies: tuple[str, ...] = ()
        elif isinstance(raw_dependencies, list):
            dependencies = tuple(
                normalized for item in raw_dependencies if (normalized := str(item or "").strip())
            )
        else:
            raise ValueError("run_task.depends_on_task_keys must be a list")
        created_at = _optional_datetime(mapping.get("created_at"), field_name="run_task.created_at")
        updated_at = _optional_datetime(mapping.get("updated_at"), field_name="run_task.updated_at")
        if created_at is None or updated_at is None:
            raise ValueError("run_task.created_at and run_task.updated_at are required")
        return cls(
            task_id=_text(mapping.get("task_id"), field_name="run_task.task_id"),
            run_id=_text(mapping.get("run_id"), field_name="run_task.run_id"),
            org_id=_text(mapping.get("org_id"), field_name="run_task.org_id"),
            project_id=_text(mapping.get("project_id"), field_name="run_task.project_id"),
            task_key=_text(mapping.get("task_key"), field_name="run_task.task_key"),
            kind=_text(mapping.get("kind"), field_name="run_task.kind"),
            public_task_state=_text(
                mapping.get("public_task_state"),
                field_name="run_task.public_task_state",
            ),
            task_state=_optional_text(mapping.get("task_state")),
            depends_on_task_keys=dependencies,
            retry_of=_optional_text(mapping.get("retry_of")),
            input=_optional_mapping(mapping.get("input"), field_name="run_task.input"),
            agent_goal_assignment=_optional_mapping_or_none(
                mapping.get("agent_goal_assignment"),
                field_name="run_task.agent_goal_assignment",
            ),
            output=_optional_mapping(mapping.get("output"), field_name="run_task.output"),
            diagnostics=_optional_mapping(
                mapping.get("diagnostics"), field_name="run_task.diagnostics"
            ),
            execution_owner=_text(
                mapping.get("execution_owner"),
                field_name="run_task.execution_owner",
            ),
            task_dispatch=_optional_mapping(
                mapping.get("task_dispatch"), field_name="run_task.task_dispatch"
            ),
            worker_pool=_optional_text(mapping.get("worker_pool")),
            claimed_by=_optional_text(mapping.get("claimed_by")),
            lease_expires_at=_optional_datetime(
                mapping.get("lease_expires_at"),
                field_name="run_task.lease_expires_at",
            ),
            last_heartbeat_at=_optional_datetime(
                mapping.get("last_heartbeat_at"),
                field_name="run_task.last_heartbeat_at",
            ),
            created_at=created_at,
            started_at=_optional_datetime(
                mapping.get("started_at"), field_name="run_task.started_at"
            ),
            finished_at=_optional_datetime(
                mapping.get("finished_at"), field_name="run_task.finished_at"
            ),
            updated_at=updated_at,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "org_id": self.org_id,
            "project_id": self.project_id,
            "task_key": self.task_key,
            "kind": self.kind,
            "public_task_state": self.public_task_state,
            "task_state": self.task_state,
            "depends_on_task_keys": list(self.depends_on_task_keys),
            "retry_of": self.retry_of,
            "input": dict(self.input),
            "agent_goal_assignment": (
                dict(self.agent_goal_assignment) if self.agent_goal_assignment is not None else None
            ),
            "output": dict(self.output),
            "diagnostics": dict(self.diagnostics),
            "execution_owner": self.execution_owner,
            "task_dispatch": dict(self.task_dispatch),
            "worker_pool": self.worker_pool,
            "claimed_by": self.claimed_by,
            "lease_expires_at": _iso(self.lease_expires_at),
            "last_heartbeat_at": _iso(self.last_heartbeat_at),
            "created_at": self.created_at.isoformat(),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ManagedResearchExecutionTurn:
    """Canonical execution-turn row projected by backend authority readouts."""

    execution_turn_id: str
    task_id: str
    task_key: str
    phase: str
    participant_session_id: str | None = None
    participant_role: str | None = None
    execution_owner: str | None = None
    terminal_status: str | None = None
    completion_state: str | None = None
    started_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    completed_at: datetime | None = None
    thread_id: str | None = None
    codex_turn_id: str | None = None
    event_projection: dict[str, Any] = field(default_factory=dict)
    turn_projection: dict[str, Any] = field(default_factory=dict)
    turn_progress: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchExecutionTurn:
        mapping = _mapping(payload, field_name="execution turn")
        return cls(
            execution_turn_id=_text(
                mapping.get("execution_turn_id"),
                field_name="execution_turn.execution_turn_id",
            ),
            task_id=_text(mapping.get("task_id"), field_name="execution_turn.task_id"),
            task_key=_text(mapping.get("task_key"), field_name="execution_turn.task_key"),
            participant_session_id=_optional_text(mapping.get("participant_session_id")),
            participant_role=_optional_text(mapping.get("participant_role")),
            execution_owner=_optional_text(mapping.get("execution_owner")),
            phase=_text(mapping.get("phase"), field_name="execution_turn.phase"),
            terminal_status=_optional_text(mapping.get("terminal_status")),
            completion_state=_optional_text(mapping.get("completion_state")),
            started_at=_optional_datetime(
                mapping.get("started_at"), field_name="execution_turn.started_at"
            ),
            last_heartbeat_at=_optional_datetime(
                mapping.get("last_heartbeat_at"),
                field_name="execution_turn.last_heartbeat_at",
            ),
            completed_at=_optional_datetime(
                mapping.get("completed_at"), field_name="execution_turn.completed_at"
            ),
            thread_id=_optional_text(mapping.get("thread_id")),
            codex_turn_id=_optional_text(mapping.get("codex_turn_id")),
            event_projection=_optional_mapping(
                mapping.get("event_projection"),
                field_name="execution_turn.event_projection",
            ),
            turn_projection=_optional_mapping(
                mapping.get("turn_projection"),
                field_name="execution_turn.turn_projection",
            ),
            turn_progress=_optional_mapping(
                mapping.get("turn_progress"),
                field_name="execution_turn.turn_progress",
            ),
            metadata=_optional_mapping(
                mapping.get("metadata"), field_name="execution_turn.metadata"
            ),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "execution_turn_id": self.execution_turn_id,
            "task_id": self.task_id,
            "task_key": self.task_key,
            "participant_session_id": self.participant_session_id,
            "participant_role": self.participant_role,
            "execution_owner": self.execution_owner,
            "phase": self.phase,
            "terminal_status": self.terminal_status,
            "completion_state": self.completion_state,
            "started_at": _iso(self.started_at),
            "last_heartbeat_at": _iso(self.last_heartbeat_at),
            "completed_at": _iso(self.completed_at),
            "thread_id": self.thread_id,
            "codex_turn_id": self.codex_turn_id,
            "event_projection": dict(self.event_projection),
            "turn_projection": dict(self.turn_projection),
            "turn_progress": dict(self.turn_progress),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ManagedResearchAuthorityTask:
    """Canonical task row projected from backend runtime authority tables."""

    task_id: str
    task_key: str
    task_kind: str
    task_state: str
    execution_owner: str
    depends_on_task_keys: tuple[str, ...] = ()
    retry_of: str | None = None
    input: dict[str, Any] = field(default_factory=dict)
    worker_pool: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime | None = None
    task_policy: dict[str, Any] | None = None
    task_attempts: tuple[dict[str, Any], ...] = ()
    runtime_work_attempts: tuple[dict[str, Any], ...] = ()
    lifecycle: dict[str, Any] | None = None
    execution_claims: tuple[dict[str, Any], ...] = ()
    current_execution_progress: dict[str, Any] | None = None
    execution_turns: tuple[ManagedResearchExecutionTurn, ...] = ()
    worker_completion_claims: tuple[dict[str, Any], ...] = ()
    task_completion_assessments: tuple[dict[str, Any], ...] = ()
    failure_classifications: tuple[dict[str, Any], ...] = ()
    recovery_incidents: tuple[dict[str, Any], ...] = ()
    usage_receipts: tuple[dict[str, Any], ...] = ()
    runtime_receipt_relevance: tuple[dict[str, Any], ...] = ()
    evidence_requirements: tuple[dict[str, Any], ...] = ()
    evidence_refs: tuple[dict[str, Any], ...] = ()
    evidence_adoptions: tuple[dict[str, Any], ...] = ()
    validation_requirements: tuple[dict[str, Any], ...] = ()
    validation_assessments: tuple[dict[str, Any], ...] = ()
    task_events: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchAuthorityTask:
        mapping = _mapping(payload, field_name="runtime authority task")
        raw_dependencies = mapping.get("depends_on_task_keys")
        if raw_dependencies is None:
            dependencies: tuple[str, ...] = ()
        elif isinstance(raw_dependencies, list):
            dependencies = tuple(
                normalized for item in raw_dependencies if (normalized := str(item or "").strip())
            )
        else:
            raise ValueError("runtime_authority.task.depends_on_task_keys must be a list")
        turns = tuple(
            ManagedResearchExecutionTurn.from_wire(item)
            for item in list(mapping.get("execution_turns") or [])
        )
        task_id = _text(mapping.get("task_id"), field_name="runtime_authority.task.task_id")
        task_key = _text(mapping.get("task_key"), field_name="runtime_authority.task.task_key")
        for turn in turns:
            if turn.task_id != task_id or turn.task_key != task_key:
                raise ValueError(
                    "runtime authority execution-turn identity does not match its task"
                )
        return cls(
            task_id=task_id,
            task_key=task_key,
            task_kind=_text(
                mapping.get("task_kind"), field_name="runtime_authority.task.task_kind"
            ),
            task_state=_text(
                mapping.get("task_state"), field_name="runtime_authority.task.task_state"
            ),
            execution_owner=_text(
                mapping.get("execution_owner"),
                field_name="runtime_authority.task.execution_owner",
            ),
            depends_on_task_keys=dependencies,
            retry_of=_optional_text(mapping.get("retry_of")),
            input=_optional_mapping(
                mapping.get("input"), field_name="runtime_authority.task.input"
            ),
            worker_pool=_optional_text(mapping.get("worker_pool")),
            created_at=_optional_datetime(
                mapping.get("created_at"), field_name="runtime_authority.task.created_at"
            ),
            started_at=_optional_datetime(
                mapping.get("started_at"), field_name="runtime_authority.task.started_at"
            ),
            finished_at=_optional_datetime(
                mapping.get("finished_at"), field_name="runtime_authority.task.finished_at"
            ),
            updated_at=_optional_datetime(
                mapping.get("updated_at"), field_name="runtime_authority.task.updated_at"
            ),
            task_policy=_optional_mapping_or_none(
                mapping.get("task_policy"),
                field_name="runtime_authority.task.task_policy",
            ),
            task_attempts=_mapping_tuple(
                mapping.get("task_attempts"),
                field_name="runtime_authority.task.task_attempts",
            ),
            runtime_work_attempts=_mapping_tuple(
                mapping.get("runtime_work_attempts"),
                field_name="runtime_authority.task.runtime_work_attempts",
            ),
            lifecycle=_optional_mapping_or_none(
                mapping.get("lifecycle"), field_name="runtime_authority.task.lifecycle"
            ),
            execution_claims=_mapping_tuple(
                mapping.get("execution_claims"),
                field_name="runtime_authority.task.execution_claims",
            ),
            current_execution_progress=_optional_mapping_or_none(
                mapping.get("current_execution_progress"),
                field_name="runtime_authority.task.current_execution_progress",
            ),
            execution_turns=turns,
            worker_completion_claims=_mapping_tuple(
                mapping.get("worker_completion_claims"),
                field_name="runtime_authority.task.worker_completion_claims",
            ),
            task_completion_assessments=_mapping_tuple(
                mapping.get("task_completion_assessments"),
                field_name="runtime_authority.task.task_completion_assessments",
            ),
            failure_classifications=_mapping_tuple(
                mapping.get("failure_classifications"),
                field_name="runtime_authority.task.failure_classifications",
            ),
            recovery_incidents=_mapping_tuple(
                mapping.get("recovery_incidents"),
                field_name="runtime_authority.task.recovery_incidents",
            ),
            usage_receipts=_mapping_tuple(
                mapping.get("usage_receipts"),
                field_name="runtime_authority.task.usage_receipts",
            ),
            runtime_receipt_relevance=_mapping_tuple(
                mapping.get("runtime_receipt_relevance"),
                field_name="runtime_authority.task.runtime_receipt_relevance",
            ),
            evidence_requirements=_mapping_tuple(
                mapping.get("evidence_requirements"),
                field_name="runtime_authority.task.evidence_requirements",
            ),
            evidence_refs=_mapping_tuple(
                mapping.get("evidence_refs"),
                field_name="runtime_authority.task.evidence_refs",
            ),
            evidence_adoptions=_mapping_tuple(
                mapping.get("evidence_adoptions"),
                field_name="runtime_authority.task.evidence_adoptions",
            ),
            validation_requirements=_mapping_tuple(
                mapping.get("validation_requirements"),
                field_name="runtime_authority.task.validation_requirements",
            ),
            validation_assessments=_mapping_tuple(
                mapping.get("validation_assessments"),
                field_name="runtime_authority.task.validation_assessments",
            ),
            task_events=_mapping_tuple(
                mapping.get("task_events"),
                field_name="runtime_authority.task.task_events",
            ),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_key": self.task_key,
            "task_kind": self.task_kind,
            "task_state": self.task_state,
            "depends_on_task_keys": list(self.depends_on_task_keys),
            "retry_of": self.retry_of,
            "input": dict(self.input),
            "execution_owner": self.execution_owner,
            "worker_pool": self.worker_pool,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "updated_at": _iso(self.updated_at),
            "task_policy": dict(self.task_policy) if self.task_policy is not None else None,
            "task_attempts": [dict(item) for item in self.task_attempts],
            "runtime_work_attempts": [dict(item) for item in self.runtime_work_attempts],
            "lifecycle": dict(self.lifecycle) if self.lifecycle is not None else None,
            "execution_claims": [dict(item) for item in self.execution_claims],
            "current_execution_progress": (
                dict(self.current_execution_progress)
                if self.current_execution_progress is not None
                else None
            ),
            "execution_turns": [turn.to_wire() for turn in self.execution_turns],
            "worker_completion_claims": [dict(item) for item in self.worker_completion_claims],
            "task_completion_assessments": [
                dict(item) for item in self.task_completion_assessments
            ],
            "failure_classifications": [dict(item) for item in self.failure_classifications],
            "recovery_incidents": [dict(item) for item in self.recovery_incidents],
            "usage_receipts": [dict(item) for item in self.usage_receipts],
            "runtime_receipt_relevance": [dict(item) for item in self.runtime_receipt_relevance],
            "validation_requirements": [dict(item) for item in self.validation_requirements],
            "validation_assessments": [dict(item) for item in self.validation_assessments],
            "evidence_requirements": [dict(item) for item in self.evidence_requirements],
            "evidence_refs": [dict(item) for item in self.evidence_refs],
            "evidence_adoptions": [dict(item) for item in self.evidence_adoptions],
            "task_events": [dict(item) for item in self.task_events],
        }


@dataclass(frozen=True, slots=True)
class ManagedResearchRuntimeAuthority:
    """Typed mirror of the backend runtime-authority owner read model."""

    read_model_name: str
    run_id: str
    org_id: str
    project_id: str
    source_authority_version: str
    source_row_ids: dict[str, tuple[str, ...]]
    loaded_at: datetime
    run: dict[str, Any]
    tasks: tuple[ManagedResearchAuthorityTask, ...]
    runtime_work: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)
    control: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)
    validation: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)
    finalization: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)
    resources: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)
    human_control: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)
    evidence: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchRuntimeAuthority:
        mapping = _mapping(payload, field_name="runtime authority")
        read_model_name = _text(
            mapping.get("read_model_name"),
            field_name="runtime_authority.read_model_name",
        )
        if read_model_name != "runtime_authority":
            raise ValueError("runtime_authority.read_model_name must be runtime_authority")
        run_id = _text(mapping.get("run_id"), field_name="runtime_authority.run_id")
        org_id = _text(mapping.get("org_id"), field_name="runtime_authority.org_id")
        project_id = _text(mapping.get("project_id"), field_name="runtime_authority.project_id")
        source_authority_version = _text(
            mapping.get("source_authority_version"),
            field_name="runtime_authority.source_authority_version",
        )
        loaded_at = _optional_datetime(
            mapping.get("loaded_at"), field_name="runtime_authority.loaded_at"
        )
        if loaded_at is None:
            raise ValueError("runtime_authority.loaded_at is required")
        run = _optional_mapping(mapping.get("run"), field_name="runtime_authority.run")
        for key, expected in (
            ("run_id", run_id),
            ("org_id", org_id),
            ("project_id", project_id),
        ):
            if _optional_text(run.get(key)) != expected:
                raise ValueError(f"runtime_authority.run.{key} does not match the owner envelope")
        raw_tasks = mapping.get("tasks")
        if not isinstance(raw_tasks, list):
            raise ValueError("runtime_authority.tasks must be a list")
        raw_source_row_ids = _mapping(
            mapping.get("source_row_ids"),
            field_name="runtime_authority.source_row_ids",
        )
        return cls(
            read_model_name=read_model_name,
            run_id=run_id,
            org_id=org_id,
            project_id=project_id,
            source_authority_version=source_authority_version,
            source_row_ids={
                str(key): _string_tuple(
                    value,
                    field_name=f"runtime_authority.source_row_ids.{key}",
                )
                for key, value in raw_source_row_ids.items()
            },
            loaded_at=loaded_at,
            run=run,
            tasks=tuple(ManagedResearchAuthorityTask.from_wire(item) for item in raw_tasks),
            runtime_work=_mapping_group(
                mapping.get("runtime_work"),
                field_name="runtime_authority.runtime_work",
            ),
            control=_mapping_group(mapping.get("control"), field_name="runtime_authority.control"),
            validation=_mapping_group(
                mapping.get("validation"),
                field_name="runtime_authority.validation",
            ),
            finalization=_mapping_group(
                mapping.get("finalization"),
                field_name="runtime_authority.finalization",
            ),
            resources=_mapping_group(
                mapping.get("resources"),
                field_name="runtime_authority.resources",
            ),
            human_control=_mapping_group(
                mapping.get("human_control"),
                field_name="runtime_authority.human_control",
            ),
            evidence=_mapping_group(
                mapping.get("evidence"), field_name="runtime_authority.evidence"
            ),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "read_model_name": self.read_model_name,
            "run_id": self.run_id,
            "org_id": self.org_id,
            "project_id": self.project_id,
            "source_authority_version": self.source_authority_version,
            "source_row_ids": {key: list(value) for key, value in self.source_row_ids.items()},
            "loaded_at": self.loaded_at.isoformat(),
            "run": dict(self.run),
            "tasks": [task.to_wire() for task in self.tasks],
            "runtime_work": {
                key: [dict(item) for item in value] for key, value in self.runtime_work.items()
            },
            "control": {key: [dict(item) for item in value] for key, value in self.control.items()},
            "validation": {
                key: [dict(item) for item in value] for key, value in self.validation.items()
            },
            "finalization": {
                key: [dict(item) for item in value] for key, value in self.finalization.items()
            },
            "resources": {
                key: [dict(item) for item in value] for key, value in self.resources.items()
            },
            "human_control": {
                key: [dict(item) for item in value] for key, value in self.human_control.items()
            },
            "evidence": {
                key: [dict(item) for item in value] for key, value in self.evidence.items()
            },
        }


__all__ = [
    "ManagedResearchAuthorityTask",
    "ManagedResearchExecutionTurn",
    "ManagedResearchRunTask",
    "ManagedResearchRuntimeAuthority",
]
