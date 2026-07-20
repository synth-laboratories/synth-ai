"""Typed run observability models for rich Managed Research polling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from synth_ai.managed_research.models.run_state import (
    ManagedResearchRun,
    ManagedResearchRunLivenessPhase,
    ManagedResearchRunState,
    ManagedResearchRunTerminalOutcome,
)


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


def _parse_liveness_phase(value: str | None) -> ManagedResearchRunLivenessPhase:
    if not value:
        return ManagedResearchRunLivenessPhase.UNKNOWN
    try:
        return ManagedResearchRunLivenessPhase(value)
    except ValueError:
        return ManagedResearchRunLivenessPhase.UNKNOWN


def _require_string(payload: Mapping[str, object], key: str, *, label: str) -> str:
    value = _optional_string(payload, key)
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _require_public_task_state(payload: Mapping[str, object], *, label: str) -> str:
    value = _optional_string(payload, "public_task_state") or _optional_string(
        payload, "task_state"
    )
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _optional_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer when provided")
    return value


def _optional_bool(payload: Mapping[str, object], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean when provided")
    return value


def _require_bool(payload: Mapping[str, object], key: str, *, label: str) -> bool:
    value = _optional_bool(payload, key)
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _require_int(payload: Mapping[str, object], key: str, *, label: str) -> int:
    value = _optional_int(payload, key)
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _optional_object_dict(payload: object) -> dict[str, object]:
    if payload is None:
        return {}
    mapping = _require_mapping(payload, label="metadata")
    return dict(mapping)


def _optional_dict_list(payload: object, *, label: str) -> list[dict[str, object]]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be an array when provided")
    rows: list[dict[str, object]] = []
    for item in payload:
        rows.append(dict(_require_mapping(item, label=label)))
    return rows


def _optional_string_int_map(payload: object, *, label: str) -> dict[str, int]:
    if payload is None:
        return {}
    mapping = _require_mapping(payload, label=label)
    normalized: dict[str, int] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} keys must be strings")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label}.{key} must be an integer")
        normalized[key] = value
    return normalized


def _optional_string_list(payload: object, *, label: str) -> list[str]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be an array when provided")
    normalized: list[str] = []
    for index, value in enumerate(payload):
        if not isinstance(value, str):
            raise ValueError(f"{label}[{index}] must be a string")
        normalized.append(value)
    return normalized


def _optional_mapping_field(payload: object, *, label: str) -> dict[str, object] | None:
    if payload is None:
        return None
    return dict(_require_mapping(payload, label=label))


class RunTickMode(StrEnum):
    AUTO = "auto"
    MANUAL = "manual"


class CandidatePublicationOutcome(StrEnum):
    RUNNING = "running"
    PR_PUBLISHED = "pr_published"
    AWAITING_PR_BINDING = "awaiting_pr_binding"
    FAILED_BEFORE_BRANCH = "failed_before_branch"
    FAILED_AFTER_BRANCH_NO_PR = "failed_after_branch_no_pr"


class RunAnomalyKind(StrEnum):
    RUNNING_WITHOUT_LIVE_ACTORS = "running_without_live_actors"
    RUNNING_WITHOUT_NONTERMINAL_TASKS = "running_without_nonterminal_tasks"
    RUN_TERMINAL_WITH_NONTERMINAL_TASKS = "run_terminal_with_nonterminal_tasks"
    RUN_TERMINAL_WITH_PENDING_RUNTIME_INTENTS = "run_terminal_with_pending_runtime_intents"
    MCP_UNREACHABLE = "mcp_unreachable"
    TERMINAL_WITHOUT_PUBLICATION_VERDICT = "terminal_without_publication_verdict"
    ACTOR_BINDING_PROJECTION_DIVERGENCE = "actor_binding_projection_divergence"


@dataclass(frozen=True)
class RunObservationCursor:
    latest_event_seq: int | None = None
    latest_runtime_message_seq: int | None = None
    latest_runtime_event_id: str | None = None
    generated_at: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> RunObservationCursor:
        mapping = _require_mapping(payload, label="run observation cursor")
        return cls(
            latest_event_seq=_optional_int(mapping, "latest_event_seq"),
            latest_runtime_message_seq=_optional_int(mapping, "latest_runtime_message_seq"),
            latest_runtime_event_id=_optional_string(mapping, "latest_runtime_event_id"),
            generated_at=_optional_string(mapping, "generated_at"),
        )

    def to_query_params(self) -> dict[str, object]:
        params: dict[str, object] = {}
        if self.latest_event_seq is not None:
            params["since_event_seq"] = self.latest_event_seq
        if self.latest_runtime_message_seq is not None:
            params["latest_runtime_message_seq"] = self.latest_runtime_message_seq
        if self.latest_runtime_event_id is not None:
            params["latest_runtime_event_id"] = self.latest_runtime_event_id
        return params


@dataclass(frozen=True)
class RunLifecycleLocalExecution:
    slot_id: str
    runtime_id: str
    dispatch_pool: str
    host_kind: str
    requires_hosted_capacity: bool = False

    @classmethod
    def from_wire(cls, payload: object) -> RunLifecycleLocalExecution:
        mapping = _require_mapping(payload, label="run lifecycle local execution")
        return cls(
            slot_id=_require_string(mapping, "slot_id", label="local_execution.slot_id"),
            runtime_id=_require_string(mapping, "runtime_id", label="local_execution.runtime_id"),
            dispatch_pool=_require_string(
                mapping, "dispatch_pool", label="local_execution.dispatch_pool"
            ),
            host_kind=_require_string(mapping, "host_kind", label="local_execution.host_kind"),
            requires_hosted_capacity=bool(mapping.get("requires_hosted_capacity") or False),
        )


@dataclass(frozen=True)
class RunLifecycleDispatch:
    owner: str
    pool_id: str
    host_kind: str
    local_execution: RunLifecycleLocalExecution | None = None

    @classmethod
    def from_wire(cls, payload: object) -> RunLifecycleDispatch:
        mapping = _require_mapping(payload, label="run lifecycle dispatch")
        local_execution = mapping.get("local_execution")
        return cls(
            owner=_require_string(mapping, "owner", label="dispatch.owner"),
            pool_id=_require_string(mapping, "pool_id", label="dispatch.pool_id"),
            host_kind=_require_string(mapping, "host_kind", label="dispatch.host_kind"),
            local_execution=(
                RunLifecycleLocalExecution.from_wire(local_execution)
                if local_execution is not None
                else None
            ),
        )


@dataclass(frozen=True)
class RunLifecycleFailure:
    family: str
    detail: str
    reason: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> RunLifecycleFailure:
        mapping = _require_mapping(payload, label="run lifecycle failure")
        return cls(
            family=_require_string(mapping, "family", label="failure.family"),
            detail=_require_string(mapping, "detail", label="failure.detail"),
            reason=_optional_string(mapping, "reason"),
        )


@dataclass(frozen=True)
class RunLifecycleView:
    authority_phase: str
    bootstrap_phase: str | None
    terminal_phase: str
    terminal_outcome: str | None
    dispatch: RunLifecycleDispatch
    failure: RunLifecycleFailure | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    updated_at: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> RunLifecycleView:
        mapping = _require_mapping(payload, label="run lifecycle")
        failure = mapping.get("failure")
        return cls(
            authority_phase=_require_string(
                mapping, "authority_phase", label="lifecycle.authority_phase"
            ),
            bootstrap_phase=_optional_string(mapping, "bootstrap_phase"),
            terminal_phase=_require_string(
                mapping, "terminal_phase", label="lifecycle.terminal_phase"
            ),
            terminal_outcome=_optional_string(mapping, "terminal_outcome"),
            dispatch=RunLifecycleDispatch.from_wire(mapping.get("dispatch")),
            failure=RunLifecycleFailure.from_wire(failure) if failure is not None else None,
            metadata=_optional_object_dict(mapping.get("metadata")),
            updated_at=_optional_string(mapping, "updated_at"),
        )


@dataclass(frozen=True)
class CandidatePublicationView:
    outcome: CandidatePublicationOutcome
    branch_name: str | None = None
    head_commit_sha: str | None = None
    pr_url: str | None = None
    pr_number: int | None = None
    repo: str | None = None
    base_branch: str | None = None
    failure_code: str | None = None
    failure_detail: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> CandidatePublicationView:
        mapping = _require_mapping(payload, label="candidate publication")
        outcome_raw = _require_string(mapping, "outcome", label="candidate_publication.outcome")
        return cls(
            outcome=CandidatePublicationOutcome(outcome_raw),
            branch_name=_optional_string(mapping, "branch_name"),
            head_commit_sha=_optional_string(mapping, "head_commit_sha"),
            pr_url=_optional_string(mapping, "pr_url"),
            pr_number=_optional_int(mapping, "pr_number"),
            repo=_optional_string(mapping, "repo"),
            base_branch=_optional_string(mapping, "base_branch"),
            failure_code=_optional_string(mapping, "failure_code"),
            failure_detail=_optional_string(mapping, "failure_detail"),
        )


@dataclass(frozen=True)
class ActorSnapshot:
    actor_id: str
    actor_type: str
    state: str
    profile_id: str | None = None
    agent_harness: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    host_kind: str | None = None
    participant_role: str | None = None
    phase: str | None = None
    live_session: bool = False
    session_status: str | None = None
    task_id: str | None = None
    task_key: str | None = None
    runtime_source: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    last_heartbeat_at: str | None = None
    labels: dict[str, object] = field(default_factory=dict)
    runtime_bootstrap: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> ActorSnapshot:
        mapping = _require_mapping(payload, label="actor snapshot")
        return cls(
            actor_id=_require_string(mapping, "actor_id", label="actor.actor_id"),
            actor_type=_require_string(mapping, "actor_type", label="actor.actor_type"),
            state=_require_string(mapping, "state", label="actor.state"),
            profile_id=_optional_string(mapping, "profile_id"),
            agent_harness=_optional_string(mapping, "agent_harness"),
            model=_optional_string(mapping, "model"),
            reasoning_effort=_optional_string(mapping, "reasoning_effort"),
            host_kind=_optional_string(mapping, "host_kind"),
            participant_role=_optional_string(mapping, "participant_role"),
            phase=_optional_string(mapping, "phase"),
            live_session=bool(_optional_bool(mapping, "live_session")),
            session_status=_optional_string(mapping, "session_status"),
            task_id=_optional_string(mapping, "task_id"),
            task_key=_optional_string(mapping, "task_key"),
            runtime_source=_optional_string(mapping, "runtime_source"),
            started_at=_optional_string(mapping, "started_at"),
            updated_at=_optional_string(mapping, "updated_at"),
            completed_at=_optional_string(mapping, "completed_at"),
            last_heartbeat_at=_optional_string(mapping, "last_heartbeat_at"),
            labels=_optional_object_dict(mapping.get("labels")),
            runtime_bootstrap=_optional_object_dict(mapping.get("runtime_bootstrap")),
        )


@dataclass(frozen=True)
class ActorCollectionSnapshot:
    total_count: int
    counts_by_state: dict[str, int] = field(default_factory=dict)
    counts_by_role: dict[str, int] = field(default_factory=dict)
    items: list[ActorSnapshot] = field(default_factory=list)
    latest_transitions: list[ActorSnapshot] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> ActorCollectionSnapshot:
        mapping = _require_mapping(payload, label="actor collection")
        return cls(
            total_count=_optional_int(mapping, "total_count") or 0,
            counts_by_state=_optional_string_int_map(
                mapping.get("counts_by_state"), label="actors.counts_by_state"
            ),
            counts_by_role=_optional_string_int_map(
                mapping.get("counts_by_role"), label="actors.counts_by_role"
            ),
            items=[
                ActorSnapshot.from_wire(item)
                for item in _optional_dict_list(mapping.get("items"), label="actors.items")
            ],
            latest_transitions=[
                ActorSnapshot.from_wire(item)
                for item in _optional_dict_list(
                    mapping.get("latest_transitions"), label="actors.latest_transitions"
                )
            ],
        )


@dataclass(frozen=True)
class TaskSnapshot:
    task_id: str
    task_key: str
    kind: str
    task_state: str
    public_task_state: str
    execution_owner: str
    claimed_by: str | None = None
    worker_pool: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    updated_at: str | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> TaskSnapshot:
        mapping = _require_mapping(payload, label="task snapshot")
        public_task_state = _require_public_task_state(mapping, label="task.public_task_state")
        return cls(
            task_id=_require_string(mapping, "task_id", label="task.task_id"),
            task_key=_require_string(mapping, "task_key", label="task.task_key"),
            kind=_require_string(mapping, "kind", label="task.kind"),
            task_state=public_task_state,
            public_task_state=public_task_state,
            execution_owner=_require_string(
                mapping, "execution_owner", label="task.execution_owner"
            ),
            claimed_by=_optional_string(mapping, "claimed_by"),
            worker_pool=_optional_string(mapping, "worker_pool"),
            started_at=_optional_string(mapping, "started_at"),
            finished_at=_optional_string(mapping, "finished_at"),
            updated_at=_optional_string(mapping, "updated_at"),
            diagnostics=_optional_object_dict(mapping.get("diagnostics")),
        )


@dataclass(frozen=True)
class TaskCollectionSnapshot:
    total_count: int
    counts_by_state: dict[str, int] = field(default_factory=dict)
    counts_by_owner: dict[str, int] = field(default_factory=dict)
    items: list[TaskSnapshot] = field(default_factory=list)
    latest_transitions: list[TaskSnapshot] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> TaskCollectionSnapshot:
        mapping = _require_mapping(payload, label="task collection")
        return cls(
            total_count=_optional_int(mapping, "total_count") or 0,
            counts_by_state=_optional_string_int_map(
                mapping.get("counts_by_state"), label="tasks.counts_by_state"
            ),
            counts_by_owner=_optional_string_int_map(
                mapping.get("counts_by_owner"), label="tasks.counts_by_owner"
            ),
            items=[
                TaskSnapshot.from_wire(item)
                for item in _optional_dict_list(mapping.get("items"), label="tasks.items")
            ],
            latest_transitions=[
                TaskSnapshot.from_wire(item)
                for item in _optional_dict_list(
                    mapping.get("latest_transitions"), label="tasks.latest_transitions"
                )
            ],
        )


@dataclass(frozen=True)
class TaskSummary:
    task_id: str
    task_key: str
    kind: str
    public_task_state: str
    task_state: str | None = None
    target_kind: str | None = None
    task_affinity_key: str | None = None
    input_keys: list[str] = field(default_factory=list)
    instructions_present: bool = False
    acceptance_criteria_count: int = 0
    agent: str | None = None
    model: str | None = None
    worker_profile_id: str | None = None
    orchestrator_profile_id: str | None = None
    title: str | None = None
    url: str | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> TaskSummary:
        mapping = _require_mapping(payload, label="task summary")
        state = mapping.get("state")
        state_text = ""
        if isinstance(state, Mapping):
            state_text = (
                _optional_string(state, "name")
                or _optional_string(state, "type")
                or _optional_string(state, "id")
                or ""
            )
        return cls(
            task_id=(
                _optional_string(mapping, "task_id")
                or _require_string(mapping, "id", label="task_summary.id")
            ),
            task_key=(
                _optional_string(mapping, "task_key")
                or _optional_string(mapping, "identifier")
                or _require_string(mapping, "id", label="task_summary.id")
            ),
            kind=_optional_string(mapping, "kind")
            or _optional_string(mapping, "target_kind")
            or "task",
            public_task_state=(
                _optional_string(mapping, "public_task_state")
                or _optional_string(mapping, "task_state")
                or _optional_string(mapping, "status")
                or state_text
                or "unknown"
            ),
            task_state=_optional_string(mapping, "task_state"),
            target_kind=_optional_string(mapping, "target_kind"),
            task_affinity_key=_optional_string(mapping, "task_affinity_key"),
            input_keys=_optional_string_list(mapping.get("input_keys"), label="input_keys"),
            instructions_present=bool(_optional_bool(mapping, "instructions_present") or False),
            acceptance_criteria_count=(_optional_int(mapping, "acceptance_criteria_count") or 0),
            agent=_optional_string(mapping, "agent"),
            model=_optional_string(mapping, "model"),
            worker_profile_id=_optional_string(mapping, "worker_profile_id"),
            orchestrator_profile_id=_optional_string(mapping, "orchestrator_profile_id"),
            title=_optional_string(mapping, "title"),
            url=_optional_string(mapping, "url"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class RuntimeMessageView:
    message_id: str
    created_at: str
    seq: int
    mode: str
    sender: str
    target: str | None = None
    action: str | None = None
    body: str | None = None
    status: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> RuntimeMessageView:
        mapping = _require_mapping(payload, label="runtime message")
        return cls(
            message_id=_require_string(mapping, "message_id", label="runtime.message_id"),
            created_at=_require_string(mapping, "created_at", label="runtime.created_at"),
            seq=_optional_int(mapping, "seq") or 0,
            mode=_require_string(mapping, "mode", label="runtime.mode"),
            sender=_require_string(mapping, "sender", label="runtime.sender"),
            target=_optional_string(mapping, "target"),
            action=_optional_string(mapping, "action"),
            body=_optional_string(mapping, "body"),
            status=_optional_string(mapping, "status"),
        )


@dataclass(frozen=True)
class MessageQueueMessage:
    message_id: str | None = None
    thread_id: str | None = None
    parent_message_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    status: str | None = None
    intent: str | None = None
    message_kind: str | None = None
    body: str | None = None
    payload: dict[str, object] | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> MessageQueueMessage:
        mapping = _require_mapping(payload, label="message queue message")
        return cls(
            message_id=_optional_string(mapping, "message_id"),
            thread_id=_optional_string(mapping, "thread_id"),
            parent_message_id=_optional_string(mapping, "parent_message_id"),
            created_at=_optional_string(mapping, "created_at"),
            updated_at=_optional_string(mapping, "updated_at"),
            status=_optional_string(mapping, "status"),
            intent=_optional_string(mapping, "intent"),
            message_kind=_optional_string(mapping, "message_kind"),
            body=_optional_string(mapping, "body"),
            payload=_optional_mapping_field(mapping.get("payload"), label="message.payload"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class MessageQueueThread:
    thread_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    status: str | None = None
    message_count: int | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> MessageQueueThread:
        mapping = _require_mapping(payload, label="message queue thread")
        return cls(
            thread_id=_optional_string(mapping, "thread_id"),
            created_at=_optional_string(mapping, "created_at"),
            updated_at=_optional_string(mapping, "updated_at"),
            status=_optional_string(mapping, "status"),
            message_count=_optional_int(mapping, "message_count"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class MessageQueueInteraction:
    interaction_id: str | None = None
    message_id: str | None = None
    thread_id: str | None = None
    status: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> MessageQueueInteraction:
        mapping = _require_mapping(payload, label="message queue interaction")
        return cls(
            interaction_id=_optional_string(mapping, "interaction_id"),
            message_id=_optional_string(mapping, "message_id"),
            thread_id=_optional_string(mapping, "thread_id"),
            status=_optional_string(mapping, "status"),
            created_at=_optional_string(mapping, "created_at"),
            updated_at=_optional_string(mapping, "updated_at"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class RuntimeDeliveryView:
    message_id: str
    created_at: str
    mode: str
    sender: str
    target: str | None = None
    status: str | None = None
    error: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> RuntimeDeliveryView:
        mapping = _require_mapping(payload, label="runtime delivery")
        return cls(
            message_id=_require_string(mapping, "message_id", label="delivery.message_id"),
            created_at=_require_string(mapping, "created_at", label="delivery.created_at"),
            mode=_require_string(mapping, "mode", label="delivery.mode"),
            sender=_require_string(mapping, "sender", label="delivery.sender"),
            target=_optional_string(mapping, "target"),
            status=_optional_string(mapping, "status"),
            error=_optional_string(mapping, "error"),
        )


@dataclass(frozen=True)
class RuntimeEventView:
    event_id: str
    created_at: str
    kind: str
    source: str
    summary: str
    task_id: str | None = None
    task_key: str | None = None
    worker_id: str | None = None
    state: str | None = None
    detail: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> RuntimeEventView:
        mapping = _require_mapping(payload, label="runtime event")
        return cls(
            event_id=_require_string(mapping, "event_id", label="runtime.event_id"),
            created_at=_require_string(mapping, "created_at", label="runtime.created_at"),
            kind=_require_string(mapping, "kind", label="runtime.kind"),
            source=_require_string(mapping, "source", label="runtime.source"),
            summary=_require_string(mapping, "summary", label="runtime.summary"),
            task_id=_optional_string(mapping, "task_id"),
            task_key=_optional_string(mapping, "task_key"),
            worker_id=_optional_string(mapping, "worker_id"),
            state=_optional_string(mapping, "state"),
            detail=_optional_object_dict(mapping.get("detail")),
        )


@dataclass(frozen=True)
class RuntimeObservability:
    last_progress_at: str | None = None
    latest_kind: str | None = None
    latest_summary: str | None = None
    failure_summary: str | None = None
    messages: list[RuntimeMessageView] = field(default_factory=list)
    deliveries: list[RuntimeDeliveryView] = field(default_factory=list)
    events: list[RuntimeEventView] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> RuntimeObservability:
        mapping = _require_mapping(payload, label="runtime observability")
        return cls(
            last_progress_at=_optional_string(mapping, "last_progress_at"),
            latest_kind=_optional_string(mapping, "latest_kind"),
            latest_summary=_optional_string(mapping, "latest_summary"),
            failure_summary=_optional_string(mapping, "failure_summary"),
            messages=[
                RuntimeMessageView.from_wire(item)
                for item in _optional_dict_list(mapping.get("messages"), label="runtime.messages")
            ],
            deliveries=[
                RuntimeDeliveryView.from_wire(item)
                for item in _optional_dict_list(
                    mapping.get("deliveries"), label="runtime.deliveries"
                )
            ],
            events=[
                RuntimeEventView.from_wire(item)
                for item in _optional_dict_list(mapping.get("events"), label="runtime.events")
            ],
        )


@dataclass(frozen=True)
class RunManualTickRequest:
    request_id: str
    run_id: str
    project_id: str
    status: str
    reason: str | None = None
    lease_owner: str | None = None
    lease_started_at: str | None = None
    lease_expires_at: str | None = None
    last_progress_at: str | None = None
    runtime_tick_count: int = 0
    actor_step_count: int = 0
    completion_reason: str | None = None
    failure_reason: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    consumed_at: str | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> RunManualTickRequest:
        mapping = _require_mapping(payload, label="manual tick request")
        return cls(
            request_id=_require_string(mapping, "request_id", label="manual_tick.request_id"),
            run_id=_require_string(mapping, "run_id", label="manual_tick.run_id"),
            project_id=_require_string(mapping, "project_id", label="manual_tick.project_id"),
            status=_require_string(mapping, "status", label="manual_tick.status"),
            reason=_optional_string(mapping, "reason"),
            lease_owner=_optional_string(mapping, "lease_owner"),
            lease_started_at=_optional_string(mapping, "lease_started_at"),
            lease_expires_at=_optional_string(mapping, "lease_expires_at"),
            last_progress_at=_optional_string(mapping, "last_progress_at"),
            runtime_tick_count=_optional_int(mapping, "runtime_tick_count") or 0,
            actor_step_count=_optional_int(mapping, "actor_step_count") or 0,
            completion_reason=_optional_string(mapping, "completion_reason"),
            failure_reason=_optional_string(mapping, "failure_reason"),
            created_at=_optional_string(mapping, "created_at"),
            updated_at=_optional_string(mapping, "updated_at"),
            consumed_at=_optional_string(mapping, "consumed_at"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class RunTickingStatus:
    project_id: str
    run_id: str
    tick_mode: RunTickMode
    manual_tick_pending_count: int = 0
    active_manual_tick: RunManualTickRequest | None = None
    last_manual_tick: RunManualTickRequest | None = None
    last_tick_at: str | None = None
    last_tick_reason: str | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> RunTickingStatus:
        mapping = _require_mapping(payload, label="run ticking status")
        active_manual_tick = mapping.get("active_manual_tick")
        last_manual_tick = mapping.get("last_manual_tick")
        return cls(
            project_id=_require_string(mapping, "project_id", label="ticking.project_id"),
            run_id=_require_string(mapping, "run_id", label="ticking.run_id"),
            tick_mode=RunTickMode(_require_string(mapping, "tick_mode", label="ticking.tick_mode")),
            manual_tick_pending_count=(_optional_int(mapping, "manual_tick_pending_count") or 0),
            active_manual_tick=(
                RunManualTickRequest.from_wire(active_manual_tick)
                if active_manual_tick is not None
                else None
            ),
            last_manual_tick=(
                RunManualTickRequest.from_wire(last_manual_tick)
                if last_manual_tick is not None
                else None
            ),
            last_tick_at=_optional_string(mapping, "last_tick_at"),
            last_tick_reason=_optional_string(mapping, "last_tick_reason"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class RunTickingUpdate:
    tick_mode: RunTickMode | str
    reason: str | None = None

    def to_wire(self) -> dict[str, object]:
        mode = self.tick_mode.value if isinstance(self.tick_mode, RunTickMode) else self.tick_mode
        normalized_mode = str(mode or "").strip()
        if not normalized_mode:
            raise ValueError("tick_mode is required")
        payload: dict[str, object] = {"tick_mode": normalized_mode}
        if self.reason and self.reason.strip():
            payload["reason"] = self.reason.strip()
        return payload


@dataclass(frozen=True)
class RunAnomaly:
    kind: RunAnomalyKind
    detail: str

    @classmethod
    def from_wire(cls, payload: object) -> RunAnomaly:
        mapping = _require_mapping(payload, label="run anomaly")
        return cls(
            kind=RunAnomalyKind(_require_string(mapping, "kind", label="anomaly.kind")),
            detail=_require_string(mapping, "detail", label="anomaly.detail"),
        )


@dataclass(frozen=True)
class ManagedResearchRunContractLifecycle:
    phase: str

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchRunContractLifecycle:
        mapping = _require_mapping(payload, label="run_contract.lifecycle")
        return cls(
            phase=_require_string(
                mapping,
                "phase",
                label="run_contract.lifecycle.phase",
            ),
        )


@dataclass(frozen=True)
class ManagedResearchRunContractFinalization:
    status: str
    reason: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchRunContractFinalization:
        mapping = _require_mapping(payload, label="run_contract.finalization")
        return cls(
            status=_require_string(
                mapping,
                "status",
                label="run_contract.finalization.status",
            ),
            reason=_optional_string(mapping, "reason"),
        )


@dataclass(frozen=True)
class ManagedResearchRunContractRecovery:
    status: str
    next_retry_at: str | None = None
    reason: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchRunContractRecovery:
        mapping = _require_mapping(payload, label="run_contract.recovery")
        return cls(
            status=_require_string(
                mapping,
                "status",
                label="run_contract.recovery.status",
            ),
            next_retry_at=_optional_string(mapping, "next_retry_at"),
            reason=_optional_string(mapping, "reason"),
        )


@dataclass(frozen=True)
class ManagedResearchRunContractTasks:
    total: int
    terminal: int
    nonterminal: int

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchRunContractTasks:
        mapping = _require_mapping(payload, label="run_contract.tasks")
        return cls(
            total=_require_int(mapping, "total", label="run_contract.tasks.total"),
            terminal=_require_int(
                mapping,
                "terminal",
                label="run_contract.tasks.terminal",
            ),
            nonterminal=_require_int(
                mapping,
                "nonterminal",
                label="run_contract.tasks.nonterminal",
            ),
        )


@dataclass(frozen=True)
class ManagedResearchRunContractArtifacts:
    readiness: str

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchRunContractArtifacts:
        mapping = _require_mapping(payload, label="run_contract.artifacts")
        return cls(
            readiness=_require_string(
                mapping,
                "readiness",
                label="run_contract.artifacts.readiness",
            ),
        )


@dataclass(frozen=True)
class ManagedResearchRunContractExecutionRoute:
    route: str
    model: str | None = None
    pool_required: bool = False
    pool_account_claimed: bool = False
    active_actor_claims: int = 0
    route_reason: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchRunContractExecutionRoute:
        mapping = _require_mapping(payload, label="run_contract.execution_route")
        return cls(
            route=_require_string(
                mapping,
                "route",
                label="run_contract.execution_route.route",
            ),
            model=_optional_string(mapping, "model"),
            pool_required=bool(_optional_bool(mapping, "pool_required") or False),
            pool_account_claimed=bool(_optional_bool(mapping, "pool_account_claimed") or False),
            active_actor_claims=(_optional_int(mapping, "active_actor_claims") or 0),
            route_reason=_optional_string(mapping, "route_reason"),
        )


@dataclass(frozen=True)
class ManagedResearchRunWorkProductArtifactLink:
    work_product_artifact_id: str
    work_product_id: str
    artifact_id: str
    role: str
    label: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchRunWorkProductArtifactLink:
        mapping = _require_mapping(
            payload,
            label="run_contract.work_products.items.artifact_links",
        )
        return cls(
            work_product_artifact_id=_require_string(
                mapping,
                "work_product_artifact_id",
                label=("run_contract.work_products.items.artifact_links.work_product_artifact_id"),
            ),
            work_product_id=_require_string(
                mapping,
                "work_product_id",
                label="run_contract.work_products.items.artifact_links.work_product_id",
            ),
            artifact_id=_require_string(
                mapping,
                "artifact_id",
                label="run_contract.work_products.items.artifact_links.artifact_id",
            ),
            role=_require_string(
                mapping,
                "role",
                label="run_contract.work_products.items.artifact_links.role",
            ),
            label=_optional_string(mapping, "label"),
            metadata=dict(
                _require_mapping(
                    mapping.get("metadata", {}),
                    label="run_contract.work_products.items.artifact_links.metadata",
                )
            ),
        )


@dataclass(frozen=True)
class ManagedResearchRunWorkProduct:
    work_product_id: str
    kind: str
    title: str
    status: str
    readiness: str
    instance_id: str | None = None
    artifact_id: str | None = None
    artifact_links: list[ManagedResearchRunWorkProductArtifactLink] = field(default_factory=list)
    detail_url: str | None = None
    content_url: str | None = None
    supported_export_destinations: list[str] = field(default_factory=list)
    latest_export_id: str | None = None
    blocker: dict[str, object] | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchRunWorkProduct:
        mapping = _require_mapping(payload, label="run_contract.work_products.items")
        destinations = mapping.get("supported_export_destinations")
        if destinations is None:
            normalized_destinations: list[str] = []
        elif isinstance(destinations, list):
            normalized_destinations = [str(item) for item in destinations]
        else:
            raise ValueError(
                "run_contract.work_products.items.supported_export_destinations "
                "must be an array when provided"
            )
        blocker = mapping.get("blocker")
        artifact_links = mapping.get("artifact_links")
        if artifact_links is None:
            normalized_artifact_links: list[ManagedResearchRunWorkProductArtifactLink] = []
        elif isinstance(artifact_links, list):
            normalized_artifact_links = [
                ManagedResearchRunWorkProductArtifactLink.from_wire(item) for item in artifact_links
            ]
        else:
            raise ValueError(
                "run_contract.work_products.items.artifact_links must be an array when provided"
            )
        return cls(
            work_product_id=_require_string(
                mapping,
                "work_product_id",
                label="run_contract.work_products.items.work_product_id",
            ),
            kind=_require_string(
                mapping,
                "kind",
                label="run_contract.work_products.items.kind",
            ),
            title=_require_string(
                mapping,
                "title",
                label="run_contract.work_products.items.title",
            ),
            status=_require_string(
                mapping,
                "status",
                label="run_contract.work_products.items.status",
            ),
            readiness=_require_string(
                mapping,
                "readiness",
                label="run_contract.work_products.items.readiness",
            ),
            instance_id=_optional_string(mapping, "instance_id"),
            artifact_id=_optional_string(mapping, "artifact_id"),
            artifact_links=normalized_artifact_links,
            detail_url=_optional_string(mapping, "detail_url"),
            content_url=_optional_string(mapping, "content_url"),
            supported_export_destinations=normalized_destinations,
            latest_export_id=_optional_string(mapping, "latest_export_id"),
            blocker=dict(_require_mapping(blocker, label="work_product.blocker"))
            if blocker is not None
            else None,
        )


@dataclass(frozen=True)
class ManagedResearchRunWorkProducts:
    total: int = 0
    ready: int = 0
    blocked: int = 0
    failed: int = 0
    items: list[ManagedResearchRunWorkProduct] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchRunWorkProducts:
        if payload is None:
            return cls()
        mapping = _require_mapping(payload, label="run_contract.work_products")
        return cls(
            total=_optional_int(mapping, "total") or 0,
            ready=_optional_int(mapping, "ready") or 0,
            blocked=_optional_int(mapping, "blocked") or 0,
            failed=_optional_int(mapping, "failed") or 0,
            items=[
                ManagedResearchRunWorkProduct.from_wire(item)
                for item in _optional_dict_list(
                    mapping.get("items"),
                    label="run_contract.work_products.items",
                )
            ],
        )


@dataclass(frozen=True)
class ManagedResearchTrainedModel:
    model_id: str
    work_product_id: str | None
    provider: str
    base_model: str
    method: str
    status: str
    export_status: str
    metrics: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchTrainedModel:
        mapping = _require_mapping(payload, label="run_contract.trained_models.models")
        return cls(
            model_id=_require_string(
                mapping,
                "model_id",
                label="run_contract.trained_models.models.model_id",
            ),
            work_product_id=_optional_string(mapping, "work_product_id"),
            provider=_optional_string(mapping, "provider") or "tinker",
            base_model=_require_string(
                mapping,
                "base_model",
                label="run_contract.trained_models.models.base_model",
            ),
            method=_require_string(
                mapping,
                "method",
                label="run_contract.trained_models.models.method",
            ),
            status=_require_string(
                mapping,
                "status",
                label="run_contract.trained_models.models.status",
            ),
            export_status=_optional_string(mapping, "export_status") or "not_exported",
            metrics=_optional_object_dict(mapping.get("metrics")),
        )


@dataclass(frozen=True)
class ManagedResearchRunContractTrainedModels:
    total: int = 0
    registered: int = 0
    exported: int = 0
    evaluated: int = 0
    failed: int = 0
    models: list[ManagedResearchTrainedModel] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchRunContractTrainedModels:
        if payload is None:
            return cls()
        mapping = _require_mapping(payload, label="run_contract.trained_models")
        return cls(
            total=_optional_int(mapping, "total") or 0,
            registered=_optional_int(mapping, "registered") or 0,
            exported=_optional_int(mapping, "exported") or 0,
            evaluated=_optional_int(mapping, "evaluated") or 0,
            failed=_optional_int(mapping, "failed") or 0,
            models=[
                ManagedResearchTrainedModel.from_wire(item)
                for item in _optional_dict_list(
                    mapping.get("models"),
                    label="run_contract.trained_models.models",
                )
            ],
        )


@dataclass(frozen=True)
class ManagedResearchContainerEvalPackage:
    package_id: str
    work_product_id: str | None
    kind: str
    name: str
    version: str | None
    status: str
    validation_status: str
    artifact_id: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchContainerEvalPackage:
        mapping = _require_mapping(
            payload,
            label="run_contract.container_eval_packages.packages",
        )
        return cls(
            package_id=_require_string(
                mapping,
                "package_id",
                label="run_contract.container_eval_packages.packages.package_id",
            ),
            work_product_id=_optional_string(mapping, "work_product_id"),
            kind=_require_string(
                mapping,
                "kind",
                label="run_contract.container_eval_packages.packages.kind",
            ),
            name=_require_string(
                mapping,
                "name",
                label="run_contract.container_eval_packages.packages.name",
            ),
            version=_optional_string(mapping, "version"),
            status=_require_string(
                mapping,
                "status",
                label="run_contract.container_eval_packages.packages.status",
            ),
            validation_status=_require_string(
                mapping,
                "validation_status",
                label=("run_contract.container_eval_packages.packages.validation_status"),
            ),
            artifact_id=_optional_string(mapping, "artifact_id"),
        )


@dataclass(frozen=True)
class ManagedResearchRunContractContainerEvalPackages:
    total: int = 0
    valid: int = 0
    invalid: int = 0
    imported: int = 0
    packages: list[ManagedResearchContainerEvalPackage] = field(default_factory=list)

    @classmethod
    def from_wire(
        cls,
        payload: object,
    ) -> ManagedResearchRunContractContainerEvalPackages:
        if payload is None:
            return cls()
        mapping = _require_mapping(
            payload,
            label="run_contract.container_eval_packages",
        )
        return cls(
            total=_optional_int(mapping, "total") or 0,
            valid=_optional_int(mapping, "valid") or 0,
            invalid=_optional_int(mapping, "invalid") or 0,
            imported=_optional_int(mapping, "imported") or 0,
            packages=[
                ManagedResearchContainerEvalPackage.from_wire(item)
                for item in _optional_dict_list(
                    mapping.get("packages"),
                    label="run_contract.container_eval_packages.packages",
                )
            ],
        )


@dataclass(frozen=True)
class ManagedResearchRunContractIncidents:
    unresolved: int

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchRunContractIncidents:
        mapping = _require_mapping(payload, label="run_contract.incidents")
        return cls(
            unresolved=_require_int(
                mapping,
                "unresolved",
                label="run_contract.incidents.unresolved",
            ),
        )


@dataclass(frozen=True)
class ManagedResearchRunContractDiagnostics:
    lifecycle_invariants: list[dict[str, object]] = field(default_factory=list)
    resource_wait: dict[str, object] | None = None
    failure_classification: dict[str, object] | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchRunContractDiagnostics:
        mapping = _require_mapping(payload, label="run_contract.diagnostics")
        resource_wait = mapping.get("resource_wait")
        failure_classification = mapping.get("failure_classification")
        return cls(
            lifecycle_invariants=_optional_dict_list(
                mapping.get("lifecycle_invariants"),
                label="run_contract.diagnostics.lifecycle_invariants",
            ),
            resource_wait=dict(
                _require_mapping(
                    resource_wait,
                    label="run_contract.diagnostics.resource_wait",
                )
            )
            if resource_wait is not None
            else None,
            failure_classification=dict(
                _require_mapping(
                    failure_classification,
                    label="run_contract.diagnostics.failure_classification",
                )
            )
            if failure_classification is not None
            else None,
        )


@dataclass(frozen=True)
class ManagedResearchRunContract:
    schema_version: str
    project_id: str
    run_id: str
    public_state: ManagedResearchRunState
    terminal: bool
    terminal_outcome: ManagedResearchRunTerminalOutcome | None
    lifecycle: ManagedResearchRunContractLifecycle
    finalization: ManagedResearchRunContractFinalization
    recovery: ManagedResearchRunContractRecovery
    tasks: ManagedResearchRunContractTasks
    artifacts: ManagedResearchRunContractArtifacts
    execution_route: ManagedResearchRunContractExecutionRoute
    work_products: ManagedResearchRunWorkProducts
    trained_models: ManagedResearchRunContractTrainedModels
    container_eval_packages: ManagedResearchRunContractContainerEvalPackages
    incidents: ManagedResearchRunContractIncidents
    diagnostics: ManagedResearchRunContractDiagnostics

    @classmethod
    def from_wire(cls, payload: object) -> ManagedResearchRunContract:
        mapping = _require_mapping(payload, label="run_contract")
        terminal_outcome = _optional_string(mapping, "terminal_outcome")
        return cls(
            schema_version=_require_string(
                mapping,
                "schema_version",
                label="run_contract.schema_version",
            ),
            project_id=_require_string(
                mapping,
                "project_id",
                label="run_contract.project_id",
            ),
            run_id=_require_string(mapping, "run_id", label="run_contract.run_id"),
            public_state=ManagedResearchRunState(
                _require_string(mapping, "public_state", label="run_contract.public_state")
            ),
            terminal=_require_bool(
                mapping,
                "terminal",
                label="run_contract.terminal",
            ),
            terminal_outcome=(
                ManagedResearchRunTerminalOutcome(terminal_outcome)
                if terminal_outcome is not None
                else None
            ),
            lifecycle=ManagedResearchRunContractLifecycle.from_wire(mapping.get("lifecycle")),
            finalization=ManagedResearchRunContractFinalization.from_wire(
                mapping.get("finalization")
            ),
            recovery=ManagedResearchRunContractRecovery.from_wire(mapping.get("recovery")),
            tasks=ManagedResearchRunContractTasks.from_wire(mapping.get("tasks")),
            artifacts=ManagedResearchRunContractArtifacts.from_wire(mapping.get("artifacts")),
            execution_route=ManagedResearchRunContractExecutionRoute.from_wire(
                mapping.get("execution_route")
            ),
            work_products=ManagedResearchRunWorkProducts.from_wire(mapping.get("work_products")),
            trained_models=ManagedResearchRunContractTrainedModels.from_wire(
                mapping.get("trained_models")
            ),
            container_eval_packages=ManagedResearchRunContractContainerEvalPackages.from_wire(
                mapping.get("container_eval_packages")
            ),
            incidents=ManagedResearchRunContractIncidents.from_wire(mapping.get("incidents")),
            diagnostics=ManagedResearchRunContractDiagnostics.from_wire(mapping.get("diagnostics")),
        )


@dataclass(frozen=True)
class RunObservabilitySnapshot:
    schema_version: str
    project_id: str
    run_id: str
    generated_at: str
    run: ManagedResearchRun
    lifecycle: RunLifecycleView
    public_state: ManagedResearchRunState
    terminal_outcome: ManagedResearchRunTerminalOutcome | None
    liveness_phase: ManagedResearchRunLivenessPhase
    status_reason: str | None
    projection_authority: str
    work_completed: bool
    completion_classifier: str | None
    run_contract: ManagedResearchRunContract
    candidate_publication: CandidatePublicationView
    actors: ActorCollectionSnapshot
    tasks: TaskCollectionSnapshot
    runtime: RuntimeObservability
    recent_project_events: list[dict[str, object]] = field(default_factory=list)
    latest_event_seq: int | None = None
    open_questions: list[dict[str, object]] = field(default_factory=list)
    anomalies: list[RunAnomaly] = field(default_factory=list)
    cursor: RunObservationCursor = field(default_factory=RunObservationCursor)

    @classmethod
    def from_wire(cls, payload: object) -> RunObservabilitySnapshot:
        mapping = _require_mapping(payload, label="run observability snapshot")
        return cls(
            schema_version=_require_string(
                mapping, "schema_version", label="snapshot.schema_version"
            ),
            project_id=_require_string(mapping, "project_id", label="snapshot.project_id"),
            run_id=_require_string(mapping, "run_id", label="snapshot.run_id"),
            generated_at=_require_string(mapping, "generated_at", label="snapshot.generated_at"),
            run=ManagedResearchRun.from_wire(mapping.get("run")),
            lifecycle=RunLifecycleView.from_wire(mapping.get("lifecycle")),
            public_state=ManagedResearchRunState(
                _require_string(mapping, "public_state", label="snapshot.public_state")
            ),
            terminal_outcome=(
                ManagedResearchRunTerminalOutcome(
                    _require_string(
                        mapping,
                        "terminal_outcome",
                        label="snapshot.terminal_outcome",
                    )
                )
                if _optional_string(mapping, "terminal_outcome") is not None
                else None
            ),
            liveness_phase=_parse_liveness_phase(_optional_string(mapping, "liveness_phase")),
            status_reason=_optional_string(mapping, "status_reason"),
            projection_authority=(
                _optional_string(mapping, "projection_authority")
                or "backend_public_run_state_projection.v1"
            ),
            work_completed=bool(_optional_bool(mapping, "work_completed")),
            completion_classifier=_optional_string(mapping, "completion_classifier"),
            run_contract=ManagedResearchRunContract.from_wire(mapping.get("run_contract")),
            candidate_publication=CandidatePublicationView.from_wire(
                mapping.get("candidate_publication")
            ),
            actors=ActorCollectionSnapshot.from_wire(mapping.get("actors")),
            tasks=TaskCollectionSnapshot.from_wire(mapping.get("tasks")),
            runtime=RuntimeObservability.from_wire(mapping.get("runtime")),
            recent_project_events=_optional_dict_list(
                mapping.get("recent_project_events"),
                label="snapshot.recent_project_events",
            ),
            latest_event_seq=_optional_int(mapping, "latest_event_seq"),
            open_questions=_optional_dict_list(
                mapping.get("open_questions"),
                label="snapshot.open_questions",
            ),
            anomalies=[
                RunAnomaly.from_wire(item)
                for item in _optional_dict_list(
                    mapping.get("anomalies"), label="snapshot.anomalies"
                )
            ],
            cursor=RunObservationCursor.from_wire(mapping.get("cursor")),
        )


__all__ = [
    "ActorCollectionSnapshot",
    "ActorSnapshot",
    "CandidatePublicationOutcome",
    "CandidatePublicationView",
    "MessageQueueInteraction",
    "MessageQueueMessage",
    "MessageQueueThread",
    "ManagedResearchRun",
    "ManagedResearchRunLivenessPhase",
    "ManagedResearchRunContract",
    "ManagedResearchRunContractArtifacts",
    "ManagedResearchRunContractContainerEvalPackages",
    "ManagedResearchRunContractDiagnostics",
    "ManagedResearchRunContractExecutionRoute",
    "ManagedResearchRunContractFinalization",
    "ManagedResearchRunContractIncidents",
    "ManagedResearchRunContractLifecycle",
    "ManagedResearchRunContractRecovery",
    "ManagedResearchRunContractTasks",
    "ManagedResearchRunContractTrainedModels",
    "ManagedResearchRunWorkProduct",
    "ManagedResearchRunWorkProductArtifactLink",
    "ManagedResearchRunWorkProducts",
    "ManagedResearchContainerEvalPackage",
    "ManagedResearchTrainedModel",
    "ManagedResearchRunState",
    "ManagedResearchRunTerminalOutcome",
    "RunAnomaly",
    "RunAnomalyKind",
    "RunLifecycleDispatch",
    "RunLifecycleFailure",
    "RunLifecycleLocalExecution",
    "RunLifecycleView",
    "RunObservationCursor",
    "RunObservabilitySnapshot",
    "RunManualTickRequest",
    "RunTickingStatus",
    "RunTickingUpdate",
    "RunTickMode",
    "RuntimeDeliveryView",
    "RuntimeEventView",
    "RuntimeMessageView",
    "RuntimeObservability",
    "TaskCollectionSnapshot",
    "TaskSnapshot",
    "TaskSummary",
]
