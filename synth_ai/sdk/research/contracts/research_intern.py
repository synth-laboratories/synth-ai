"""Typed contracts for one organization Research Intern and its resources.

Backend remains the contract authority. These models intentionally preserve
scoped policies and owner-authored evidence instead of flattening them into an
SDK-side source of truth.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from synth_ai.sdk.research.contracts.dataset_revisions import (
    DatasetRevisionCreateRequest,
    DatasetRevisionLifecycleRequest,
    DatasetRevisionResponse,
)
from synth_ai.sdk.research.contracts.project_runtime import ProjectComputerState
from synth_ai.sdk.research.contracts.project_workspace_evidence import (
    ProjectComputerWorkspaceSnapshot,
)
from synth_ai.sdk.research.contracts.traces import TracePromotionReceipt


class _StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_wire(cls, value: object) -> Self:
        return cls.model_validate(value)


class ResearchInternStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class MagiMode(StrEnum):
    SYNC = "sync"
    ASYNC = "async"
    SERAPH = "seraph"


class MagiCanonicalUser(StrEnum):
    CASPER = "Casper"
    MELCHIOR = "Melchior"
    BALTHASAR = "Balthasar"


MAGI_CANONICAL_USER_BY_MODE: dict[MagiMode, MagiCanonicalUser] = {
    MagiMode.SYNC: MagiCanonicalUser.CASPER,
    MagiMode.ASYNC: MagiCanonicalUser.MELCHIOR,
    MagiMode.SERAPH: MagiCanonicalUser.BALTHASAR,
}


class MagiDecisionKind(StrEnum):
    DELEGATE = "delegate"
    INSPECT = "inspect"
    PAUSE = "pause"
    INTERVENE = "intervene"
    RESUME = "resume"
    VERDICT = "verdict"
    REVISE = "revise"


class MagiActuationStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    APPLIED = "applied"
    NOOP = "noop"


class MagiDecisionActuationReceipt(_StrictContract):
    """Durable proof that a control decision reached its runtime authority."""

    status: MagiActuationStatus
    factory_id: str | None = None
    project_id: str | None = None
    effort_id: str | None = None
    run_id: str | None = None
    scheduler_action: Literal["pause", "resume"] | None = None
    scheduler_decision: str | None = None
    factory_status: str | None = None
    run_action: Literal["pause", "intervene", "resume"] | None = None
    run_state: str | None = None
    runtime_message_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class ResearchInternPolicySet(_StrictContract):
    organization: dict[str, Any] = Field(default_factory=dict)
    team: dict[str, Any] = Field(default_factory=dict)
    user: dict[str, Any] = Field(default_factory=dict)
    organization_policy_ref: str | None = None
    team_policy_ref: str | None = None
    user_policy_ref: str | None = None


class ResearchInternProvisionRequest(_StrictContract):
    display_name: str = Field(default="Research Intern", min_length=1, max_length=255)
    policies: ResearchInternPolicySet = Field(default_factory=ResearchInternPolicySet)
    attribution_team_id: str | None = Field(default=None, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchInternPatchRequest(_StrictContract):
    display_name: str | None = Field(default=None, min_length=1, max_length=255)
    status: ResearchInternStatus | None = None
    policies: ResearchInternPolicySet | None = None
    attribution_team_id: str | None = Field(default=None, max_length=255)
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def require_change(self) -> ResearchInternPatchRequest:
        if not self.model_fields_set:
            raise ValueError("ResearchInternPatchRequest must change at least one field")
        return self

    def to_wire(self) -> dict[str, Any]:
        """Preserve explicit nulls so attribution and metadata can be cleared."""
        return self.model_dump(mode="json", exclude_unset=True)


class ResearchInternResponse(_StrictContract):
    research_intern_id: str
    org_id: str
    display_name: str
    status: ResearchInternStatus
    policies: ResearchInternPolicySet
    attribution_user_id: str | None = None
    attribution_team_id: str | None = None
    state_generation: int = Field(ge=0)
    durable_state: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class InternRuntimeBinding(_StrictContract):
    factory_id: str | None = None
    project_id: str | None = None
    effort_id: str | None = None
    run_id: str | None = None


class InternSyncStatus(StrEnum):
    CREATED = "created"
    READY = "ready"
    THINKING = "thinking"
    WAITING_FOR_OPERATOR = "waiting_for_operator"
    PAUSED = "paused"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


class InternRuntimeOutcome(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class InternSyncCommandKind(StrEnum):
    OPERATOR_MESSAGE = "operator_message"
    INTERVENE = "intervene"
    ANSWER_INTERACTION = "answer_interaction"
    PAUSE = "pause"
    RESUME = "resume"
    CLOSE = "close"


class InternSyncSessionCreateRequest(_StrictContract):
    objective: str = Field(min_length=1, max_length=20_000)
    idempotency_key: str = Field(min_length=1, max_length=512)
    binding: InternRuntimeBinding = Field(default_factory=InternRuntimeBinding)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InternSyncSession(_StrictContract):
    schema_version: Literal["smr.intern-sync-session.v1"]
    sync_session_id: str
    research_intern_id: str
    org_id: str
    objective: str
    status: InternSyncStatus
    state_generation: int = Field(ge=0)
    last_event_sequence: int = Field(ge=0)
    binding: InternRuntimeBinding
    pending_turn_id: str | None = None
    pending_action_id: str | None = None
    pending_interaction_id: str | None = None
    outcome: InternRuntimeOutcome | None = None
    failure_code: str | None = None
    temporal_workflow_id: str
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None


class InternSyncCommandRequest(_StrictContract):
    command_id: str = Field(min_length=1, max_length=512)
    idempotency_key: str = Field(min_length=1, max_length=512)
    expected_generation: int = Field(ge=0)
    command_kind: InternSyncCommandKind
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_command_payload(self) -> InternSyncCommandRequest:
        def required_text(*names: str) -> str:
            for name in names:
                value = self.payload.get(name)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            raise ValueError(f"{self.command_kind.value} requires {' or '.join(names)}")

        if self.command_kind in {
            InternSyncCommandKind.OPERATOR_MESSAGE,
            InternSyncCommandKind.INTERVENE,
        }:
            required_text("body")
        elif self.command_kind is InternSyncCommandKind.ANSWER_INTERACTION:
            required_text("interaction_id")
            required_text("body", "answer")
        elif self.command_kind is InternSyncCommandKind.PAUSE:
            required_text("reason", "rationale")
        elif self.command_kind is InternSyncCommandKind.CLOSE:
            outcome = required_text("outcome", "status")
            if outcome not in {item.value for item in InternRuntimeOutcome}:
                raise ValueError("close requires a valid runtime outcome")
            required_text("reason", "rationale")
        return self


class InternSyncCommandReceipt(_StrictContract):
    schema_version: Literal["smr.intern-runtime-command-receipt.v1"]
    command_id: str
    runtime_kind: Literal["sync"]
    runtime_id: str
    status: Literal[
        "received",
        "delivered",
        "applied",
        "noop",
        "refused",
        "superseded",
        "conflict",
    ]
    previous_generation: int = Field(ge=0)
    state_generation: int = Field(ge=0)
    decision_code: str
    created_at: datetime


class InternSyncEvent(_StrictContract):
    schema_version: Literal["smr.intern-runtime-event.v1"]
    event_id: str
    runtime_kind: Literal["sync"]
    runtime_id: str
    sequence: int = Field(ge=1)
    previous_state_generation: int = Field(ge=0)
    state_generation: int = Field(ge=1)
    event_kind: str
    command_id: str
    payload: dict[str, Any]
    created_at: datetime


class InternSyncEventStreamEnvelope(_StrictContract):
    schema_version: Literal["smr.intern-runtime-event-stream.v1"]
    event: InternSyncEvent


class InternSyncEventPage(_StrictContract):
    """SDK-local page over the backend's bare Sync event array."""

    events: tuple[InternSyncEvent, ...]
    next_sequence: int = Field(ge=0)


class InternAsyncStatus(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    EXECUTING_CYCLE = "executing_cycle"
    CHECKPOINTING = "checkpointing"
    SLEEPING = "sleeping"
    RECONCILING = "reconciling"
    AWAITING_INPUT = "awaiting_input"
    AWAITING_EVIDENCE = "awaiting_evidence"
    PAUSED = "paused"
    BLOCKED = "blocked"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class InternAsyncExternalExecutionStatus(StrEnum):
    NOT_STARTED = "not_started"
    ACTIVE = "active"
    TERMINAL = "terminal"


class InternAsyncEvidenceReadiness(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    FINALIZING = "finalizing"
    READY = "ready"
    INCOMPLETE = "incomplete"


class InternAsyncInstructionKind(StrEnum):
    MESSAGE = "message"
    INTERVENE = "intervene"
    REDIRECT_OBJECTIVE = "redirect_objective"
    REQUEST_CHECKPOINT = "request_checkpoint"


class InternAsyncCommandKind(StrEnum):
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    PROVIDE_INPUT = "provide_input"
    ANSWER_INTERACTION = "answer_interaction"
    MESSAGE = "message"
    INTERVENE = "intervene"
    REDIRECT_OBJECTIVE = "redirect_objective"
    REQUEST_CHECKPOINT = "request_checkpoint"


class InternAsyncRuntimeBudget(_StrictContract):
    maximum_cost_cents: int | None = Field(default=None, ge=0)
    maximum_cycles: int | None = Field(default=None, ge=1)
    maximum_concurrent_runs: int = Field(default=1, ge=1)


class InternAsyncEnsureRequest(_StrictContract):
    objective: str = Field(min_length=1, max_length=20_000)
    idempotency_key: str = Field(min_length=1, max_length=512)
    binding: InternRuntimeBinding = Field(default_factory=InternRuntimeBinding)
    budget: InternAsyncRuntimeBudget = Field(default_factory=InternAsyncRuntimeBudget)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InternAsyncCheckpoint(_StrictContract):
    checkpoint_id: str
    summary: str
    evidence_refs: list[str]
    unresolved_questions: list[str]
    next_action: str | None = None
    research_records: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime


class InternAsyncBlocker(_StrictContract):
    code: str
    message: str
    retryable: bool
    next_retry_at: datetime | None = None
    operator_action_required: bool = False


class InternAsyncRuntime(_StrictContract):
    schema_version: Literal["smr.intern-async-runtime.v1"]
    async_runtime_id: str
    async_assignment_id: str
    cardinality: Literal["one_per_organization"]
    instance_kind: Literal["organization_async_intern"]
    research_intern_id: str
    org_id: str
    objective: str
    status: InternAsyncStatus
    state_generation: int = Field(ge=0)
    last_event_sequence: int = Field(ge=0)
    cycle_number: int = Field(ge=0)
    plan: dict[str, Any] = Field(default_factory=dict)
    pending_interaction_id: str | None = None
    pending_action_id: str | None = None
    pending_actor_message_id: str | None = None
    pending_instruction_count: int = Field(default=0, ge=0, le=32)
    binding: InternRuntimeBinding
    external_execution_status: InternAsyncExternalExecutionStatus
    evidence_readiness: InternAsyncEvidenceReadiness
    next_wake_at: datetime | None = None
    checkpoint: InternAsyncCheckpoint | None = None
    budget: InternAsyncRuntimeBudget
    blocker: InternAsyncBlocker | None = None
    temporal_workflow_id: str
    leave_safe: Literal[True]
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_runtime_identity_and_evidence(self) -> InternAsyncRuntime:
        if self.async_runtime_id != self.async_assignment_id:
            raise ValueError("Async Intern compatibility identity drifted")
        if (
            self.status is InternAsyncStatus.COMPLETED
            and self.evidence_readiness is not InternAsyncEvidenceReadiness.READY
        ):
            raise ValueError("completed Async Intern requires ready evidence")
        return self


class InternAsyncCommandRequest(_StrictContract):
    command_id: str = Field(min_length=1, max_length=512)
    idempotency_key: str = Field(min_length=1, max_length=512)
    expected_generation: int = Field(ge=0)
    command_kind: InternAsyncCommandKind
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_command_payload(self) -> InternAsyncCommandRequest:
        def required_text(name: str) -> str:
            value = self.payload.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{self.command_kind.value} requires {name}")
            return value.strip()

        if self.command_kind in {
            InternAsyncCommandKind.PAUSE,
            InternAsyncCommandKind.CANCEL,
        }:
            required_text("reason")
        elif self.command_kind in {
            InternAsyncCommandKind.PROVIDE_INPUT,
            InternAsyncCommandKind.ANSWER_INTERACTION,
        }:
            required_text("interaction_id")
            required_text("body")
        elif self.command_kind in {
            InternAsyncCommandKind.MESSAGE,
            InternAsyncCommandKind.INTERVENE,
            InternAsyncCommandKind.REDIRECT_OBJECTIVE,
        }:
            required_text("body")
        elif self.command_kind is InternAsyncCommandKind.REQUEST_CHECKPOINT and self.payload.get(
            "body"
        ):
            raise ValueError("request_checkpoint does not accept body")
        return self


class InternAsyncInstructionRequest(_StrictContract):
    command_id: str = Field(min_length=1, max_length=512)
    idempotency_key: str = Field(min_length=1, max_length=512)
    expected_generation: int = Field(ge=0)
    instruction_kind: InternAsyncInstructionKind
    body: str | None = Field(default=None, max_length=20_000)
    context: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_instruction_body(self) -> InternAsyncInstructionRequest:
        body = self.body.strip() if self.body else None
        if self.instruction_kind is InternAsyncInstructionKind.REQUEST_CHECKPOINT:
            if body:
                raise ValueError("request_checkpoint does not accept body")
        elif body is None:
            raise ValueError("instruction body is required")
        return self

    def to_command(self) -> InternAsyncCommandRequest:
        return InternAsyncCommandRequest(
            command_id=self.command_id,
            idempotency_key=self.idempotency_key,
            expected_generation=self.expected_generation,
            command_kind=InternAsyncCommandKind(self.instruction_kind.value),
            payload={"body": self.body, "context": self.context},
        )


class InternAsyncCommandReceipt(_StrictContract):
    schema_version: Literal["smr.intern-runtime-command-receipt.v1"]
    command_id: str
    runtime_kind: Literal["async"]
    runtime_id: str
    status: Literal[
        "received",
        "delivered",
        "applied",
        "noop",
        "refused",
        "superseded",
        "conflict",
    ]
    previous_generation: int = Field(ge=0)
    state_generation: int = Field(ge=0)
    decision_code: str
    created_at: datetime


class InternAsyncEvent(_StrictContract):
    schema_version: Literal["smr.intern-runtime-event.v1"]
    event_id: str
    runtime_kind: Literal["async"]
    runtime_id: str
    sequence: int = Field(ge=1)
    previous_state_generation: int = Field(ge=0)
    state_generation: int = Field(ge=1)
    event_kind: str
    command_id: str
    payload: dict[str, Any]
    created_at: datetime


class InternAsyncEventStreamEnvelope(_StrictContract):
    schema_version: Literal["smr.intern-runtime-event-stream.v1"]
    event: InternAsyncEvent


class InternAsyncEventPage(_StrictContract):
    """SDK-local page over the backend's bare event array."""

    events: tuple[InternAsyncEvent, ...]
    next_sequence: int = Field(ge=0)


class ResearchInternFactoryMembershipResponse(_StrictContract):
    research_intern_id: str
    org_id: str
    factory_id: str
    role: str
    attached_by_user_id: str | None = None
    attached_at: datetime


class MagiDecisionRequest(_StrictContract):
    mode: MagiMode
    decision_kind: MagiDecisionKind
    idempotency_key: str = Field(min_length=1, max_length=512)
    factory_id: str | None = None
    project_id: str | None = None
    effort_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    expected_state_generation: int | None = Field(default=None, ge=0)
    experiment_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    state_patch: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(min_length=1, max_length=20_000)
    verdict: str | None = Field(default=None, max_length=255)
    uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_seraph_verdict(self) -> MagiDecisionRequest:
        if self.decision_kind is MagiDecisionKind.VERDICT and (
            self.mode is not MagiMode.SERAPH or not self.verdict
        ):
            raise ValueError("verdict decisions require Seraph mode and verdict")
        if self.decision_kind in {
            MagiDecisionKind.PAUSE,
            MagiDecisionKind.INTERVENE,
            MagiDecisionKind.RESUME,
        }:
            missing = [
                name
                for name in ("factory_id", "project_id", "effort_id", "run_id")
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(
                    "control decisions require exact factory/project/effort/run "
                    f"bindings; missing {', '.join(missing)}"
                )
        if self.decision_kind is MagiDecisionKind.INTERVENE and not self.state_patch:
            raise ValueError("intervene decisions require a non-empty state_patch")
        return self


class MagiDecisionReceiptResponse(_StrictContract):
    schema_version: Literal["smr.magi-decision-receipt.v2"] = "smr.magi-decision-receipt.v2"
    receipt_id: str
    content_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    receipt_url: str = Field(min_length=1, max_length=4096)
    research_intern_id: str
    org_id: str
    factory_id: str | None = None
    project_id: str | None = None
    effort_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    experiment_id: str | None = None
    mode: MagiMode
    canonical_user: MagiCanonicalUser
    decision_kind: MagiDecisionKind
    idempotency_key: str
    expected_state_generation: int | None = Field(default=None, ge=0)
    previous_state_generation: int = Field(ge=0)
    state_generation: int = Field(ge=0)
    evidence_refs: list[str]
    state_patch: dict[str, Any]
    rationale: str
    verdict: str | None = None
    uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)
    actuation: MagiDecisionActuationReceipt
    decided_by_user_id: str | None = None
    created_at: datetime

    @model_validator(mode="after")
    def require_canonical_mode_user(self) -> MagiDecisionReceiptResponse:
        if self.canonical_user != MAGI_CANONICAL_USER_BY_MODE[self.mode]:
            raise ValueError("canonical_user does not match Magi mode")
        return self


class ResearchInternSessionStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    STOPPED = "stopped"
    CANCELED = "canceled"
    ARCHIVED = "archived"


class ResearchInternEventKind(StrEnum):
    OBJECTIVE = "objective"
    OPERATOR_MESSAGE = "operator_message"
    AGENT_MESSAGE = "agent_message"
    PROGRESS = "progress"
    MAGI_DECISION = "magi_decision"
    STATE_SNAPSHOT = "state_snapshot"
    ERROR = "error"
    CLOSED = "closed"


class ResearchInternEventActorKind(StrEnum):
    OPERATOR = "operator"
    RESEARCH_INTERN = "research_intern"
    MAGI = "magi"
    SYSTEM = "system"


class ResearchInternSessionCreateRequest(_StrictContract):
    factory_id: str = Field(min_length=1, max_length=255)
    project_id: str = Field(min_length=1, max_length=255)
    effort_id: str = Field(min_length=1, max_length=255)
    run_id: str | None = Field(default=None, max_length=255)
    objective: str = Field(min_length=1, max_length=20_000)
    objective_bounds: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=1, max_length=512)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchInternSessionResponse(_StrictContract):
    session_id: str
    research_intern_id: str
    org_id: str
    factory_id: str
    project_id: str
    effort_id: str
    run_id: str | None = None
    objective: str
    objective_bounds: dict[str, Any]
    status: ResearchInternSessionStatus
    state_generation: int = Field(ge=0)
    last_event_sequence: int = Field(ge=0)
    metadata: dict[str, Any]
    created_by_user_id: str | None = None
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None


class ResearchInternEventAppendRequest(_StrictContract):
    event_kind: ResearchInternEventKind
    mode: MagiMode | None = None
    idempotency_key: str = Field(min_length=1, max_length=512)
    expected_state_generation: int = Field(ge=0)
    body: str | None = Field(default=None, max_length=20_000)
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_public_event(self) -> ResearchInternEventAppendRequest:
        if self.event_kind in {
            ResearchInternEventKind.OBJECTIVE,
            ResearchInternEventKind.AGENT_MESSAGE,
            ResearchInternEventKind.MAGI_DECISION,
            ResearchInternEventKind.CLOSED,
        }:
            raise ValueError(f"{self.event_kind.value} is emitted by the server")
        if self.event_kind is ResearchInternEventKind.OPERATOR_MESSAGE and self.mode is not None:
            raise ValueError("operator_message does not accept mode")
        if self.event_kind is ResearchInternEventKind.OPERATOR_MESSAGE and not self.body:
            raise ValueError("operator_message requires body")
        if not self.body and not self.payload:
            raise ValueError("event requires body or payload")
        return self


class ResearchInternEventResponse(_StrictContract):
    schema_version: Literal["smr.research-intern-event.v1"] = "smr.research-intern-event.v1"
    event_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    session_id: str
    research_intern_id: str
    org_id: str
    sequence: int = Field(ge=1)
    previous_state_generation: int = Field(ge=0)
    state_generation: int = Field(ge=0)
    event_kind: ResearchInternEventKind
    actor_kind: ResearchInternEventActorKind
    actor_id: str | None = None
    mode: MagiMode | None = None
    canonical_user: MagiCanonicalUser | None = None
    receipt_id: str | None = None
    idempotency_key: str
    body: str | None = None
    payload: dict[str, Any]
    evidence_refs: list[str]
    created_at: datetime

    @model_validator(mode="after")
    def validate_state_generation_chain(self) -> ResearchInternEventResponse:
        if self.state_generation != self.previous_state_generation + 1:
            raise ValueError(
                "event state_generation must immediately follow previous_state_generation"
            )
        return self


class ResearchInternEventStreamCursor(_StrictContract):
    """Exact durable event-log position carried by the Intern SSE API."""

    schema_version: Literal["smr.research-intern-event-stream-cursor.v1"] = (
        "smr.research-intern-event-stream-cursor.v1"
    )
    after_sequence: int = Field(ge=1)
    event_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    state_generation: int = Field(ge=1)


class ResearchInternEventStreamEvent(_StrictContract):
    """One durable Research Intern event framed by the backend stream."""

    schema_version: Literal["smr.research-intern-event-stream.v1"] = (
        "smr.research-intern-event-stream.v1"
    )
    kind: Literal["event"] = "event"
    session_id: str
    cursor: ResearchInternEventStreamCursor
    event: ResearchInternEventResponse

    @model_validator(mode="after")
    def validate_cursor_chain(self) -> ResearchInternEventStreamEvent:
        if (
            self.session_id != self.event.session_id
            or self.cursor.after_sequence != self.event.sequence
            or self.cursor.event_id != self.event.event_id
            or self.cursor.state_generation != self.event.state_generation
        ):
            raise ValueError("event stream cursor must identify the framed event")
        return self


class ResearchInternEventStreamHeartbeat(_StrictContract):
    """Non-durable liveness frame whose cursor names the last durable event."""

    schema_version: Literal["smr.research-intern-event-stream.v1"] = (
        "smr.research-intern-event-stream.v1"
    )
    kind: Literal["heartbeat"] = "heartbeat"
    session_id: str
    cursor: ResearchInternEventStreamCursor | None = None
    reconnect_after_ms: int = Field(ge=1_000, le=30_000)
    emitted_at: datetime


ResearchInternEventStreamPayload = Annotated[
    ResearchInternEventStreamEvent | ResearchInternEventStreamHeartbeat,
    Field(discriminator="kind"),
]


class ResearchInternEventStreamEnvelope(RootModel[ResearchInternEventStreamPayload]):
    """OpenAPI-visible discriminated union for Intern SSE data payloads."""


class ResearchInternSessionCloseRequest(_StrictContract):
    idempotency_key: str = Field(min_length=1, max_length=512)
    expected_state_generation: int = Field(ge=0)
    status: Literal["completed", "partial", "failed", "stopped", "canceled", "archived"]
    rationale: str = Field(min_length=1, max_length=20_000)
    evidence_refs: list[str] = Field(default_factory=list)


class ResearchInternTracePublicationRequest(_StrictContract):
    """Optimistic fence for publishing one terminal Intern event chain."""

    schema_version: Literal["smr.research-intern-trace-publication-request.v1"] = (
        "smr.research-intern-trace-publication-request.v1"
    )
    idempotency_key: str = Field(min_length=1, max_length=512)
    expected_state_generation: int = Field(ge=1)


class ResearchInternTracePublicationResponse(_StrictContract):
    """Factory Trace Store receipt for one genuine terminal Intern Trace V5."""

    schema_version: Literal["smr.research-intern-trace-publication.v1"] = (
        "smr.research-intern-trace-publication.v1"
    )
    session_id: str
    research_intern_id: str
    idempotency_key: str = Field(min_length=1, max_length=512)
    org_id: str
    factory_id: str
    project_id: str
    effort_id: str
    run_id: str
    state_generation: int = Field(ge=1)
    event_count: int = Field(ge=1)
    trace_id: str
    trace_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    capture_id: str
    bundle_id: str
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    promotion: TracePromotionReceipt

    @model_validator(mode="after")
    def validate_promotion_identity(self) -> ResearchInternTracePublicationResponse:
        if (
            self.factory_id != self.promotion.factory_id
            or self.bundle_id != self.promotion.bundle_id
            or self.manifest_digest != self.promotion.manifest_digest
            or self.promotion.trace_digests != [self.trace_digest]
        ):
            raise ValueError("Research Intern trace response must match its promotion receipt")
        return self


class ResearchInternSessionSyncResponse(_StrictContract):
    schema_version: Literal["smr.research-intern-session-sync.v1"] = (
        "smr.research-intern-session-sync.v1"
    )
    session: ResearchInternSessionResponse
    events: list[ResearchInternEventResponse]
    source_run_id: str
    projected_count: int = Field(ge=0)
    has_more: bool


class ResearchInternTurnControl(StrEnum):
    PAUSE = "pause"
    INTERVENE = "intervene"
    RESUME = "resume"


class ResearchInternTurnStatus(StrEnum):
    ACCEPTED = "accepted"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


class ResearchInternTurnRequest(_StrictContract):
    """One browser-callable operator turn against the bound real runtime."""

    body: str = Field(min_length=1, max_length=20_000)
    mode: MagiMode = MagiMode.SYNC
    idempotency_key: str = Field(min_length=1, max_length=512)
    expected_session_state_generation: int = Field(ge=0)
    expected_intern_state_generation: int = Field(ge=0)
    control: ResearchInternTurnControl | None = None
    rationale: str | None = Field(default=None, max_length=20_000)
    state_patch: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    wait_timeout_seconds: float = Field(default=15.0, ge=0.0, le=30.0)
    poll_interval_ms: int = Field(default=250, ge=100, le=2_000)

    @model_validator(mode="after")
    def validate_turn_control(self) -> ResearchInternTurnRequest:
        if self.control is not None and not str(self.rationale or "").strip():
            raise ValueError("controlled turns require rationale")
        if self.control is ResearchInternTurnControl.INTERVENE and not self.state_patch:
            raise ValueError("intervene turns require a non-empty state_patch")
        if self.control is None and self.state_patch:
            raise ValueError("state_patch requires a controlled turn")
        return self


class ResearchInternTurnError(_StrictContract):
    error_code: str
    message: str
    retryable: bool
    detail: dict[str, Any] = Field(default_factory=dict)


class ResearchInternTurnResponse(_StrictContract):
    schema_version: Literal["smr.research-intern-turn.v1"] = "smr.research-intern-turn.v1"
    turn_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    status: ResearchInternTurnStatus
    session: ResearchInternSessionResponse
    run_id: str
    operator_event: ResearchInternEventResponse
    decision_receipt: MagiDecisionReceiptResponse | None = None
    agent_event: ResearchInternEventResponse | None = None
    projected_events: list[ResearchInternEventResponse] = Field(default_factory=list)
    reconnect_after_sequence: int = Field(ge=0)
    waited_seconds: float = Field(ge=0)
    replayed: bool
    error: ResearchInternTurnError | None = None


class ResearchInternAcceptanceReceiptPublicationRequest(_StrictContract):
    schema_version: Literal["smr.research-intern-acceptance-receipt-publication.v1"] = (
        "smr.research-intern-acceptance-receipt-publication.v1"
    )
    receipt_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    lane: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    candidate_id: str = Field(min_length=1, max_length=255)
    receipt: dict[str, Any]


class ResearchInternAcceptanceReceiptPublicationResponse(_StrictContract):
    schema_version: Literal["smr.research-intern-acceptance-receipt-publication.v1"] = (
        "smr.research-intern-acceptance-receipt-publication.v1"
    )
    receipt_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    receipt_url: str = Field(min_length=1, max_length=4096)
    content_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    org_id: str
    lane: str
    candidate_id: str
    receipt: dict[str, Any]
    replayed: bool
    created_at: datetime


ProjectComputerLifecycle = ProjectComputerState


class ProjectComputerProvisionRequest(_StrictContract):
    factory_id: str = Field(min_length=1, max_length=255)
    cloud_deployment_id: str = Field(min_length=1, max_length=255)
    source_repository_id: str = Field(min_length=1, max_length=255)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    snapshot_digest: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    workspace_snapshot: ProjectComputerWorkspaceSnapshot | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_snapshot_binding(self) -> ProjectComputerProvisionRequest:
        if (self.snapshot_digest is None) != (self.workspace_snapshot is None):
            raise ValueError("snapshot_digest and workspace_snapshot must be supplied together")
        if self.workspace_snapshot is not None and (
            self.snapshot_digest != self.workspace_snapshot.manifest.manifest_digest
            or self.source_revision != self.workspace_snapshot.commit_sha
        ):
            raise ValueError("Project Computer source does not match its snapshot")
        return self


class ProjectComputerResponse(_StrictContract):
    project_computer_id: str
    org_id: str
    research_intern_id: str
    factory_id: str
    project_id: str
    cloud_deployment_id: str
    adapter_kind: str
    provider_kind: str
    source_repository_id: str
    source_revision: str
    snapshot_digest: str | None = None
    lifecycle: ProjectComputerLifecycle
    generation: int = Field(ge=0)
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ProjectComputerProvisionReceipt(_StrictContract):
    org_id: str
    factory_id: str
    project_id: str
    generation: int
    cloud_deployment_id: str
    adapter_kind: str
    deployment_state: str = Field(min_length=1, max_length=100)
    source_repository_id: str = Field(min_length=1, max_length=255)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    snapshot_digest: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    ready: Literal[True]
    workspace_head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    workspace_clean: Literal[True]
    workspace_restore_receipt: dict[str, Any] | None = None
    receipt_ref: str = Field(min_length=1, max_length=1000)


class ProjectComputerRestorationReceipt(_StrictContract):
    org_id: str
    factory_id: str
    project_id: str
    generation: int
    cloud_deployment_id: str
    adapter_kind: str
    deployment_state: str = Field(min_length=1, max_length=100)
    source_repository_id: str
    source_revision: str
    snapshot_digest: str
    restored: Literal[True]
    workspace_restore_receipt: dict[str, Any]
    previous_cloud_deployment_retired: Literal[True]
    previous_retirement_receipt_ref: str = Field(min_length=1, max_length=1000)
    receipt_ref: str = Field(min_length=1, max_length=1000)


class ProjectComputerRetirementReceipt(_StrictContract):
    org_id: str
    factory_id: str
    project_id: str
    generation: int
    cloud_deployment_id: str
    adapter_kind: str
    deployment_state: str = Field(min_length=1, max_length=100)
    retired: bool
    receipt_ref: str = Field(min_length=1, max_length=1000)


class ProjectComputerCleanupRequest(_StrictContract):
    idempotency_key: str = Field(min_length=1, max_length=512)


class ProjectComputerCleanupReceiptResponse(_StrictContract):
    schema_version: Literal["smr.project-computer-cleanup.v1"]
    service_origin: Literal["urn:synth:research-intern:project-computer"]
    owner: Literal["research_intern_control_plane"]
    receipt_id: str
    content_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    receipt_uri: str = Field(min_length=1, max_length=2000)
    org_id: str
    research_intern_id: str
    factory_id: str
    idempotency_key: str
    pre_inventory: list[ProjectComputerResponse]
    workspace_materialization_receipts: list[
        ProjectComputerProvisionReceipt | ProjectComputerRestorationReceipt
    ]
    retirement_receipts: list[ProjectComputerRetirementReceipt]
    post_inventory: list[ProjectComputerResponse]
    cleanup_complete: bool
    created_at: datetime

    @model_validator(mode="after")
    def validate_owner_receipt(self) -> ProjectComputerCleanupReceiptResponse:
        if self.receipt_id != self.content_digest:
            raise ValueError("cleanup receipt_id must equal its content_digest")
        if self.content_digest.removeprefix("sha256:") not in self.receipt_uri:
            raise ValueError("cleanup receipt URI must contain its content digest")
        if self.cleanup_complete != (not self.post_inventory):
            raise ValueError("cleanup_complete must match the post-cleanup inventory")
        if any(
            computer.factory_id != self.factory_id
            for computer in (*self.pre_inventory, *self.post_inventory)
        ):
            raise ValueError("cleanup inventory crossed its Factory boundary")
        if any(receipt.factory_id != self.factory_id for receipt in self.retirement_receipts):
            raise ValueError("retirement receipt crossed its Factory boundary")
        return self


class ProjectComputerReplaceRequest(_StrictContract):
    factory_id: str = Field(min_length=1, max_length=255)
    cloud_deployment_id: str = Field(min_length=1, max_length=255)
    source_repository_id: str = Field(min_length=1, max_length=255)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    snapshot_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    workspace_snapshot: ProjectComputerWorkspaceSnapshot
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_snapshot_binding(self) -> ProjectComputerReplaceRequest:
        if (
            self.snapshot_digest != self.workspace_snapshot.manifest.manifest_digest
            or self.source_revision != self.workspace_snapshot.commit_sha
        ):
            raise ValueError("replacement source does not match its snapshot")
        return self


class DataBindingCreateRequest(_StrictContract):
    factory_id: str
    name: str = Field(min_length=1, max_length=255)
    dataset_id: str = Field(min_length=1, max_length=255)
    data_contract_version: str = Field(min_length=1, max_length=255)
    binding_kind: str = Field(min_length=1, max_length=100)
    authority_ref: str = Field(min_length=1, max_length=1000)
    access_policy: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataBindingResponse(_StrictContract):
    data_binding_id: UUID
    org_id: str
    research_intern_id: str
    factory_id: str
    project_id: str
    name: str
    dataset_id: str
    data_contract_version: str
    generation: int = Field(ge=1)
    binding_kind: str
    authority_ref: str
    access_policy: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime


__all__ = [
    "DataBindingCreateRequest",
    "DataBindingResponse",
    "DatasetRevisionCreateRequest",
    "DatasetRevisionLifecycleRequest",
    "DatasetRevisionResponse",
    "MAGI_CANONICAL_USER_BY_MODE",
    "MagiActuationStatus",
    "MagiCanonicalUser",
    "MagiDecisionActuationReceipt",
    "MagiDecisionKind",
    "MagiDecisionReceiptResponse",
    "MagiDecisionRequest",
    "MagiMode",
    "InternAsyncBlocker",
    "InternAsyncCheckpoint",
    "InternAsyncCommandKind",
    "InternAsyncCommandReceipt",
    "InternAsyncCommandRequest",
    "InternAsyncEnsureRequest",
    "InternAsyncEvent",
    "InternAsyncEventPage",
    "InternAsyncEventStreamEnvelope",
    "InternAsyncEvidenceReadiness",
    "InternAsyncExternalExecutionStatus",
    "InternAsyncInstructionKind",
    "InternAsyncInstructionRequest",
    "InternAsyncRuntime",
    "InternAsyncRuntimeBudget",
    "InternAsyncStatus",
    "InternRuntimeBinding",
    "ProjectComputerCleanupReceiptResponse",
    "ProjectComputerCleanupRequest",
    "ProjectComputerLifecycle",
    "ProjectComputerProvisionRequest",
    "ProjectComputerProvisionReceipt",
    "ProjectComputerReplaceRequest",
    "ProjectComputerResponse",
    "ProjectComputerRestorationReceipt",
    "ProjectComputerRetirementReceipt",
    "ProjectComputerWorkspaceSnapshot",
    "ResearchInternAcceptanceReceiptPublicationRequest",
    "ResearchInternAcceptanceReceiptPublicationResponse",
    "ResearchInternEventActorKind",
    "ResearchInternEventAppendRequest",
    "ResearchInternEventKind",
    "ResearchInternEventResponse",
    "ResearchInternFactoryMembershipResponse",
    "ResearchInternPatchRequest",
    "ResearchInternPolicySet",
    "ResearchInternProvisionRequest",
    "ResearchInternResponse",
    "ResearchInternSessionCloseRequest",
    "ResearchInternSessionCreateRequest",
    "ResearchInternSessionResponse",
    "ResearchInternSessionSyncResponse",
    "ResearchInternSessionStatus",
    "ResearchInternStatus",
    "ResearchInternTurnControl",
    "ResearchInternTurnError",
    "ResearchInternTurnRequest",
    "ResearchInternTurnResponse",
    "ResearchInternTurnStatus",
]
