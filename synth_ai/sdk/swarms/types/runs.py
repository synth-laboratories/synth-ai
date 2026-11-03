"""Inspectable Swarms outcomes and independent finality. See SDK surface.md."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import (
    AwareDatetime,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    StrictInt,
    model_serializer,
    model_validator,
)

from synth_ai.errors import OperationFailedError
from synth_ai.sdk.core.artifact_types import ResponseLabels
from synth_ai.types.base import (
    Digest,
    EventCursor,
    Page,
    ResponseBase,
    ResponseIdentifier,
    ResponseModel,
)
from synth_ai.types.rate_limits import RateLimitSnapshot
from synth_ai.types.receipts import ControlCommand, ControlState
from synth_ai.types.refs import Ref
from synth_ai.types.unknown import UnknownValue
from synth_ai.types.usage import UsageSummary

from .launch import (
    Inputs,
    Limits,
    RepoSelector,
    ResolvedModelId,
    ResolvedModels,
    ResolvedProviders,
)
from .methods import ResolvedResearchMethod
from .objectives import GuidanceCoverage, ObjectiveResult
from .overrides import ProfileOverrideReadback
from .problems import Problem, ReasonLink
from .status_facets import RunStatusFacets

KnownRunState = Literal[
    "queued",
    "running",
    "pausing",
    "paused",
    "cancelling",
    "completed",
    "partial",
    "failed",
    "cancelled",
]
RunState = KnownRunState | UnknownValue

KnownTaskState = Literal[
    "pending",
    "queued",
    "running",
    "blocked",
    "review_required",
    "repair_required",
    "completed",
    "partial",
    "failed",
    "cancelled",
]

TaskState = KnownTaskState | UnknownValue


class AdmittedResourceGrant(ResponseBase):
    """Secret-free summary of one frozen resource grant."""

    grant_id: str
    plugin: str
    version: StrictInt
    roles: tuple[str, ...]
    limits: dict[str, StrictInt]
    role_limits: dict[str, dict[str, StrictInt]]
    models: tuple[str, ...]
    collections: tuple[str, ...]
    selection: dict[str, Any] | None
    funding_lane: str


class ResolvedConfiguration(ResponseModel):
    method: ResolvedResearchMethod | None = None
    run_id: ResponseIdentifier | None
    project_id: ResponseIdentifier
    configuration_id: ResponseIdentifier
    digest_sha256: Digest
    profile_id: ResponseIdentifier
    runtime_image_digest: Digest
    limits: Limits
    inputs: Inputs
    providers: ResolvedProviders
    models: ResolvedModels
    repo: RepoSelector | None = None
    resource_grants: tuple[AdmittedResourceGrant, ...] | None = None
    # SYN-4848: echoed only when the launch selects profile overrides.
    profile_overrides: ProfileOverrideReadback | None = None

    @model_serializer(mode="wrap")
    def omit_absent_overrides(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        value = handler(self)
        if self.profile_overrides is None:
            value.pop("profile_overrides", None)
        return value


class LaunchPreflight(ResponseModel):
    ready: bool
    blockers: tuple[str, ...] = Field(max_length=32)
    configuration: ResolvedConfiguration | None
    expires_at: AwareDatetime | None
    reservation_created: Literal[False] = False
    # Response-header capacity (SYN-4680); not part of the JSON body or schema.
    _rate_limits: tuple[RateLimitSnapshot, ...] = PrivateAttr(default=())

    @property
    def rate_limits(self) -> tuple[RateLimitSnapshot, ...]:
        """``x-ratelimit-*`` capacity the server reported on this preflight response."""
        return self._rate_limits

    def with_rate_limits(self, rate_limits: tuple[RateLimitSnapshot, ...]) -> Self:
        self._rate_limits = rate_limits
        return self

    @model_validator(mode="after")
    def validate_preview(self) -> Self:
        if self.ready and (self.blockers or self.configuration is None or self.expires_at is None):
            raise ValueError("ready preview requires an expiring configuration and no blockers")
        if self.configuration is not None and self.configuration.run_id is not None:
            raise ValueError("preflight must not create a run")
        return self


class RunFailure(ResponseBase):
    code: ResponseIdentifier
    retryable: bool
    # Root-cause explanation (synth.problems.v1). Defaults keep a response from
    # an older server valid; a new server always fills them.
    problem_id: ResponseIdentifier | None = None
    reason_chain: tuple[ReasonLink, ...] = Field(default=(), max_length=16)


class RunControlSummary(ResponseBase):
    operation_id: ResponseIdentifier
    command: ControlCommand
    state: ControlState


class RunSnapshot(ResponseModel):
    method: ResolvedResearchMethod | None = None
    run_id: ResponseIdentifier
    project_id: ResponseIdentifier
    ref: Ref | None = None
    labels: ResponseLabels = Field(default_factory=dict)
    labels_revision: StrictInt = Field(default=1, ge=1)
    objectives: tuple[ObjectiveResult, ...] = Field(default=(), max_length=64)
    guidance_coverage: tuple[GuidanceCoverage, ...] = Field(default=(), max_length=32)
    state: RunState
    model: ResolvedModelId
    provider: ResponseIdentifier
    result_available: bool
    execution_closed: bool
    accounting_final: bool
    cleanup_complete: bool
    control: RunControlSummary | None = None
    paused_at: AwareDatetime | None = None
    resume_by: AwareDatetime | None = None
    deadline_at: AwareDatetime | None = None
    pause_count: StrictInt | None = Field(default=None, ge=0, le=2**32 - 1)
    # Ranked open problems, root cause first (synth.problems.v1).
    problems: tuple[Problem, ...] = Field(default=(), max_length=32)
    # False when Orchestra's problems could not be read (see the
    # `owner_unavailable` problem) or were not requested (list reads).
    problems_observed: bool = False
    # Set for a failed run, and for a run whose cancel was accepted but is not
    # terminal: the root cause, not a generic code.
    failure: RunFailure | None = None
    # SYN-4848: the frozen model/effort overrides and, once prepared, their resolution.
    profile_overrides: ProfileOverrideReadback | None = None
    # SYN-4849: execution, publication, settlement and cleanup read independently.
    # Absent from servers that predate it; ``state`` is unchanged.
    status_facets: RunStatusFacets | None = None

    @model_serializer(mode="wrap")
    def omit_absent_controls(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        value = handler(self)
        for name in (
            "ref",
            "control",
            "paused_at",
            "resume_by",
            "deadline_at",
            "pause_count",
            "profile_overrides",
            "status_facets",
        ):
            if getattr(self, name) is None:
                value.pop(name, None)
        return value

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if self.ref is not None and (
            self.ref.authority != "swarms"
            or self.ref.kind != "run"
            or self.ref.record_id != self.run_id
            or self.ref.revision != "1"
        ):
            raise ValueError("Run reference must identify this exact frozen launch request")
        return self


class SwarmResult(ResponseModel):
    objectives: tuple[ObjectiveResult, ...] = Field(default=(), max_length=64)
    guidance_coverage: tuple[GuidanceCoverage, ...] = Field(default=(), max_length=32)
    run_id: ResponseIdentifier
    status: Literal["completed", "partial", "failed", "cancelled"]
    text: str | None = Field(default=None, max_length=65536)
    output_ids: tuple[ResponseIdentifier, ...] = Field(max_length=256)
    output_count: StrictInt = Field(ge=0)
    output_ready: bool
    warnings: tuple[str, ...] = Field(max_length=32)
    failure: RunFailure | None
    usage: UsageSummary
    accounting_final: bool
    cleanup_complete: bool

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if (self.status == "failed") != (self.failure is not None):
            raise ValueError("failed outcome requires exactly one failure description")
        if self.output_count < len(self.output_ids) or len(self.output_ids) != len(
            set(self.output_ids)
        ):
            raise ValueError("output identities/count are inconsistent")
        if self.accounting_final != (self.usage.financial_state == "final"):
            raise ValueError("result and usage finality disagree")
        return self

    def raise_for_failure(self) -> None:
        """Opt into raising without losing the canonical result."""
        if self.status in {"failed", "cancelled"}:
            root = (
                self.failure.reason_chain[0] if self.failure and self.failure.reason_chain else None
            )
            raise OperationFailedError(
                "Swarm did not complete successfully"
                + (f": root cause {root.service}/{root.code} at {root.stage}" if root else ""),
                code=self.failure.code if self.failure else "swarm_cancelled",
                request_id=self.metadata.request_id,
            )


TaskActivity = Literal[
    "run_held",
    "operator_held",
    "assignment_held",
    "output_publication_pending",
    "reviewer_assessment",
    "review_approved",
    "orchestrator_repair",
    "repair_requested",
    "repair_dispatch_due",
    "allocation_without_session",
    "session_without_completion",
    "stop_requested",
]


class TaskExecutionFacts(ResponseBase):
    """Independent native execution facts; activity does not assert finality."""

    activity: tuple[TaskActivity, ...] = Field(max_length=12)
    physical_completion_count: StrictInt = Field(ge=0, le=2**32 - 1)
    finalizer: (
        Literal[
            "positive", "negative", "limit", "no_launch", "dispatch_abort", "limit_dispatch_abort"
        ]
        | None
    )
    outcome: Literal["succeeded", "failed", "cancelled"] | None
    closing_recorded: bool

    @model_validator(mode="after")
    def validate_facts(self) -> Self:
        if len(self.activity) != len(set(self.activity)):
            raise ValueError("task activity must be unique")
        if (self.finalizer is None) != (self.outcome is None):
            raise ValueError("task finalizer and outcome must be present together")
        if self.closing_recorded and self.finalizer is None:
            raise ValueError("closing requires reviewed finalization")
        if self.outcome == "succeeded" and self.finalizer != "positive":
            raise ValueError("success requires positive finalization")
        return self


class TaskOutcome(ResponseModel):
    run_id: ResponseIdentifier
    task_id: ResponseIdentifier
    state: TaskState
    attempts: StrictInt = Field(ge=0, le=2**32 - 1)
    failure: RunFailure | None
    # Null preserves historical responses without fabricating native facts.
    execution: TaskExecutionFacts | None = None

    @model_validator(mode="after")
    def validate_execution(self) -> Self:
        if self.execution is not None:
            if self.execution.physical_completion_count > self.attempts:
                raise ValueError("physical completions exceed allocated attempts")
            expected = {
                None: "pending",
                "succeeded": "completed",
                "failed": "failed",
                "cancelled": "cancelled",
            }[self.execution.outcome]
            if self.state != expected:
                raise ValueError("native state must follow reviewed finalization")
        return self


INTERACTION_EVENT_KINDS = frozenset(
    ("approval.parked", "approval.decided", "wait.opened", "wait.resolved")
)


class SwarmEvent(ResponseBase):
    event_id: ResponseIdentifier
    run_id: ResponseIdentifier
    sequence: StrictInt = Field(ge=1)
    kind: (
        Literal[
            "created",
            "objective.created",
            "objective.revised",
            "objective.not_applicable",
            "objective.claimed",
            "objective.answered",
            "objective.verified",
            "objective.rejected",
            "objective.partial",
            "objective.revision_approved",
            "running",
            "completed",
            "partial",
            "failed",
            "cancelled",
            "paused",
            "resumed",
            "cancel_requested",
            "usage_updated",
            "approval.parked",
            "approval.decided",
            "wait.opened",
            "wait.resolved",
            "unknown",
        ]
        | UnknownValue
    )
    original_kind: str | None = Field(default=None, max_length=128)
    usage: UsageSummary | None = None
    observed_at: AwareDatetime | None = None
    # SYN-4547/4650: the approval or wait (interaction) the event is about.
    approval_id: ResponseIdentifier | None = None

    @model_serializer(mode="wrap")
    def omit_event_time(self, handler):
        value = handler(self)
        if self.observed_at is None:
            value.pop("observed_at", None)
        if self.approval_id is None:
            value.pop("approval_id", None)
        return value

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if (self.kind == "unknown") != (self.original_kind is not None):
            raise ValueError("unknown event requires its original discriminant")
        if (self.kind == "usage_updated") != (self.usage is not None):
            raise ValueError("only usage_updated carries usage")
        if (self.kind in INTERACTION_EVENT_KINDS) != (self.approval_id is not None):
            raise ValueError("only approval and wait events carry approval_id")
        return self


class SwarmEventPage(Page[SwarmEvent, EventCursor]):
    """A bounded page with an optional reusable observation position."""

    resume_cursor: EventCursor | None = None
