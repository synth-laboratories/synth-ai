"""Inspectable Swarms outcomes and independent finality. See SDK surface.md."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import (
    AwareDatetime,
    Field,
    SerializerFunctionWrapHandler,
    StrictInt,
    model_serializer,
    model_validator,
)

from synth_ai.errors import OperationFailedError
from synth_ai.types.base import Digest, Identifier, Model, ResponseModel
from synth_ai.types.receipts import ControlCommand, ControlState
from synth_ai.types.usage import UsageSummary

from .launch import Inputs, Limits, ResolvedModelId, ResolvedModels, ResolvedProviders
from .problems import Problem, ReasonLink

RunState = Literal[
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
TaskState = Literal[
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


class ResolvedConfiguration(ResponseModel):
    run_id: Identifier | None
    project_id: Identifier
    configuration_id: Identifier
    digest_sha256: Digest
    profile_id: Identifier
    runtime_image_digest: Digest
    limits: Limits
    inputs: Inputs
    providers: ResolvedProviders
    models: ResolvedModels


class LaunchPreflight(ResponseModel):
    ready: bool
    blockers: tuple[str, ...] = Field(max_length=32)
    configuration: ResolvedConfiguration | None
    expires_at: AwareDatetime | None
    reservation_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_preview(self) -> Self:
        if self.ready and (self.blockers or self.configuration is None or self.expires_at is None):
            raise ValueError("ready preview requires an expiring configuration and no blockers")
        if self.configuration is not None and self.configuration.run_id is not None:
            raise ValueError("preflight must not create a run")
        return self


class RunFailure(Model):
    code: Identifier
    retryable: bool
    # Root-cause explanation (synth.problems.v1). Defaults keep a response from
    # an older server valid; a new server always fills them.
    problem_id: Identifier | None = None
    reason_chain: tuple[ReasonLink, ...] = Field(default=(), max_length=16)


class RunControlSummary(Model):
    operation_id: Identifier
    command: ControlCommand
    state: ControlState


class RunSnapshot(ResponseModel):
    run_id: Identifier
    project_id: Identifier
    state: RunState
    model: ResolvedModelId
    provider: Identifier
    result_available: bool
    execution_closed: bool
    accounting_final: bool
    cleanup_complete: bool
    control: RunControlSummary | None = None
    paused_at: AwareDatetime | None = None
    resume_by: AwareDatetime | None = None
    deadline_at: AwareDatetime | None = None
    # Ranked open problems, root cause first (synth.problems.v1).
    problems: tuple[Problem, ...] = Field(default=(), max_length=32)
    # False when Orchestra's problems could not be read (see the
    # `owner_unavailable` problem) or were not requested (list reads).
    problems_observed: bool = False
    # Set for a failed run, and for a run whose cancel was accepted but is not
    # terminal: the root cause, not a generic code.
    failure: RunFailure | None = None

    @model_serializer(mode="wrap")
    def omit_absent_controls(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        value = handler(self)
        for name in ("control", "paused_at", "resume_by", "deadline_at"):
            if getattr(self, name) is None:
                value.pop(name, None)
        return value


class SwarmResult(ResponseModel):
    run_id: Identifier
    status: Literal["completed", "partial", "failed", "cancelled"]
    text: str | None = Field(default=None, max_length=65536)
    output_ids: tuple[Identifier, ...] = Field(max_length=256)
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


class TaskExecutionFacts(Model):
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
    run_id: Identifier
    task_id: Identifier
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


class SwarmEvent(Model):
    event_id: Identifier
    run_id: Identifier
    sequence: StrictInt = Field(ge=1)
    kind: Literal[
        "created",
        "running",
        "completed",
        "partial",
        "failed",
        "cancelled",
        "paused",
        "resumed",
        "cancel_requested",
        "usage_updated",
        "unknown",
    ]
    original_kind: str | None = Field(default=None, max_length=128)
    usage: UsageSummary | None = None

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if (self.kind == "unknown") != (self.original_kind is not None):
            raise ValueError("unknown event requires its original discriminant")
        if (self.kind == "usage_updated") != (self.usage is not None):
            raise ValueError("only usage_updated carries usage")
        return self
