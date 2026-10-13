"""Inspectable Swarms outcomes and independent finality. See SDK surface.md."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import AwareDatetime, Field, StrictInt, model_validator

from synth_ai.errors import OperationFailedError
from synth_ai.types.base import Digest, Identifier, Model, ResponseModel
from synth_ai.types.usage import UsageSummary

from .launch import Inputs, Limits, ResolvedModelId, ResolvedModels, ResolvedProviders

RunState = Literal["queued", "running", "completed", "partial", "failed", "cancelled"]


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


class RunFailure(Model):
    code: Identifier
    retryable: bool


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
            raise OperationFailedError(
                "Swarm did not complete successfully",
                code=self.failure.code if self.failure else "swarm_cancelled",
                request_id=self.metadata.request_id,
            )


class TaskOutcome(ResponseModel):
    run_id: Identifier
    task_id: Identifier
    state: RunState
    attempts: StrictInt = Field(ge=0, le=100)
    failure: RunFailure | None


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
