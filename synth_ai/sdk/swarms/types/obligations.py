"""Customer view of the native obligations-v1 summary; never log inference."""

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from synth_ai.types.base import ResponseBase, ResponseIdentifier

ObligationKind = Literal[
    "worker_task",
    "task_review",
    "run_review",
    "publication",
    "trace_upload",
    "sandbox_teardown",
    "settlement",
    "workspace_confirmation",
]
WaiverReason = Literal[
    "policy_not_required",
    "execution_never_started",
    "no_workspace_push",
    "task_failed",
    "run_failed",
    "owner_refused",
    "superseded",
]


class ObligationWaiver(ResponseBase):
    kind: ObligationKind
    id: ResponseIdentifier
    reason: WaiverReason
    receipt_id: ResponseIdentifier
    refusal_code: ResponseIdentifier | None = None


class RunObligations(ResponseBase):
    contract: Literal["obligations_v1"] = "obligations_v1"
    total: StrictInt = Field(ge=1, le=65536)
    closed: bool
    success_ready: bool
    waiver_count: StrictInt = Field(ge=0, le=65536)
    waivers_complete: bool
    waivers: tuple[ObligationWaiver, ...] = Field(max_length=256)

    @model_validator(mode="after")
    def validate_waiver_coverage(self) -> Self:
        if (
            self.waiver_count > self.total
            or self.waiver_count < len(self.waivers)
            or self.waivers_complete != (self.waiver_count == len(self.waivers))
        ):
            raise ValueError("waiver coverage must match the native complete count")
        if len({(row.kind, row.id) for row in self.waivers}) != len(self.waivers):
            raise ValueError("waiver identities must be unique")
        return self
