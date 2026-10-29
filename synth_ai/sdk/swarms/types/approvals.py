"""Customer approval surface for approval-gated capabilities (SYN-4547).

Orchestra owns every approval and journals each verdict; the backend only
authorizes the caller and relays one decision with the caller's expected
revision. A stale revision is a typed ``approval_revision_stale`` conflict.
``deadline_at``/``on_timeout`` are the run's pinned interaction policy for the
wait; they are ``None`` when the parked call's owner records no deadline.
"""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StrictInt, StringConstraints

from synth_ai.types.base import Model, ResponseIdentifier, ResponseModel

ApprovalState = Literal[
    "awaiting",
    "approved",
    "refused",
    "permit_delivered",
    "permit_refused",
    "permit_in_doubt",
]
ApprovalDecision = Literal["approve", "refuse"]
ApprovalOnTimeout = Literal["decline", "cancel", "escalate"]
ApprovalVerdict = Literal["enacted", "refused"]
ApprovalReason = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]
ApprovalRevision = Annotated[StrictInt, Field(ge=1, le=2**63 - 1)]


class Approval(ResponseModel):
    """One parked approval-gated call as Orchestra retains it."""

    approval_id: ResponseIdentifier
    run_id: ResponseIdentifier
    revision: ApprovalRevision
    state: ApprovalState
    kind: Literal["tool_call"]
    question: Annotated[str, Field(min_length=1, max_length=1024)]
    tool: ResponseIdentifier
    capability: ResponseIdentifier
    argument_digest: ResponseIdentifier
    actor_id: ResponseIdentifier
    attempt_id: ResponseIdentifier
    parked_at: AwareDatetime
    deadline_at: AwareDatetime | None
    on_timeout: ApprovalOnTimeout | None
    decidable: bool


class ApprovalList(ResponseModel):
    """Every approval Orchestra holds open for the run (bounded by the owner)."""

    run_id: ResponseIdentifier
    items: tuple[Approval, ...] = Field(max_length=1000)


class ApprovalDecisionRequest(Model):
    decision: ApprovalDecision
    expected_revision: ApprovalRevision
    reason: ApprovalReason | None = None


class ApprovalDecisionReceipt(ResponseModel):
    """Orchestra's journaled verdict for one decided revision."""

    run_id: ResponseIdentifier
    approval_id: ResponseIdentifier
    decision: ApprovalDecision
    decided_revision: ApprovalRevision
    verdict: ApprovalVerdict


__all__ = [
    "Approval",
    "ApprovalDecision",
    "ApprovalDecisionReceipt",
    "ApprovalDecisionRequest",
    "ApprovalList",
    "ApprovalOnTimeout",
    "ApprovalState",
    "ApprovalVerdict",
]
