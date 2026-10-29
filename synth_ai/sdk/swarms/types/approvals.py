"""Customer approval surface for approval-gated capabilities (SYN-4547).

Orchestra owns every approval and journals each verdict; the backend only
authorizes the caller and relays one decision with the caller's expected
revision. A stale revision is a typed ``approval_revision_stale`` conflict.
``deadline_at``/``on_timeout`` are the run's pinned interaction policy for the
wait; they are ``None`` when the parked call's owner records no deadline.

SYN-4650: the list also carries the run's open human-in-the-loop waits
(``kind="wait"``). A wait is answered (``swarms.approvals.answer``) with one of
its ``options`` (free-form when ``options`` is empty) at its ``revision``;
``wait_kind``, ``answer_schema``, ``answer_scope``, ``answerers`` and
``posture`` are the wait as Orchestra journaled it.
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
ApprovalKind = Literal["tool_call", "wait"]
WaitKind = Literal[
    "approval",
    "permission",
    "question",
    "form",
    "steer",
    "interrupt",
    "notice",
    "blocker",
]
WaitAnswerScope = Literal["once", "session", "policy"]
WaitPosture = Literal["pause_actor", "pause_run", "continue_other_work"]
WaitText = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]
ApprovalReason = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]
ApprovalRevision = Annotated[StrictInt, Field(ge=1, le=2**63 - 1)]


class Approval(ResponseModel):
    """One parked approval-gated call as Orchestra retains it."""

    approval_id: ResponseIdentifier
    run_id: ResponseIdentifier
    revision: ApprovalRevision
    state: ApprovalState
    kind: ApprovalKind
    question: Annotated[str, Field(min_length=1, max_length=4096)]
    tool: ResponseIdentifier | None
    capability: ResponseIdentifier | None
    argument_digest: ResponseIdentifier | None
    actor_id: ResponseIdentifier | None
    attempt_id: ResponseIdentifier | None
    parked_at: AwareDatetime
    deadline_at: AwareDatetime | None
    on_timeout: ApprovalOnTimeout | None
    decidable: bool
    wait_kind: WaitKind | None = None
    options: tuple[WaitText, ...] = Field(default=(), max_length=256)
    answer_schema: WaitText | None = None
    answer_scope: WaitAnswerScope | None = None
    answerers: tuple[WaitText, ...] = Field(default=(), max_length=256)
    posture: WaitPosture | None = None


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


class WaitAnswerRequest(Model):
    """One answer to an open wait at ``expected_revision``.

    ``answer_scope`` defaults to ``"once"``; ``answer_schema`` and ``answerer``
    default to the open wait's schema and the first of its ``answerers`` the
    caller holds. Pass the receipt's values to replay after the wait resolved.
    """

    decision: WaitText
    expected_revision: ApprovalRevision
    answer_scope: WaitAnswerScope | None = None
    answer_schema: WaitText | None = None
    answerer: WaitText | None = None


class WaitAnswerReceipt(ResponseModel):
    """Orchestra's journaled answer; a replay returns the original receipt."""

    run_id: ResponseIdentifier
    approval_id: ResponseIdentifier
    operation_id: ResponseIdentifier
    decision: WaitText
    answered_revision: ApprovalRevision
    revision: ApprovalRevision
    answerer: WaitText
    answer_scope: WaitAnswerScope
    answer_schema: WaitText
    answered_at: AwareDatetime


__all__ = [
    "Approval",
    "ApprovalDecision",
    "ApprovalDecisionReceipt",
    "ApprovalDecisionRequest",
    "ApprovalKind",
    "ApprovalList",
    "ApprovalOnTimeout",
    "ApprovalState",
    "ApprovalVerdict",
    "WaitAnswerReceipt",
    "WaitAnswerRequest",
    "WaitAnswerScope",
    "WaitKind",
    "WaitPosture",
]
