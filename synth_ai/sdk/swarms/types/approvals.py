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

SYN-4650 W4: the list also carries the run's open Orchestra approval gates
(``kind="gate"``, :class:`GateApproval`): a single-shot ``revision`` of 1,
``gate_kind`` (``milestone_acceptance`` or ``release``) and the gated
``target_task_id``; no tool-call or wait fields. A gate is decided with
``swarms.approvals.decide`` (``approve``/``refuse`` at ``expected_revision=1``);
Orchestra journals exactly one answer. ``ApprovalList.items`` is discriminated
on ``kind``. Wire bytes: ``tests/fixtures/swarms-v1-approval-gate/`` (frozen by
the backend, ``synth.swarms-approval-gate.v1``).
"""

from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, StrictInt, StringConstraints, model_validator

from synth_ai.types.base import Model, ResponseBase, ResponseIdentifier, ResponseModel

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
ApprovalKind = Literal["tool_call", "wait", "gate"]
# The kinds an :class:`Approval` (tool call or wait) item carries.
CallOrWaitKind = Literal["tool_call", "wait"]
# Orchestra ``PendingApprovalView.kind`` (``CreateApprovalGate``).
GateKind = Literal["milestone_acceptance", "release"]
GATE_SCHEMA_VERSION = "synth.swarms-approval-gate.v1"
# A gate is single-shot: Orchestra answers it once and journals no revision.
GATE_REVISION = 1
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
    kind: CallOrWaitKind
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


class GateApproval(ResponseBase):
    """One open Orchestra approval gate as a ``kind="gate"`` approval item."""

    approval_id: ResponseIdentifier
    run_id: ResponseIdentifier
    revision: ApprovalRevision
    state: Literal["awaiting"]
    kind: Literal["gate"]
    question: Annotated[str, Field(min_length=1, max_length=4096)]
    gate_kind: GateKind
    target_task_id: ResponseIdentifier | None
    parked_at: AwareDatetime
    decidable: Literal[True]

    @model_validator(mode="after")
    def validate_single_shot(self) -> Self:
        if self.revision != GATE_REVISION:
            raise ValueError("approval_gate_revision_invalid")
        return self


ApprovalItem = Annotated[Approval | GateApproval, Field(discriminator="kind")]


class ApprovalList(ResponseModel):
    """Every approval Orchestra holds open for the run (bounded by the owner)."""

    run_id: ResponseIdentifier
    items: tuple[ApprovalItem, ...] = Field(max_length=1000)


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
    "GATE_REVISION",
    "GATE_SCHEMA_VERSION",
    "Approval",
    "ApprovalDecision",
    "ApprovalDecisionReceipt",
    "ApprovalDecisionRequest",
    "ApprovalItem",
    "ApprovalKind",
    "ApprovalList",
    "ApprovalOnTimeout",
    "ApprovalState",
    "ApprovalVerdict",
    "CallOrWaitKind",
    "GateApproval",
    "GateKind",
    "WaitAnswerReceipt",
    "WaitAnswerRequest",
    "WaitAnswerScope",
    "WaitKind",
    "WaitPosture",
]
