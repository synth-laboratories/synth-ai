"""Revision-bound QA conversation contracts.

See notes/specifications/synth-index/contribution-qa-cases.md. These states are
QA coordination, never substitutes for Contribution review or publication.
"""

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, StrictBool, StrictInt, StringConstraints

from .contracts import (
    ContributionAudience,
    ContributionKind,
    ContributionReference,
    Identifier,
    IndexContract,
)
from .lifecycle import Digest

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]


class CaseState(StrEnum):
    SUBMITTED = "submitted"
    AUTOMATIC_QA = "automatic_qa"
    PAUSED_INTERNAL = "paused_internal"
    REVIEWER_QUEUE = "reviewer_queue"
    WAITING_FOR_CONTRIBUTOR = "waiting_for_contributor"
    ESCALATED = "escalated"
    ACCEPTED_PRIVATE = "accepted_private"
    REJECTED = "rejected"


class CaseAction(StrEnum):
    MESSAGE = "message"
    START_CHECKS = "start_checks"
    CHECKS_COMPLETE = "checks_complete"
    PAUSE_INTERNAL = "pause_internal"
    RETRY_CHECKS = "retry_checks"
    REQUEST_CHANGES = "request_changes"
    RESPOND = "respond"
    ESCALATE = "escalate"
    ADJUDICATE = "adjudicate"
    APPROVE = "approve"
    REJECT = "reject"
    APPEAL = "appeal"


class CaseRole(StrEnum):
    CONTRIBUTOR = "contributor"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"


class EventVisibility(StrEnum):
    """Shared events reach the contributor; internal notes never do."""

    SHARED = "shared"
    INTERNAL = "internal"


# The generic events route refuses these; use the dedicated fenced routes.
FENCED_ACTIONS = frozenset({CaseAction.APPEAL, CaseAction.ESCALATE, CaseAction.ADJUDICATE})


class CreateCaseSpec(IndexContract):
    reference: ContributionReference
    manifest_digest: Digest
    rubric_version: Identifier


class AssignmentSpec(IndexContract):
    reviewer_user_id: UUID
    reviewer_org_id: UUID
    expires_at: AwareDatetime


class AcceptAssignmentSpec(IndexContract):
    conflict_free: StrictBool
    provenance: Annotated[str, StringConstraints(pattern=r"^(human|agent_assisted|agent)$")]


class CaseEventSpec(IndexContract):
    expected_version: Annotated[StrictInt, Field(ge=0)]
    action: CaseAction
    message: Text


class FencedCaseRequest(IndexContract):
    """Names the exact case version, sealed manifest and rubric it was written against."""

    expected_version: Annotated[StrictInt, Field(ge=0)]
    manifest_digest: Digest
    rubric_version: Identifier
    message: Text


class AppealSpec(FencedCaseRequest):
    """Contributor appeal of a rejection or private acceptance."""


class EscalationSpec(FencedCaseRequest):
    """Contributor or reviewer escalation to a coordinator."""


class AdjudicationSpec(FencedCaseRequest):
    """Independent coordinator decision; the only outcome is fresh independent review."""

    outcome: Literal["reopen_independent_review"] = "reopen_independent_review"


class InternalNoteSpec(FencedCaseRequest):
    """Reviewer/coordinator-only note, never delivered to the contributor."""


class CaseView(IndexContract):
    case_id: UUID
    requested_audience: ContributionAudience
    contribution_kind: ContributionKind
    reference: ContributionReference
    manifest_digest: Digest
    rubric_version: Identifier
    state: CaseState
    version: Annotated[StrictInt, Field(ge=0)]
    created_at: AwareDatetime
    updated_at: AwareDatetime
    # Coordinating approval is insufficient for release. Authoritative lifecycle
    # assessments and rights clearance are separate, exact-manifest gates.
    publication_authorized: Literal[False] = False


class AssignmentView(IndexContract):
    assignment_id: UUID
    case_id: UUID
    reviewer_user_id: UUID
    reviewer_org_id: UUID
    expires_at: AwareDatetime
    accepted_at: AwareDatetime | None = None
    revoked_at: AwareDatetime | None = None
    provenance: str | None = None


class CaseEventView(IndexContract):
    event_id: UUID
    sequence: Annotated[StrictInt, Field(ge=1)]
    action: CaseAction
    message: Text
    actor_user_id: UUID
    role: CaseRole
    created_at: AwareDatetime
    visibility: EventVisibility = EventVisibility.SHARED


class CaseEvents(IndexContract):
    items: tuple[CaseEventView, ...] = Field(max_length=100)
    next_after: Annotated[StrictInt, Field(ge=0)] | None = None
