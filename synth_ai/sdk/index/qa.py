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
    """Current QA coordination state; acceptance does not authorize publication."""
    SUBMITTED = "submitted"
    AUTOMATIC_QA = "automatic_qa"
    PAUSED_INTERNAL = "paused_internal"
    REVIEWER_QUEUE = "reviewer_queue"
    WAITING_FOR_CONTRIBUTOR = "waiting_for_contributor"
    ESCALATED = "escalated"
    ACCEPTED_PRIVATE = "accepted_private"
    REJECTED = "rejected"


class CaseAction(StrEnum):
    """Action recorded in a QA conversation; fenced actions use dedicated endpoints."""
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
    """Actor role admitted for the current QA case."""
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
    """Open QA against an exact Contribution revision, sealed manifest and rubric."""
    #: Exact Contribution and revision under review.
    reference: ContributionReference
    #: Digest of the sealed revision manifest; must match the reviewed bytes.
    manifest_digest: Digest
    #: Pinned rubric identity used for these judgments.
    rubric_version: Identifier


class AssignmentSpec(IndexContract):
    """Assign an independent reviewer with an explicit organization and expiration."""
    #: User identifier of the assigned reviewer.
    reviewer_user_id: UUID
    #: Organization identifier under which the reviewer acts.
    reviewer_org_id: UUID
    #: Timezone-aware assignment expiration; expired assignments cannot authorize review.
    expires_at: AwareDatetime


class AcceptAssignmentSpec(IndexContract):
    """Declare assignment consent, absence of conflicts and review provenance."""
    #: Reviewer declaration that the assignment has no disqualifying conflict.
    conflict_free: StrictBool
    #: Declared source of the work; not proof of a verified human identity.
    provenance: Annotated[str, StringConstraints(pattern=r"^(human|agent_assisted|agent)$")]


class CaseEventSpec(IndexContract):
    """Append a shared QA action against the expected conversation version."""
    #: Current case version expected by this write; stale writes must be reconciled.
    expected_version: Annotated[StrictInt, Field(ge=0)]
    #: QA conversation action; appeals, escalation and adjudication use fenced operations.
    action: CaseAction
    #: Bounded message for this action, disclosed according to event visibility.
    message: Text


class FencedCaseRequest(IndexContract):
    """Names the exact case version, sealed manifest and rubric it was written against."""

    #: Current case version expected by this write; stale writes must be reconciled.
    expected_version: Annotated[StrictInt, Field(ge=0)]
    #: Digest of the sealed revision manifest; must match the reviewed bytes.
    manifest_digest: Digest
    #: Pinned rubric identity used for these judgments.
    rubric_version: Identifier
    #: Bounded message for this action, disclosed according to event visibility.
    message: Text


class AppealSpec(FencedCaseRequest):
    """Contributor appeal of a rejection or private acceptance."""


class EscalationSpec(FencedCaseRequest):
    """Contributor or reviewer escalation to a coordinator."""


class AdjudicationSpec(FencedCaseRequest):
    """Independent coordinator decision; the only outcome is fresh independent review."""

    #: Recorded outcome; fail, inconclusive and not-applicable remain distinct.
    outcome: Literal["reopen_independent_review"] = "reopen_independent_review"


class InternalNoteSpec(FencedCaseRequest):
    """Reviewer/coordinator-only note, never delivered to the contributor."""


class CaseView(IndexContract):
    """Current revision-bound QA case; publication authority remains separate."""
    #: Identifier of the revision-bound QA case.
    case_id: UUID
    #: Audience requested for the Contribution; does not authorize publication.
    requested_audience: ContributionAudience
    #: Contribution kind selecting the applicable content criteria.
    contribution_kind: ContributionKind
    #: Exact Contribution and revision under review.
    reference: ContributionReference
    #: Digest of the sealed revision manifest; must match the reviewed bytes.
    manifest_digest: Digest
    #: Pinned rubric identity used for these judgments.
    rubric_version: Identifier
    #: Current QA coordination state, separate from publication state.
    state: CaseState
    #: Current conversation version used to fence subsequent writes.
    version: Annotated[StrictInt, Field(ge=0)]
    #: Timezone-aware creation timestamp.
    created_at: AwareDatetime
    #: Timezone-aware timestamp of the latest recorded change.
    updated_at: AwareDatetime
    # Coordinating approval is insufficient for release. Authoritative lifecycle
    # assessments and rights clearance are separate, exact-manifest gates.
    #: Always false: this QA record cannot grant publication authority.
    publication_authorized: Literal[False] = False


class AssignmentView(IndexContract):
    """Reviewer assignment and its acceptance, expiration and revocation timestamps."""
    #: Identifier of the independent reviewer assignment.
    assignment_id: UUID
    #: Identifier of the revision-bound QA case.
    case_id: UUID
    #: User identifier of the assigned reviewer.
    reviewer_user_id: UUID
    #: Organization identifier under which the reviewer acts.
    reviewer_org_id: UUID
    #: Timezone-aware assignment expiration; expired assignments cannot authorize review.
    expires_at: AwareDatetime
    #: Timezone-aware assignment acceptance timestamp, or none before acceptance.
    accepted_at: AwareDatetime | None = None
    #: Timezone-aware revocation timestamp, or none while not revoked.
    revoked_at: AwareDatetime | None = None
    #: Declared source of the work; not proof of a verified human identity.
    provenance: str | None = None


class CaseEventView(IndexContract):
    """Recorded QA event with actor, ordering and disclosure visibility."""
    #: Identifier of the recorded QA event.
    event_id: UUID
    #: Monotonic sequence number ordering events within the case.
    sequence: Annotated[StrictInt, Field(ge=1)]
    #: QA conversation action; appeals, escalation and adjudication use fenced operations.
    action: CaseAction
    #: Bounded message for this action, disclosed according to event visibility.
    message: Text
    #: User identifier responsible for the event.
    actor_user_id: UUID
    #: Case role admitted for the event actor.
    role: CaseRole
    #: Timezone-aware creation timestamp.
    created_at: AwareDatetime
    #: Disclosure audience; internal notes must not reach the contributor.
    visibility: EventVisibility = EventVisibility.SHARED


class CaseEvents(IndexContract):
    """Bounded QA event page with a continuation cursor."""
    #: Ordered items in this bounded response page.
    items: tuple[CaseEventView, ...] = Field(max_length=100)
    #: Continuation sequence cursor, or none when this page has no continuation.
    next_after: Annotated[StrictInt, Field(ge=0)] | None = None
