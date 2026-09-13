"""Contribution read/review/publication wire mirrors; backend routes are authority.

See sibling docs/drafts/synth-index-api-design-2026-09-12.md and the shared lane
contract file. Review, publication and withdrawal are distinct authorities; the
SDK exposes them but the backend decides whether the caller holds them.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StrictInt, StringConstraints

from .contracts import (
    ContributionAudience,
    ContributionReference,
    Identifier,
    IndexContract,
    ShortText,
)
from .submission import RevisionStatus

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ReviewComments = Annotated[str, StringConstraints(min_length=1, max_length=20_000)]


class ReviewDecision(StrEnum):
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    REJECT = "reject"


class Assessment(IndexContract):
    assessment_id: Identifier
    reference: ContributionReference
    manifest_digest: Digest
    decision: ReviewDecision
    comments: ReviewComments
    created_at: AwareDatetime


class ContributionRevision(IndexContract):
    reference: ContributionReference
    status: RevisionStatus
    manifest_digest: Digest | None = None
    parent_revision_id: Identifier | None = None
    assessments: tuple[Assessment, ...] = Field(default=(), max_length=64)


class Contribution(IndexContract):
    contribution_id: Identifier
    current_revision_id: Identifier | None = None
    generation: Annotated[StrictInt, Field(ge=0)]
    status: Literal["draft", "published", "withdrawn"]
    visibility: ContributionAudience


class Publication(IndexContract):
    contribution_id: Identifier
    current_revision_id: Identifier | None = None
    generation: Annotated[StrictInt, Field(ge=0)]
    status: Literal["published", "withdrawn"]


class RevisionCreateSpec(IndexContract):
    parent_revision_id: Identifier


class AssessmentCreateSpec(IndexContract):
    """Reviewer decision bound to the exact sealed manifest being judged."""

    manifest_digest: Digest
    decision: ReviewDecision
    comments: ReviewComments


class PublishSpec(IndexContract):
    revision_id: Identifier
    expected_generation: Annotated[StrictInt, Field(ge=0)]


class WithdrawSpec(IndexContract):
    expected_generation: Annotated[StrictInt, Field(ge=0)]
    reason: ShortText


class AssessmentList(IndexContract):
    items: tuple[Assessment, ...] = Field(max_length=256)


class ContributionRevisionList(IndexContract):
    items: tuple[ContributionRevision, ...] = Field(max_length=1000)
