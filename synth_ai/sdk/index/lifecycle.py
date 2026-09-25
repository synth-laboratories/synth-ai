"""Review, publication and read wire mirrors of backend ``packages/contributions/views.py``.

Identity rule: ``principal_id``, ``reviewer_id`` and package contributor
``principal_id`` are the same app-user UUID string. Review, publication and
withdrawal are distinct authorities; the backend decides whether the caller has
them, and owners/credited contributors can never review or publish their own work.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, AwareDatetime, Field, StrictInt, StringConstraints

from .contracts import (
    ContributionAudience,
    ContributionKind,
    ContributionOrigin,
    ContributionReference,
    Identifier,
    IndexContract,
    ShortText,
)
from .package import ContributionPackage
from .submission import RevisionStatus

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Title = Annotated[str, StringConstraints(min_length=1, max_length=200)]
ReviewComments = Annotated[str, StringConstraints(min_length=1, max_length=16_384)]
ClaimCommentText = Annotated[str, StringConstraints(min_length=1, max_length=4096)]
Capability = Literal["award", "contest", "publish", "review", "research_import"]
Generation = Annotated[StrictInt, Field(ge=0)]
IndexingState = Literal["not_indexed", "pending", "indexed", "failed"]


class ReviewDecision(StrEnum):
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    REJECT = "reject"


class ReviewClaimComment(IndexContract):
    claim_id: Identifier
    comment: ClaimCommentText


class ReviewSpec(IndexContract):
    """Reviewer decision; ``manifest_digest`` optionally pins the sealed bytes judged."""

    decision: ReviewDecision
    comments: ReviewComments
    claim_comments: tuple[ReviewClaimComment, ...] = Field(default=(), max_length=128)
    manifest_digest: Digest | None = None


class Assessment(IndexContract):
    """Comments are None/empty for readers without review or owner access."""

    assessment_id: Identifier
    reference: ContributionReference
    revision_id: Identifier
    manifest_digest: Digest
    reviewer_id: Identifier
    decision: ReviewDecision
    comments: ReviewComments | None = None
    claim_comments: tuple[ReviewClaimComment, ...] = Field(default=(), max_length=128)
    created_at: AwareDatetime | None = None
    status: RevisionStatus | None = None


class Assessments(IndexContract):
    items: tuple[Assessment, ...] = Field(max_length=64)


class RevisionCreateSpec(IndexContract):
    parent_revision_id: Identifier


class PublicationSpec(IndexContract):
    """Audience must equal the sealed requested audience; publisher grant required."""

    revision_id: Identifier
    audience: ContributionAudience
    expected_generation: Generation | None = None
    assessment_id: Identifier | None = None


class WithdrawalSpec(IndexContract):
    reason: ShortText
    expected_generation: Generation | None = None


class PublicationStatus(IndexContract):
    contribution_id: Identifier
    generation: Generation
    current_revision_id: Identifier | None = None
    status: Literal["published", "withdrawn", "unpublished"]


class ReviewListItem(IndexContract):
    reference: ContributionReference
    manifest_digest: Digest | None = None
    title: Title | None = None
    kind: ContributionKind | None = None
    status: RevisionStatus
    requested_audience: ContributionAudience | None = None
    contributor_ids: tuple[Identifier, ...] = Field(default=(), max_length=32)
    submitted_at: AwareDatetime | None = None


class ReviewList(IndexContract):
    items: tuple[ReviewListItem, ...] = Field(max_length=50)
    next_cursor: Identifier | None = None


class Citation(IndexContract):
    contribution_id: Identifier
    revision_id: Identifier
    manifest_digest: Digest
    url: AnyHttpUrl


class RevisionSummary(IndexContract):
    revision_id: Identifier
    reference: ContributionReference
    parent_revision_id: Identifier | None = None
    status: RevisionStatus
    is_current: bool = False
    title: Title | None = None
    created_at: AwareDatetime | None = None
    submitted_at: AwareDatetime | None = None
    published_at: AwareDatetime | None = None


class RevisionView(IndexContract):
    """Superseded = status published and not ``is_current``."""

    reference: ContributionReference
    status: RevisionStatus
    is_current: bool = False
    current_revision_id: Identifier | None = None
    parent_revision_id: Identifier | None = None
    manifest_digest: Digest | None = None
    publication_id: Identifier | None = None
    created_at: AwareDatetime | None = None
    submitted_at: AwareDatetime | None = None
    published_at: AwareDatetime | None = None
    indexing_state: IndexingState | None = None
    package: ContributionPackage | None = None
    assessments: tuple[Assessment, ...] = Field(default=(), max_length=64)
    citation: Citation | None = None


class ContributionView(IndexContract):
    contribution_id: Identifier
    title: Title | None = None
    owner_id: Identifier
    origin: ContributionOrigin
    audience: ContributionAudience | None = None
    status: Literal["draft", "published", "withdrawn"]
    generation: Generation
    current_revision_id: Identifier | None = None
    withdrawn: bool = False
    withdrawal_reason: ShortText | None = None
    contest_ids: tuple[Identifier, ...] = Field(default=(), max_length=16)
    indexing_state: IndexingState | None = None
    current: RevisionView | None = None
    revisions: tuple[RevisionSummary, ...] = Field(default=(), max_length=256)


class MyContribution(IndexContract):
    reference: ContributionReference
    contribution_id: Identifier
    title: Title | None = None
    status: RevisionStatus
    audience: ContributionAudience | None = None
    current_revision_id: Identifier | None = None
    generation: Generation
    updated_at: AwareDatetime | None = None


class MyContributions(IndexContract):
    items: tuple[MyContribution, ...] = Field(max_length=200)


class MeView(IndexContract):
    principal_id: Identifier
    org_id: Identifier
    display_name: str | None = None
    reviewer: bool
    capabilities: tuple[Capability, ...] = ()


__all__ = [
    "Assessment",
    "Assessments",
    "Capability",
    "Citation",
    "ContributionView",
    "MeView",
    "MyContribution",
    "MyContributions",
    "PublicationSpec",
    "PublicationStatus",
    "ReviewClaimComment",
    "ReviewDecision",
    "ReviewList",
    "ReviewListItem",
    "ReviewSpec",
    "RevisionCreateSpec",
    "RevisionSummary",
    "RevisionView",
    "WithdrawalSpec",
]
