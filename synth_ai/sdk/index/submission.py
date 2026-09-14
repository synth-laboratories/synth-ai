"""Backend-owned submission wire mirror; unreleased.

See sibling backend/packages/contributions/submission.py.
"""

from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import StringConstraints

from .contracts import ContributionReference, IndexContract


class RevisionStatus(StrEnum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    CHANGES_REQUESTED = "changes_requested"
    REJECTED = "rejected"
    QUALIFIED = "qualified"
    PUBLISHED = "published"
    WITHDRAWN = "withdrawn"


class ContributionSubmitSpec(IndexContract):
    publication_id: UUID


class ContributionSubmission(IndexContract):
    reference: ContributionReference
    status: RevisionStatus
    manifest_digest: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
