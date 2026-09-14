"""Backend-owned Contribution draft wire mirror; unreleased.

See sibling backend/packages/contributions/upload.py and cross-repo schema tests.
"""

from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from .artifacts import ArtifactPublicationPrepareResponse
from .contracts import ContributionReference, IndexContract
from .package import ContributionPackage


class ContributionDraft(IndexContract):
    reference: ContributionReference
    collection_id: UUID
    status: Literal["draft"] = "draft"
    visibility: Literal["private"] = "private"
    artifact_revision: Literal[1] = 1
    manifest_schema_version: Literal["synth.contribution.v1"] = "synth.contribution.v1"


class ContributionUploadSpec(IndexContract):
    publication_id: UUID
    package: ContributionPackage


class ContributionUploadPrepared(IndexContract):
    """Transient transfer instructions; never store signed URLs in replay receipts."""

    descriptor_json: str = Field(max_length=1_048_576)
    transfer: ArtifactPublicationPrepareResponse

    @model_validator(mode="after")
    def check_bounds(self):
        if len(self.descriptor_json.encode("utf-8")) > 1_048_576:
            raise ValueError("Descriptor exceeds 1 MiB")
        if len(self.transfer.upload_targets) > 1025:
            raise ValueError("Contribution transfer exceeds asset bound")
        return self
