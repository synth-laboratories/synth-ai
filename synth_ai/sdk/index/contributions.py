"""Backend-owned Contribution draft wire mirror; unreleased.

See sibling backend/packages/contributions/upload.py and cross-repo schema tests.
"""

from datetime import date, datetime
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import Field, StringConstraints, field_validator, model_validator

from .artifacts import ArtifactPublicationPrepareResponse
from .contracts import ContributionReference, IndexContract
from .package import ContributionPackage

NonEmpty = Annotated[str, StringConstraints(min_length=1, max_length=2048)]


class ContributionDraft(IndexContract):
    reference: ContributionReference
    collection_id: UUID
    status: Literal["draft"] = "draft"
    visibility: Literal["private"] = "private"
    artifact_revision: Literal[1] = 1
    manifest_schema_version: Literal["synth.contribution.v1"] = "synth.contribution.v1"


class ResearchSource(IndexContract):
    """Immutable source identity accepted by the private research-draft route."""

    organization_id: NonEmpty
    project_id: NonEmpty
    arc_id: NonEmpty
    source_repository_url: NonEmpty
    source_revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    observed_date: date | None = None
    session_ids: tuple[NonEmpty, ...] = Field(min_length=1, max_length=256)
    tools: tuple[NonEmpty, ...] = Field(min_length=1, max_length=64)
    source_paths: tuple[NonEmpty, ...] = Field(min_length=1, max_length=1024)

    @field_validator("source_repository_url")
    @classmethod
    def validate_repository_url(cls, value: str) -> str:
        url = urlsplit(value)
        if (
            url.scheme not in {"https", "http"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "repository URL must be HTTP(S) without embedded credentials or query data"
            )
        return value

    @model_validator(mode="after")
    def validate_span(self):
        if (self.started_at is None) != (self.finished_at is None):
            raise ValueError("source needs both timestamps or neither")
        if self.started_at is None and self.observed_date is None:
            raise ValueError("source needs timestamps or a day-level observed date")
        if self.started_at and self.finished_at and self.finished_at < self.started_at:
            raise ValueError("source time span is reversed")
        return self


class ResearchDraftSpec(IndexContract):
    """Server allocates identity and SYNTH provenance; caller supplies source proof only."""

    bundle_digest: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
    source: ResearchSource


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
