"""Owner contribution intent without review/publication authority. See SDK index_contract.md."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from synth_ai.types.base import Digest, Identifier, Model, ResponseModel

from .annotations import AuthorAnnotations, ContributionAnnotations


class ContributionFilter(Model):
    query: str = Field(default="", max_length=200)
    status: Literal["published", "draft", "submitted"] | None = None


class Contribution(ResponseModel):
    contribution_id: Identifier
    revision_id: Identifier | None
    title: str | None = Field(max_length=256)
    status: Literal[
        "draft", "submitted", "changes_requested", "rejected", "qualified", "published", "withdrawn"
    ]
    contents_readable: bool
    annotations: ContributionAnnotations | None = None
    draft: ContributionDraft | None = None

    @model_validator(mode="after")
    def validate_annotations(self) -> Self:
        if self.draft is not None and (self.status != "draft" or self.draft.expected_revision_id != self.revision_id):
            raise ValueError("draft metadata requires the exact current draft revision")
        if self.annotations is not None and (
            self.annotations.contribution_id != self.contribution_id
            or self.annotations.revision_id != self.revision_id
        ):
            raise ValueError("annotations belong to another contribution revision")
        return self


class ContributionDraft(Model):
    expected_revision_id: Identifier | None
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=2048)
    annotations: AuthorAnnotations = Field(default_factory=AuthorAnnotations)


class ContributionRevision(ResponseModel):
    contribution_id: Identifier
    revision_id: Identifier
    digest_sha256: Digest | None
    parent_revision_id: Identifier | None = None


class RightsAttestation(Model):
    revision_id: Identifier
    digest_sha256: Digest
    license_expressions: tuple[str, ...] = Field(min_length=1, max_length=64)
    notices_digest_sha256: Digest
    rights_decision_reference: str = Field(min_length=1, max_length=512)
    authorized_to_contribute: Literal[True]


class AttestationReceipt(ResponseModel):
    contribution_id: Identifier
    revision_id: Identifier
    accepted: bool
    state: Literal["pending_submission", "attested"]
    reason: str | None = Field(max_length=512)


class SubmissionRequest(Model):
    revision_id: Identifier


class SubmissionReceipt(ResponseModel):
    contribution_id: Identifier
    revision_id: Identifier
    state: Literal["submitted", "refused"]
    reason: str | None = Field(max_length=512)


class ResearchCorrection(Model):
    source_revision_id: Identifier
    replacement_revision_id: Identifier
    reason: str = Field(min_length=1, max_length=2048)


class CorrectionReceipt(ResponseModel):
    contribution_id: Identifier
    revision_id: Identifier
    state: Literal["recorded", "refused"]
    reason: str | None = Field(max_length=512)
