"""Owner contribution intent without review/publication authority. See SDK index_contract.md."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from synth_ai.types.base import Digest, Identifier, Model, ResponseModel


class ContributionFilter(Model):
    query: str = Field(default="", max_length=200)
    status: Literal["published", "draft", "submitted"] | None = None


class Contribution(ResponseModel):
    contribution_id: Identifier
    revision_id: Identifier | None
    title: str | None = Field(max_length=256)
    status: Literal["draft", "submitted", "published", "withdrawn"]
    contents_readable: bool


class ContributionDraft(Model):
    expected_revision_id: Identifier | None
    title: str = Field(min_length=1, max_length=256)
    summary: str = Field(min_length=1, max_length=8192)


class ContributionRevision(ResponseModel):
    contribution_id: Identifier
    revision_id: Identifier
    digest_sha256: Digest


class RightsAttestation(Model):
    revision_id: Identifier
    digest_sha256: Digest
    license_expression: str = Field(min_length=1, max_length=256)
    authorized_to_contribute: Literal[True]


class AttestationReceipt(ResponseModel):
    contribution_id: Identifier
    revision_id: Identifier
    accepted: bool
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
