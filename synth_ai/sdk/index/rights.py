"""Exact revision rights claims; the backend alone decides attester authority.

See backend notes/specifications/synth-index/research-rights-attestation.md.
Rights attestation is separate from consent, QA acceptance and publication.
"""

from datetime import datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import Field, StringConstraints, field_validator

from .contracts import IndexContract

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
LicenseExpression = Annotated[str, StringConstraints(min_length=1, max_length=256)]
DecisionReference = Annotated[str, StringConstraints(min_length=1, max_length=512)]


class RightsQueue(StrEnum):
    """The configured queue for Synth-origin rights claims."""

    INDEX_CORPUS_QA = "index-corpus-qa"


class RightsAuthority(StrEnum):
    """Authority selected by the backend from the sealed contribution origin."""

    QUEUE_RIGHTS_OWNER = "queue_rights_owner"
    CONTRIBUTING_ORG_ADMIN = "contributing_org_admin"


class RightsAttestationSpec(IndexContract):
    """Exact rights claim for sealed bytes, without granting publication permission."""

    manifest_digest: Digest
    descriptor_digest: Digest
    rights_decision_ref: DecisionReference
    licenses: tuple[LicenseExpression, ...] = Field(min_length=1, max_length=64)
    notices_digest: Digest

    @field_validator("licenses")
    @classmethod
    def _canonical_licenses(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("licenses must be distinct")
        return tuple(sorted(value))


class RightsAttestationView(IndexContract):
    """A stored rights fact bound to one exact revision; never a publication decision."""

    schema_version: str = "research-rights-attestation-v1"
    revision_id: str
    authority: RightsAuthority
    queue: RightsQueue | None
    attester_user_id: UUID
    attester_org_id: UUID | None
    manifest_digest: Digest
    descriptor_digest: Digest
    rights_decision_ref: DecisionReference
    licenses: tuple[LicenseExpression, ...]
    notices_digest: Digest
    attested_at: datetime
