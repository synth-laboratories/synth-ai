"""Backend-owned Contribution draft wire mirror; unreleased.

See sibling backend/packages/contributions/upload.py and cross-repo schema tests.
"""

import ipaddress
import re
from datetime import date, datetime
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import BeforeValidator, Field, StringConstraints, field_validator, model_validator

from synth_ai.sdk.research.contracts.forge.contracts import ExactReference, Identifier
from synth_ai.sdk.research.contracts.forge.operations import RevisionReference

from .artifacts import ArtifactPublicationPrepareResponse, ArtifactUuid
from .contracts import ContributionReference, IndexContract
from .package import ContributionPackage
from .research.contracts import Digest
from .scientific_provenance import PublicScientificProvenance

NonEmpty = Annotated[str, StringConstraints(min_length=1, max_length=2048)]
_URL_SECRET_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bsk-[A-Za-z0-9_-]{20,}",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"\bghp_[A-Za-z0-9]{30,}",
        r"\bxox[bpas]-[A-Za-z0-9-]{10,}",
    )
)


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
        hostname = url.hostname
        if (
            url.scheme != "https"
            or not hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or url.port not in (None, 443)
            or hostname.lower() in {"localhost", "localhost.localdomain"}
            or hostname.lower().endswith((".local", ".internal"))
            or any(pattern.search(value) for pattern in _URL_SECRET_PATTERNS)
        ):
            raise ValueError("repository URL must be credential-free public HTTPS")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError("repository URL must be credential-free public HTTPS")
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


class ForgeResearchSource(IndexContract):
    """Private Forge association; backend resolves current access and exact bytes.

    See sibling backend/notes/specifications/synth-index/forge-intake.md.
    """

    schema_version: Literal["synth.index.forge-source.v1"] = "synth.index.forge-source.v1"
    organization_id: Identifier
    project_id: Identifier
    export: ExactReference

    @model_validator(mode="after")
    def exact_export(self):
        if self.export.authority != "forge" or self.export.kind != "scientific_export":
            raise ValueError("Forge source requires an exact scientific export")
        return self


ARCHIVE_V3 = "forge.private-archive.v3"


class ForgePublicResearchSource(ForgeResearchSource):
    """Exact private archive plus safe proposed public credit; backend decides release.

    Mirrors backend ``packages/contributions/forge_release.py``. ``revision_reference``
    (Forge decision 0001, ``forge.revision-reference.v1``) binds the selected export's
    producer, operation and time, not only its payload; ``archive_schema_version`` names
    the private archive version that carries those references (v3). The backend public
    review policy refuses a Forge-origin public source that omits either
    (``forge_revision_reference_required``, ``forge_archive_v3_required``); they stay
    optional on the wire so older private registrations still decode.
    """

    schema_version: Literal["synth.index.forge-source.v2"] = "synth.index.forge-source.v2"
    archive_publication_id: ArtifactUuid
    archive_manifest_digest_sha256: Digest
    public_provenance: PublicScientificProvenance
    revision_reference: RevisionReference | None = None
    archive_schema_version: Literal["forge.private-archive.v3"] | None = None

    @model_validator(mode="after")
    def provenance_pin_names_the_export(self):
        if (
            self.revision_reference is not None
            and self.revision_reference.reference != self.export
        ):
            raise ValueError("revision reference must name the exported revision")
        return self


def _legacy_source_version(value):
    if isinstance(value, dict) and "schema_version" not in value:
        return {**value, "schema_version": "synth.index.forge-source.v1"}
    return value


VersionedForgeSource = Annotated[
    ForgeResearchSource | ForgePublicResearchSource,
    Field(discriminator="schema_version"),
    BeforeValidator(_legacy_source_version),
]


class ResearchDraftSpec(IndexContract):
    """Server allocates identity and SYNTH provenance; caller supplies source proof only."""

    bundle_digest: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
    source: ResearchSource | VersionedForgeSource

    @model_validator(mode="after")
    def forge_digest(self):
        if isinstance(self.source, ForgeResearchSource) and (
            self.bundle_digest != f"sha256:{self.source.export.digest_sha256}"
        ):
            raise ValueError("Forge bundle digest must bind the selected export")
        return self


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
