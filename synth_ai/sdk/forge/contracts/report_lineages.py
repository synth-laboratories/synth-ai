# Generated from Forge packages/forge/report_lineages.py (Forge dev 483a80d).
# fmt: off
"""Report lineage index contracts (SYN-5158).

Forge owns report lineage. These are the caller-scoped reads over the lineage
index of hosted report versions (``forge_service.report_lineages``):

- ``ReportVersionStatus``: the lineage of one version, its head and whether the
  version is ``current``, ``stale`` or ``retracted`` (``CitationStatus``), plus
  the version that superseded it, if any.
- ``ReportVersionListing``: every version in a project the caller can read,
  including private versions that were never shared. Visibility is the Forge
  project ACL (``read``); sharing is a separate, narrower audience.
- ``ReportVersionAvailability``: per-artifact availability of one version as
  the acting user sees it now (``available``, ``retained``, ``expired``,
  ``missing``) from the manifest and evidence custody.

Nothing here grants Artifact Platform access or is an Index contribution.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field

from .contracts import Contract, Digest, ExactReference, Identifier, Scope
from .project_authorization import ProjectAuthorization
from .projects import ProjectId, ProjectPrincipal
from .report_manifests import CitationStatus, MissingEvidence, ReportVersionLink

Disposition = Literal["initial", "correction", "supersession", "retraction"]
LISTING_LIMIT_MAX = 1000


class LineageVersion(Contract):
    version: Annotated[int, Field(strict=True, ge=1)]
    version_sha256: Digest
    disposition: Disposition
    previous_sha256: Digest | None = None
    report_record_id: Identifier
    report_revision: Annotated[int, Field(strict=True, ge=1)]
    published_at: AwareDatetime
    publisher_authority: Identifier
    publisher_subject_id: Identifier
    record: ExactReference


class ReportLineage(Contract):
    schema_version: Literal["forge.report-lineage.v1"] = "forge.report-lineage.v1"
    scope: Scope
    lineage_id: Identifier
    versions: tuple[LineageVersion, ...] = Field(min_length=1)
    head: ReportVersionLink
    state: Literal["active", "retracted"]


class ReportVersionStatus(Contract):
    schema_version: Literal["forge.report-version-status.v1"] = (
        "forge.report-version-status.v1"
    )
    scope: Scope
    version_sha256: Digest
    citation: CitationStatus
    superseded_by: ReportVersionLink | None = None
    lineage: ReportLineage
    index_contribution: Literal["none"] = "none"


class ReportVersionListEntry(Contract):
    lineage_id: Identifier
    version: LineageVersion
    head: ReportVersionLink
    state: Literal["current", "stale", "retracted"]


class ReportVersionListing(Contract):
    schema_version: Literal["forge.report-version-listing.v1"] = (
        "forge.report-version-listing.v1"
    )
    scope: Scope
    entries: tuple[ReportVersionListEntry, ...]
    next_after: Digest | None = None
    index_contribution: Literal["none"] = "none"


class ArtifactAvailability(Contract):
    """``available``: evidence custody serves these bytes to the caller now.
    ``retained``: custody holds no bytes for the caller, but the manifest pins
    an exact Forge artifact record that this scope still retains.
    ``expired``: custody holds the bytes but the caller's grant is not live.
    ``missing``: neither bytes nor a retained record."""

    artifact_id: Identifier
    content_sha256: Digest
    state: Literal["available", "retained", "expired", "missing"]
    record: ExactReference | None = None
    error_code: Identifier | None = None


class ReportVersionAvailability(Contract):
    schema_version: Literal["forge.report-version-availability.v1"] = (
        "forge.report-version-availability.v1"
    )
    scope: Scope
    version_sha256: Digest
    artifacts: tuple[ArtifactAvailability, ...]
    missing_evidence: tuple[MissingEvidence, ...]
    index_contribution: Literal["none"] = "none"


class LineageRebuild(Contract):
    schema_version: Literal["forge.report-lineage-rebuild.v1"] = (
        "forge.report-lineage-rebuild.v1"
    )
    scope: Scope
    lineages: int = Field(ge=0)
    indexed: int = Field(ge=0)
    already_indexed: int = Field(ge=0)


# -- signed internal-port requests (backend proof, like publishing) -----------


class _ReportRead(Contract):
    schema_version: Literal["forge.report-lineage.v1"] = "forge.report-lineage.v1"
    principal: ProjectPrincipal
    project_id: ProjectId
    authorization: ProjectAuthorization


class ReportVersionStatusRead(_ReportRead):
    version_sha256: Digest


class ReportVersionAvailabilityRead(_ReportRead):
    version_sha256: Digest


class ReportVersionListRead(_ReportRead):
    after: Digest | None = None
    limit: Annotated[int, Field(strict=True, ge=1, le=LISTING_LIMIT_MAX)] = 100


class ReportLineageRebuild(_ReportRead):
    pass
