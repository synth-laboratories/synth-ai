"""Exact release approval inputs and safe reproduction projection.

See notes/specifications/synth-index/research-archive-release.md.
"""

from typing import Literal

from pydantic import Field

from ..contracts import (
    ContributionAudience,
    Identifier,
    IndexContract,
)
from .contracts import DerivationBinding, Digest, ReleaseDisclosure, ReproductionReceipt


class ResearchBindingSpec(IndexContract):
    archive_publication_id: Identifier
    archive_manifest_digest: Digest
    binding: DerivationBinding


class ReleaseConsentSpec(IndexContract):
    manifest_digest: Digest
    disclosure_digest: Digest
    audience: ContributionAudience


class ReleaseConsentView(IndexContract):
    revision_id: Identifier
    manifest_digest: Digest
    disclosure_digest: Digest
    audience: ContributionAudience
    consented: bool


class ReleaseReproductionView(IndexContract):
    scope: Literal["artifact_reconstruction", "analysis_recomputation", "experimental_rerun"]
    outcome: Literal["passed", "failed", "not_applicable"]
    limitations: str = Field(min_length=1, max_length=2048)


class ReleaseResearchView(IndexContract):
    schema_version: Literal["synth.index.release-research.v2"] = "synth.index.release-research.v2"
    disclosure: ReleaseDisclosure
    disclosure_digest: Digest
    reproduction: tuple[ReleaseReproductionView, ...] = Field(max_length=64)


class ReproductionAttestationSpec(IndexContract):
    binding_digest: Digest
    receipt: ReproductionReceipt


class ResearchRevocationSpec(IndexContract):
    disclosure_digest: Digest


class ResearchArchiveAllocationSpec(IndexContract):
    snapshot_id: Identifier


class ResearchArchiveView(IndexContract):
    schema_version: Literal["synth.index.private-research.v2"] = "synth.index.private-research.v2"
    archive_publication_id: Identifier
    binding: DerivationBinding
    attestations: tuple[ReproductionReceipt, ...] = Field(max_length=3)
