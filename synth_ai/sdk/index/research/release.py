"""Exact release approval inputs and safe reproduction projection.

See notes/specifications/synth-index/research-archive-release.md.
"""

from typing import Literal

from pydantic import Field

from ..artifacts import ArtifactResourceScope, ArtifactUuid
from ..contracts import (
    ContributionAudience,
    Identifier,
    IndexContract,
)
from .contracts import DerivationBinding, Digest, ReleaseDisclosure, ReproductionReceipt


class ResearchBindingSpec(IndexContract):
    """Bind an exact private archive publication to validated release derivation inputs."""
    #: Private artifact publication identity retaining the frozen archive.
    archive_publication_id: Identifier
    #: SHA-256 digest of the exact private archive manifest.
    archive_manifest_digest: Digest
    #: Validated join of exact archive, recipe, disclosure and reproduction inputs.
    binding: DerivationBinding


class ReleaseConsentSpec(IndexContract):
    """Consent to the exact sealed manifest, disclosure and requested audience."""
    #: SHA-256 digest of the sealed Contribution manifest.
    manifest_digest: Digest
    #: SHA-256 digest of the canonical approved release disclosure.
    disclosure_digest: Digest
    #: Audience approved for this exact release disclosure.
    audience: ContributionAudience


class ReleaseConsentView(IndexContract):
    """Recorded exact-revision audience consent; not a publication receipt."""
    #: Exact Contribution revision to which this consent applies.
    revision_id: Identifier
    #: SHA-256 digest of the sealed Contribution manifest.
    manifest_digest: Digest
    #: SHA-256 digest of the canonical approved release disclosure.
    disclosure_digest: Digest
    #: Audience approved for this exact release disclosure.
    audience: ContributionAudience
    #: Whether exact manifest, disclosure and audience consent is recorded.
    consented: bool


class ReleaseReproductionView(IndexContract):
    """Public-safe reproduction scope, outcome and limitations without private archive identifiers."""
    #: What was actually reproduced: artifact reconstruction, analysis recomputation or experimental rerun.
    scope: Literal["artifact_reconstruction", "analysis_recomputation", "experimental_rerun"]
    #: Recorded result; failed, excluded, cancelled and abandoned attempts remain evidence.
    outcome: Literal["passed", "failed", "not_applicable"]
    #: Explicit limitations of reproduction scope or outcome.
    limitations: str = Field(min_length=1, max_length=2048)


class ReleaseResearchView(IndexContract):
    """Public-safe disclosure and reproduction projection."""
    #: Exact wire schema identifier.
    schema_version: Literal["synth.index.release-research.v2"] = "synth.index.release-research.v2"
    #: Exact audience-bound approved release projection.
    disclosure: ReleaseDisclosure
    #: SHA-256 digest of the canonical approved release disclosure.
    disclosure_digest: Digest
    #: Public-safe reproduction observations with explicit limitations.
    reproduction: tuple[ReleaseReproductionView, ...] = Field(max_length=64)


class ReproductionAttestationSpec(IndexContract):
    """Attest a reproduction receipt against the exact derivation binding digest."""
    #: SHA-256 digest of the canonical derivation binding.
    binding_digest: Digest
    #: Verifier receipt bound to the exact derivation inputs and outputs.
    receipt: ReproductionReceipt


class ResearchRevocationSpec(IndexContract):
    """Revoke access to the exact approved disclosure."""
    #: SHA-256 digest of the canonical approved release disclosure.
    disclosure_digest: Digest


class ResearchArchiveAllocationSpec(IndexContract):
    """Request private archive allocation for a frozen snapshot identity."""
    #: Frozen snapshot identity.
    snapshot_id: Identifier


class ResearchArchiveAllocation(IndexContract):
    """Customer allocation receipt; storage namespace stays server-side.

    See notes/specifications/synth-index/research-archive-release.md.
    """

    #: Exact wire schema identifier.
    schema_version: Literal["synth.index.research-archive-allocation.v1"] = (
        "synth.index.research-archive-allocation.v1"
    )
    #: Allocated private archive collection UUID.
    collection_id: ArtifactUuid
    #: What was actually reproduced: artifact reconstruction, analysis recomputation or experimental rerun.
    scope: ArtifactResourceScope


class ResearchArchiveView(IndexContract):
    """Authorized private archive binding and bounded reproduction attestations."""
    #: Exact wire schema identifier.
    schema_version: Literal["synth.index.private-research.v2"] = "synth.index.private-research.v2"
    #: Private artifact publication identity retaining the frozen archive.
    archive_publication_id: Identifier
    #: Validated join of exact archive, recipe, disclosure and reproduction inputs.
    binding: DerivationBinding
    #: Bounded verifier attestations retained with the authorized private archive.
    attestations: tuple[ReproductionReceipt, ...] = Field(max_length=3)
