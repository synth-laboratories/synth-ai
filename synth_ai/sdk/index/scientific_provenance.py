"""Backend-authored safe scientific release credit; no publication grant.

# See: backend/notes/specifications/synth-index/forge-intake.md
Private Forge/archive identities belong to the separate source registration.
"""

from typing import Literal

from pydantic import Field, model_validator

from .contracts import ContributionReference, Identifier, IndexContract, ShortText, require_unique
from .research.contracts import Digest, canonical_bytes


class PublicUpstreamCredit(IndexContract):
    reference: ContributionReference
    digest_sha256: Digest


class PublicScientificCredit(IndexContract):
    principal_id: Identifier
    role: Literal["author", "data", "implementation", "method", "review", "funding"]
    license: ShortText
    upstream: PublicUpstreamCredit | None = None


class PublicScientificAsset(IndexContract):
    asset_id: Identifier
    digest_sha256: Digest
    license: ShortText


class PublicScientificProvenance(IndexContract):
    schema_version: Literal["synth.index.scientific-provenance.v1"] = (
        "synth.index.scientific-provenance.v1"
    )
    provenance_license: str = Field(min_length=1, max_length=256)
    assets: tuple[PublicScientificAsset, ...] = Field(min_length=1, max_length=1024)
    credits: tuple[PublicScientificCredit, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def unique_selection(self):
        require_unique(tuple(item.asset_id for item in self.assets), "public assets")
        require_unique(tuple(canonical_bytes(item) for item in self.credits), "credits")
        if not any(item.role != "funding" for item in self.credits):
            raise ValueError("Contribution v1 requires an actual credited author role")
        if len({item.principal_id for item in self.credits}) > 32:
            raise ValueError("Full credit exceeds the existing 32-principal QA bound")
        return self
