"""Reviewer metadata decisions; never scientific acceptance or archive access.

See notes/specifications/synth-index/tag-classification.md.
"""

from typing import Annotated, Literal

from pydantic import Field, StrictInt, model_validator

from .contracts import (
    ContributionReference,
    Identifier,
    IndexContract,
    ShortText,
    require_unique,
)
from .research.contracts import Digest

Generation = Annotated[StrictInt, Field(ge=0, le=9_007_199_254_740_991)]


class ClassificationSpec(IndexContract):
    """Reviewer tag classification against exact manifest and registry inputs."""
    #: SHA-256 digest of the exact canonical manifest bytes.
    manifest_digest: Digest
    #: Tag registry identity against which reviewer classifications are validated.
    registry_version: Identifier
    #: Distinct registry tags accepted by the reviewer; does not certify scientific quality.
    accepted_tag_ids: tuple[Identifier, ...] = Field(max_length=10)
    #: Current classification generation expected by the write; rejects stale updates.
    expected_generation: Annotated[StrictInt, Field(ge=0, le=9_007_199_254_740_990)]
    #: Reviewer explanation of the tag classification decision.
    rationale: ShortText

    @model_validator(mode="after")
    def unique_tags(self):
        require_unique(self.accepted_tag_ids, "accepted tag IDs")
        return self


class ClassificationView(IndexContract):
    """Current effective tags and their generation; not scientific acceptance."""
    #: Exact boundary schema identifier used for serialization.
    schema_version: Literal["synth.index.classification.v1"] = "synth.index.classification.v1"
    #: Exact Contribution and revision described by this classification.
    reference: ContributionReference
    #: SHA-256 digest of the exact canonical manifest bytes.
    manifest_digest: Digest
    #: Tags originally submitted with the sealed revision.
    submitted_tag_ids: tuple[Identifier, ...] = Field(max_length=10)
    #: Tags effective after the current classification decision.
    effective_tag_ids: tuple[Identifier, ...] = Field(max_length=10)
    #: Monotonic classification generation identifying this metadata state.
    generation: Generation
    #: Tag registry identity against which reviewer classifications are validated.
    registry_version: Identifier | None
    #: Origin of effective tags: current submission, legacy submission or reviewer decision.
    source: Literal["submitted", "legacy_submission", "reviewer"]


class ClassificationDecisionView(IndexContract):
    """Stored reviewer tag decision against a sealed revision."""
    #: Exact boundary schema identifier used for serialization.
    schema_version: Literal["synth.index.classification-decision.v1"] = (
        "synth.index.classification-decision.v1"
    )
    #: Identifier of the recorded classification decision.
    decision_id: Identifier
    #: Exact Contribution and revision described by this classification.
    reference: ContributionReference
    #: SHA-256 digest of the exact canonical manifest bytes.
    manifest_digest: Digest
    #: Tag registry identity against which reviewer classifications are validated.
    registry_version: Identifier
    #: Monotonic classification generation identifying this metadata state.
    generation: Annotated[StrictInt, Field(ge=1, le=9_007_199_254_740_991)]
    #: Distinct registry tags accepted by the reviewer; does not certify scientific quality.
    accepted_tag_ids: tuple[Identifier, ...] = Field(max_length=10)
