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
    manifest_digest: Digest
    registry_version: Identifier
    accepted_tag_ids: tuple[Identifier, ...] = Field(max_length=10)
    expected_generation: Annotated[StrictInt, Field(ge=0, le=9_007_199_254_740_990)]
    rationale: ShortText

    @model_validator(mode="after")
    def unique_tags(self):
        require_unique(self.accepted_tag_ids, "accepted tag IDs")
        return self


class ClassificationView(IndexContract):
    schema_version: Literal["synth.index.classification.v1"] = "synth.index.classification.v1"
    reference: ContributionReference
    manifest_digest: Digest
    submitted_tag_ids: tuple[Identifier, ...] = Field(max_length=10)
    effective_tag_ids: tuple[Identifier, ...] = Field(max_length=10)
    generation: Generation
    registry_version: Identifier | None
    source: Literal["submitted", "legacy_submission", "reviewer"]


class ClassificationDecisionView(IndexContract):
    schema_version: Literal["synth.index.classification-decision.v1"] = (
        "synth.index.classification-decision.v1"
    )
    decision_id: Identifier
    reference: ContributionReference
    manifest_digest: Digest
    registry_version: Identifier
    generation: Annotated[StrictInt, Field(ge=1, le=9_007_199_254_740_991)]
    accepted_tag_ids: tuple[Identifier, ...] = Field(max_length=10)
