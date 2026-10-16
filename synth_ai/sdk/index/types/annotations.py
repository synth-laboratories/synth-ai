"""Revision-bound public annotations. See backend SDK index_contract.md."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from synth_ai.types.base import Digest, Identifier, Model

RatingLevel = Literal["low", "med", "high"]
Keyword = Annotated[
    str, Field(strict=True, min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
]


class ContributionMetadata(Model):
    research_areas: tuple[Identifier, ...] = Field(default=(), max_length=20)
    workflow_stages: tuple[Identifier, ...] = Field(default=(), max_length=20)
    topics: tuple[Identifier, ...] = Field(default=(), max_length=20)

    @model_validator(mode="after")
    def validate_metadata(self) -> Self:
        for values in (self.research_areas, self.workflow_stages, self.topics):
            if len(values) != len(set(values)):
                raise ValueError("metadata identifiers must be unique")
        return self


class AuthorAnnotations(Model):
    keywords: tuple[Keyword, ...] = Field(default=(), max_length=20)
    metadata: ContributionMetadata = Field(default_factory=ContributionMetadata)

    @model_validator(mode="after")
    def validate_keywords(self) -> Self:
        if len(self.keywords) != len(set(self.keywords)):
            raise ValueError("keywords must be unique")
        return self


class ContributionReview(Model):
    contribution_id: Identifier
    revision_id: Identifier
    revision_digest_sha256: Digest
    review_id: Identifier
    rubric_version: Identifier
    reviewed_at: AwareDatetime
    reviewer_id: Identifier
    authority: Literal["synth"]
    quality: RatingLevel
    confidence: RatingLevel
    correctness: RatingLevel
    evidence_references: tuple[Identifier, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if len(self.evidence_references) != len(set(self.evidence_references)):
            raise ValueError("review evidence references must be unique")
        return self


class ContributionAnnotations(AuthorAnnotations):
    contribution_id: Identifier
    revision_id: Identifier
    revision_digest_sha256: Digest
    annotation_version: Identifier
    review: ContributionReview | None

    @model_validator(mode="after")
    def validate_review_identity(self) -> Self:
        if self.review is not None and (
            self.review.contribution_id != self.contribution_id
            or self.review.revision_id != self.revision_id
            or self.review.revision_digest_sha256 != self.revision_digest_sha256
        ):
            raise ValueError("review belongs to another contribution revision")
        return self
