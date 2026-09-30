"""Independent-assignment content judgments, not release or scientific certificates.

See notes/specifications/synth-index/contribution-qa-cases.md. Kind-specific
criteria freeze one rubric. Honest negative results can pass evidence alignment;
missing scientific work cannot be converted to evidence by an author declaration.
"""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, StrictInt, model_validator

from .contracts import (
    ContributionKind,
    Identifier,
    IndexContract,
    require_unique,
)
from .lifecycle import Digest, ReviewDecision
from .qa import Text
from .qa_checks import EvidenceSelector, Outcome

RUBRIC_VERSION = "content-v1"
COMMON = (
    "content.claim_alignment",
    "content.evidence_quality",
    "content.limitations",
    "provenance.attribution",
    "privacy.review_scope",
)
KIND_CRITERIA = {
    ContributionKind.RESEARCH_REPORT: (
        "research.baseline",
        "research.protocol",
        "research.uncertainty",
    ),
    ContributionKind.REPLICATION: (
        "research.baseline",
        "research.protocol",
        "research.uncertainty",
        "replication.source_fidelity",
        "replication.differences",
    ),
    ContributionKind.DATASET: (
        "dataset.schema",
        "dataset.sampling",
        "dataset.contamination",
    ),
    ContributionKind.RECIPE: (
        "reproduction.environment",
        "reproduction.entrypoint",
        "reproduction.failure_modes",
    ),
    ContributionKind.IMPLEMENTATION: (
        "reproduction.environment",
        "reproduction.entrypoint",
        "reproduction.failure_modes",
    ),
    ContributionKind.MODEL: (
        "model.training_provenance",
        "model.evaluation",
        "model.access_restrictions",
    ),
    ContributionKind.ENGINEERING_NOTE: (
        "engineering.scope",
        "engineering.operational_evidence",
    ),
}


def required_criteria(kind):
    return COMMON + KIND_CRITERIA[ContributionKind(kind)]


class CriterionJudgment(IndexContract):
    criterion: Identifier
    outcome: Outcome
    rationale: Text
    evidence: tuple[EvidenceSelector, ...] = Field(min_length=1, max_length=32)
    check_attempt_ids: tuple[UUID, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_links(self):
        require_unique(tuple((s.kind, s.identifier) for s in self.evidence), "review evidence")
        require_unique(self.check_attempt_ids, "review check attempts")
        return self


class RecordReviewSpec(IndexContract):
    review_id: UUID
    expected_version: Annotated[StrictInt, Field(ge=0)]
    assignment_id: UUID
    manifest_digest: Digest
    rubric_version: Literal["content-v1"] = RUBRIC_VERSION
    decision: ReviewDecision
    summary: Text
    criteria: tuple[CriterionJudgment, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_judgments(self):
        require_unique(tuple(c.criterion for c in self.criteria), "review criteria")
        if self.decision == ReviewDecision.APPROVE and any(
            c.outcome not in {"pass", "not_applicable"} for c in self.criteria
        ):
            raise ValueError("Approval cannot hide failed or inconclusive criteria")
        if self.decision == ReviewDecision.REJECT and not any(
            c.outcome == "fail" for c in self.criteria
        ):
            raise ValueError("Rejection requires an evidenced failed criterion")
        if self.decision == ReviewDecision.REQUEST_CHANGES and not any(
            c.outcome in {"fail", "inconclusive"} for c in self.criteria
        ):
            raise ValueError("Changes require a failed or unresolved criterion")
        if len(self.model_dump_json().encode()) > 131072:
            raise ValueError("Review exceeds metadata byte budget")
        return self


def validate_review(spec, kind, provenance):
    required = set(required_criteria(kind))
    supplied = {c.criterion for c in spec.criteria}
    if supplied - required:
        raise ValueError("Review includes a criterion outside the pinned kind rubric")
    if spec.decision == ReviewDecision.APPROVE:
        if supplied != required:
            raise ValueError("Approval requires the full kind-specific rubric")
        if provenance not in {"human", "agent_assisted"}:
            raise ValueError("Uncalibrated agent-only approval is disabled")
        if any(c.outcome != "pass" for c in spec.criteria if c.criterion in COMMON):
            raise ValueError("Common content and privacy-scope judgments cannot be waived")
    if provenance not in {"human", "agent_assisted", "agent"}:
        raise ValueError("Unknown accepted assignment provenance")


class ReviewFact(IndexContract):
    case_id: UUID
    case_sequence: Annotated[StrictInt, Field(ge=1)]
    event_id: UUID
    reviewer_user_id: UUID
    reviewer_org_id: UUID
    declared_provenance: Literal["human", "agent_assisted", "agent"]
    review: RecordReviewSpec
    # Assignment consent/provenance are attestations, never proof of a human turn.
    human_identity_verified: Literal[False] = False
    scientific_verification: Literal[False] = False
    qualification_authorized: Literal[False] = False
    publication_authorized: Literal[False] = False


class ReviewReport(IndexContract):
    reviews: tuple[ReviewFact, ...] = Field(max_length=3)
    next_after: Annotated[StrictInt, Field(ge=1)] | None = None
