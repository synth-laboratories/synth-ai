"""Independent-assignment content judgments, not release or scientific certificates.

See notes/specifications/synth-index/contribution-qa-cases.md. Kind-specific
criteria freeze one rubric. Honest negative results can pass evidence alignment;
missing scientific work cannot be converted to evidence by an author declaration.
"""

from enum import StrEnum
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


def required_criteria(kind: ContributionKind | str) -> tuple[str, ...]:
    """Return the fixed common and kind-specific content rubric criteria.

    Args:
        kind: Supported Contribution kind selecting the frozen content rubric.
    Returns:
        Common criteria followed by criteria specific to the selected kind.
    Raises:
        ValueError: The selected kind is not a supported ContributionKind.
    Examples:
        criteria = required_criteria(ContributionKind.RESEARCH_REPORT)
    """
    return COMMON + KIND_CRITERIA[ContributionKind(kind)]


class CriterionJudgment(IndexContract):
    """Evidence-backed outcome for one criterion in the frozen content rubric."""
    #: Identifier of the frozen rubric criterion being judged.
    criterion: Identifier
    #: Recorded outcome; fail, inconclusive and not-applicable remain distinct.
    outcome: Outcome
    #: Evidence-backed explanation of this criterion judgment.
    rationale: Text
    #: Manifest item selectors supporting the recorded judgment or check.
    evidence: tuple[EvidenceSelector, ...] = Field(min_length=1, max_length=32)
    #: Identifiers of recorded check attempts referenced by this judgment.
    check_attempt_ids: tuple[UUID, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_links(self):
        require_unique(tuple((s.kind, s.identifier) for s in self.evidence), "review evidence")
        require_unique(self.check_attempt_ids, "review check attempts")
        return self


class RecordReviewSpec(IndexContract):
    """Record an independent assignment review against exact case and manifest inputs."""
    #: Identifier of the independent review.
    review_id: UUID
    #: Current case version expected by this write; stale writes must be reconciled.
    expected_version: Annotated[StrictInt, Field(ge=0)]
    #: Identifier of the independent reviewer assignment.
    assignment_id: UUID
    #: Digest of the sealed revision manifest; must match the reviewed bytes.
    manifest_digest: Digest
    #: Pinned rubric identity used for these judgments.
    rubric_version: Literal["content-v1"] = RUBRIC_VERSION
    #: Review decision constrained by evidenced criterion outcomes.
    decision: ReviewDecision
    #: Bounded reviewer summary explaining the decision.
    summary: Text
    #: Kind-specific criterion judgments required by the pinned rubric.
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


def validate_review(
    spec: RecordReviewSpec, kind: ContributionKind | str, provenance: str
) -> None:
    """Validate content review scope and provenance without authorizing publication.

    Args:
        spec: Exact revision-bound criterion judgments and recorded decision.
        kind: Contribution kind whose frozen rubric must match the criteria.
        provenance: Accepted human, agent_assisted or agent reviewer declaration.
    Returns:
        None; invalid judgments raise before recording a review.
    Raises:
        ValueError: Criteria are outside the rubric, approval is incomplete,
            common criteria are waived or provenance cannot authorize the decision.
    Examples:
        validate_review(spec, ContributionKind.RESEARCH_REPORT, "human")
    """
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


class ReviewerPolicy(StrEnum):
    """Recorded policy under which a review's approval power was admitted.

    Derived from the accepted assignment's declared provenance; never a claim of
    human identity.
    """

    HUMAN = "qa-human-review-v1"
    AGENT_ASSISTED = "qa-agent-assisted-review-v1"
    AGENT_INDEPENDENT = "qa-agent-independent-review-v1"


class ReviewFact(IndexContract):
    """Stored review, declared provenance and admitted reviewer policy."""
    #: Identifier of the revision-bound QA case.
    case_id: UUID
    #: Conversation sequence at which this fact was recorded.
    case_sequence: Annotated[StrictInt, Field(ge=1)]
    #: Identifier of the recorded QA event.
    event_id: UUID
    #: User identifier of the assigned reviewer.
    reviewer_user_id: UUID
    #: Organization identifier under which the reviewer acts.
    reviewer_org_id: UUID
    #: Accepted reviewer declaration; does not attest to human identity.
    declared_provenance: Literal["human", "agent_assisted", "agent"]
    #: Recorded policy that admitted the review under its declared provenance.
    reviewer_policy: ReviewerPolicy
    #: Exact recorded independent review and its criterion judgments.
    review: RecordReviewSpec
    # Assignment consent/provenance are attestations, never proof of a human turn.
    #: Always false: assignment declarations do not verify human identity.
    human_identity_verified: Literal[False] = False
    #: Always false: this review record is not a scientific verification certificate.
    scientific_verification: Literal[False] = False
    #: Always false: content judgments alone do not authorize release qualification.
    qualification_authorized: Literal[False] = False
    #: Always false: this QA record cannot grant publication authority.
    publication_authorized: Literal[False] = False


class ReviewReport(IndexContract):
    """Bounded independent review page with a continuation cursor."""
    #: Bounded recorded independent review facts.
    reviews: tuple[ReviewFact, ...] = Field(max_length=3)
    #: Continuation sequence cursor, or none when this page has no continuation.
    next_after: Annotated[StrictInt, Field(ge=1)] | None = None
