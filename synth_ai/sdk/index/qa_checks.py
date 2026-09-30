"""Immutable, revision-bound QA observations and actionable findings.

See notes/specifications/synth-index/contribution-qa-cases.md. A typed receipt is
producer evidence, not independent scientific verification or publication power.
Missing receipts and infrastructure failures are explicitly inconclusive.
"""

import json
from datetime import timedelta
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, StrictInt, model_validator

from .contracts import Identifier, IndexContract, require_unique
from .lifecycle import Digest
from .qa import Text

Outcome = Literal["pass", "fail", "inconclusive", "not_applicable"]
Visibility = Literal["contributor", "reviewer", "internal"]
Cost = Annotated[Decimal, Field(ge=0, le=1000000, max_digits=16, decimal_places=8)]


class EvidenceSelector(IndexContract):
    """Select one manifest asset, claim or evidence item by its identifier."""
    #: Manifest selector kind: asset, claim or evidence.
    kind: Literal["asset", "claim", "evidence"]
    #: Identifier of the selected manifest item.
    identifier: Identifier


class ReceiptMetric(IndexContract):
    """Named numerical measurement with an explicit unit."""
    #: Unique metric name within the receipt.
    name: Identifier
    #: Numerical metric value interpreted using its explicit unit.
    value: Annotated[Decimal, Field(ge=-(10**18), le=10**18, max_digits=30, decimal_places=10)]
    #: Unit in which the metric value is expressed.
    unit: Identifier


class CheckReceipt(IndexContract):
    """Producer receipt binding a check result to exact inputs and execution identity."""
    #: Exact receipt schema identity for serialization and validation.
    schema_version: Literal["synth.qa.check-receipt.v1"] = "synth.qa.check-receipt.v1"
    #: Digest of the sealed revision manifest; must match the reviewed bytes.
    manifest_digest: Digest
    #: Pinned rubric identity used for these judgments.
    rubric_version: Identifier
    #: Identifier of the check gate whose result is recorded.
    gate: Identifier
    #: Producer run identifier linking the result to execution evidence.
    run_id: Identifier
    #: Exact tool version used to produce the check.
    tool_version: Identifier
    #: Policy identity governing the producer check.
    policy_version: Identifier
    #: Recorded outcome; fail, inconclusive and not-applicable remain distinct.
    outcome: Outcome
    #: Timezone-aware check start timestamp.
    started_at: AwareDatetime
    #: Timezone-aware check finish timestamp, no earlier than its start.
    finished_at: AwareDatetime
    #: Bounded measurements with unique names and explicit units.
    metrics: tuple[ReceiptMetric, ...] = Field(default=(), max_length=64)
    #: Manifest item selectors supporting the recorded judgment or check.
    evidence: tuple[EvidenceSelector, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_receipt(self):
        if not self.started_at <= self.finished_at <= self.started_at + timedelta(days=30):
            raise ValueError("QA receipt time range invalid")
        require_unique(tuple(metric.name for metric in self.metrics), "metric names")
        require_unique(tuple((s.kind, s.identifier) for s in self.evidence), "receipt selectors")
        return self


class FindingSpec(IndexContract):
    """Actionable observation with reproduction, remediation and visibility."""
    #: Identifier of the actionable finding.
    finding_id: UUID
    #: Identifier of the frozen rubric criterion being judged.
    criterion: Identifier
    #: Whether the observation blocks acceptance, requests a change or is advisory.
    severity: Literal["blocker", "change_requested", "advisory"]
    #: Candidate defect, infrastructure failure or unresolved work; infrastructure failures are inconclusive.
    category: Literal["candidate_defect", "infrastructure_error", "unresolved"]
    #: Disclosure audience; internal notes must not reach the contributor.
    visibility: Visibility
    #: Manifest items relevant to this finding.
    selectors: tuple[EvidenceSelector, ...] = Field(default=(), max_length=32)
    #: Observed behavior or evidence supporting the finding.
    observed: Text
    #: Expected behavior against which the observation was judged.
    expected: Text
    #: Instructions or evidence needed to reproduce the observation.
    reproduction: Text
    #: Suggested corrective action for the finding.
    remediation: Text

    @model_validator(mode="after")
    def validate_selectors(self):
        require_unique(tuple((s.kind, s.identifier) for s in self.selectors), "finding selectors")
        # UTF-8 byte bounds keep report pages inside Monitor's buffered envelope.
        if (
            len(json.dumps(self.model_dump(mode="json"), ensure_ascii=False).encode("utf-8"))
            > 28672
        ):
            raise ValueError("Finding text exceeds the report byte budget")
        return self


class RecordCheckSpec(IndexContract):
    """Record a bounded check attempt; missing receipts must be inconclusive."""
    #: Current case version expected by this write; stale writes must be reconciled.
    expected_version: Annotated[StrictInt, Field(ge=0)]
    #: Identifier of this check attempt, distinct from the producer run.
    attempt_id: UUID
    #: Digest of the sealed revision manifest; must match the reviewed bytes.
    manifest_digest: Digest
    #: Pinned rubric identity used for these judgments.
    rubric_version: Identifier
    #: Identifier of the check gate whose result is recorded.
    gate: Identifier
    #: Producer run identifier linking the result to execution evidence.
    run_id: Identifier
    #: Exact tool version used to produce the check.
    tool_version: Identifier
    #: Policy identity governing the producer check.
    policy_version: Identifier
    #: Recorded outcome; fail, inconclusive and not-applicable remain distinct.
    outcome: Outcome
    #: Explanation of the recorded outcome and unresolved conditions.
    reason: Text
    #: Timezone-aware check start timestamp.
    started_at: AwareDatetime
    #: Timezone-aware check finish timestamp, no earlier than its start.
    finished_at: AwareDatetime
    #: Reported producer cost in US dollars; not an independently settled charge.
    cost_usd: Cost = Decimal(0)
    #: Declared source of the work; not proof of a verified human identity.
    provenance: Literal["human", "automation", "agent"]
    #: Matching typed producer receipt, or none for an inconclusive attempt.
    receipt: CheckReceipt | None = None
    #: Bounded actionable findings associated with this attempt.
    findings: tuple[FindingSpec, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def validate_attempt(self):
        if not self.started_at <= self.finished_at <= self.started_at + timedelta(days=30):
            raise ValueError("QA attempt time range invalid")
        require_unique(tuple(f.finding_id for f in self.findings), "finding IDs")
        if self.receipt is None and self.outcome != "inconclusive":
            raise ValueError("Missing receipt requires inconclusive outcome")
        if self.receipt is not None:
            for key in [
                "manifest_digest",
                "rubric_version",
                "gate",
                "run_id",
                "tool_version",
                "policy_version",
                "outcome",
                "started_at",
                "finished_at",
            ]:
                if getattr(self.receipt, key) != getattr(self, key):
                    raise ValueError(f"Receipt {key} differs from attempt")
        blocking = [f for f in self.findings if f.severity != "advisory"]
        if self.outcome in {"pass", "not_applicable"} and blocking:
            raise ValueError("Successful outcome cannot hide blocking findings")
        if (
            any(f.category == "infrastructure_error" for f in self.findings)
            and self.outcome != "inconclusive"
        ):
            raise ValueError("Infrastructure errors are inconclusive, never candidate failures")
        if self.outcome == "fail" and not any(
            f.category == "candidate_defect" and f.severity != "advisory" for f in self.findings
        ):
            raise ValueError("Failure needs an actionable candidate finding")
        if self.outcome == "inconclusive" and not any(
            f.category in {"infrastructure_error", "unresolved"} for f in self.findings
        ):
            raise ValueError(
                "Inconclusive outcome needs an explicit unresolved or infrastructure finding"
            )
        return self


class FindingView(FindingSpec):
    """Stored finding with producer and attempt attribution."""
    #: Identifier of this check attempt, distinct from the producer run.
    attempt_id: UUID
    #: User identifier that submitted the producer evidence.
    producer_user_id: UUID
    #: Declared source of the work; not proof of a verified human identity.
    provenance: Literal["human", "automation", "agent"]


class CheckAttemptView(IndexContract):
    """Stored producer check evidence, not independent scientific verification."""
    #: Identifier of this check attempt, distinct from the producer run.
    attempt_id: UUID
    #: Identifier of the revision-bound QA case.
    case_id: UUID
    #: Identifier of the recorded QA event.
    event_id: UUID
    #: Conversation sequence at which this fact was recorded.
    case_sequence: Annotated[StrictInt, Field(ge=1)]
    #: Digest of the sealed revision manifest; must match the reviewed bytes.
    manifest_digest: Digest
    #: Pinned rubric identity used for these judgments.
    rubric_version: Identifier
    #: Identifier of the check gate whose result is recorded.
    gate: Identifier
    #: Producer run identifier linking the result to execution evidence.
    run_id: Identifier
    #: Exact tool version used to produce the check.
    tool_version: Identifier
    #: Policy identity governing the producer check.
    policy_version: Identifier
    #: Recorded outcome; fail, inconclusive and not-applicable remain distinct.
    outcome: Outcome
    #: Explanation of the recorded outcome and unresolved conditions.
    reason: Text
    #: Timezone-aware check start timestamp.
    started_at: AwareDatetime
    #: Timezone-aware check finish timestamp, no earlier than its start.
    finished_at: AwareDatetime
    #: Reported producer cost in US dollars; not an independently settled charge.
    cost_usd: Cost
    #: User identifier that submitted the producer evidence.
    producer_user_id: UUID
    #: Declared source of the work; not proof of a verified human identity.
    provenance: Literal["human", "automation", "agent"]
    #: Matching typed producer receipt, or none for an inconclusive attempt.
    receipt: CheckReceipt | None
    #: Digest of canonical stored receipt bytes, or none when no receipt exists.
    receipt_digest: Digest | None
    # Integrity of canonical stored JSON is not scientific verification.
    #: Always false: stored producer receipts are not independent verification.
    independent_verification: Literal[False] = False
    #: Always false: this QA record cannot grant publication authority.
    publication_authorized: Literal[False] = False


class CheckReport(IndexContract):
    """Paginated recorded check attempts and findings."""
    attempts: tuple[CheckAttemptView, ...] = Field(max_length=3)
    #: Bounded actionable findings associated with this attempt.
    findings: tuple[FindingView, ...] = Field(max_length=192)
    #: Continuation sequence cursor, or none when this page has no continuation.
    next_after: Annotated[StrictInt, Field(ge=1)] | None = None
