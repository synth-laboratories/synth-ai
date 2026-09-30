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
    kind: Literal["asset", "claim", "evidence"]
    identifier: Identifier


class ReceiptMetric(IndexContract):
    name: Identifier
    value: Annotated[Decimal, Field(ge=-(10**18), le=10**18, max_digits=30, decimal_places=10)]
    unit: Identifier


class CheckReceipt(IndexContract):
    schema_version: Literal["synth.qa.check-receipt.v1"] = "synth.qa.check-receipt.v1"
    manifest_digest: Digest
    rubric_version: Identifier
    gate: Identifier
    run_id: Identifier
    tool_version: Identifier
    policy_version: Identifier
    outcome: Outcome
    started_at: AwareDatetime
    finished_at: AwareDatetime
    metrics: tuple[ReceiptMetric, ...] = Field(default=(), max_length=64)
    evidence: tuple[EvidenceSelector, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_receipt(self):
        if not self.started_at <= self.finished_at <= self.started_at + timedelta(days=30):
            raise ValueError("QA receipt time range invalid")
        require_unique(tuple(metric.name for metric in self.metrics), "metric names")
        require_unique(tuple((s.kind, s.identifier) for s in self.evidence), "receipt selectors")
        return self


class FindingSpec(IndexContract):
    finding_id: UUID
    criterion: Identifier
    severity: Literal["blocker", "change_requested", "advisory"]
    category: Literal["candidate_defect", "infrastructure_error", "unresolved"]
    visibility: Visibility
    selectors: tuple[EvidenceSelector, ...] = Field(default=(), max_length=32)
    observed: Text
    expected: Text
    reproduction: Text
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
    expected_version: Annotated[StrictInt, Field(ge=0)]
    attempt_id: UUID
    manifest_digest: Digest
    rubric_version: Identifier
    gate: Identifier
    run_id: Identifier
    tool_version: Identifier
    policy_version: Identifier
    outcome: Outcome
    reason: Text
    started_at: AwareDatetime
    finished_at: AwareDatetime
    cost_usd: Cost = Decimal(0)
    provenance: Literal["human", "automation", "agent"]
    receipt: CheckReceipt | None = None
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
    attempt_id: UUID
    producer_user_id: UUID
    provenance: Literal["human", "automation", "agent"]


class CheckAttemptView(IndexContract):
    attempt_id: UUID
    case_id: UUID
    event_id: UUID
    case_sequence: Annotated[StrictInt, Field(ge=1)]
    manifest_digest: Digest
    rubric_version: Identifier
    gate: Identifier
    run_id: Identifier
    tool_version: Identifier
    policy_version: Identifier
    outcome: Outcome
    reason: Text
    started_at: AwareDatetime
    finished_at: AwareDatetime
    cost_usd: Cost
    producer_user_id: UUID
    provenance: Literal["human", "automation", "agent"]
    receipt: CheckReceipt | None
    receipt_digest: Digest | None
    # Integrity of canonical stored JSON is not scientific verification.
    independent_verification: Literal[False] = False
    publication_authorized: Literal[False] = False


class CheckReport(IndexContract):
    attempts: tuple[CheckAttemptView, ...] = Field(max_length=3)
    findings: tuple[FindingView, ...] = Field(max_length=192)
    next_after: Annotated[StrictInt, Field(ge=1)] | None = None
