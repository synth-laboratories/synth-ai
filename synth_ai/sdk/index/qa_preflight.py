"""Frozen backend metadata-preflight wire mirror; not a local execution engine.

See sibling backend notes/specifications/synth-index/contribution-qa-cases.md.
"""

from typing import Annotated

from pydantic import Field, StrictInt, model_validator

from .contracts import Identifier, IndexContract
from .qa_checks import CheckAttemptView

TOOL_VERSION = "synth-qa-preflight-v1"
POLICY_VERSION = "structural-v1"
GATES = (
    "claims.evidence_links",
    "reproduction.manifest",
    "rights.declarations",
    "scientific.verification",
    "privacy.scan",
)


class RunPreflightSpec(IndexContract):
    """Request the fixed metadata preflight batch against the current case version."""
    #: Producer run identity shared by every attempt in the batch.
    run_id: Identifier
    #: Case version expected by this write; stale input must be reconciled.
    expected_version: Annotated[StrictInt, Field(ge=0)]


class PreflightResult(IndexContract):
    """Five ordered attempts; unexecuted science/privacy gates remain inconclusive."""
    #: Exactly the five fixed metadata/science/privacy gate observations.
    attempts: tuple[CheckAttemptView, ...] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def validate_batch(self):
        first = self.attempts[0]
        if len({a.attempt_id for a in self.attempts}) != len(GATES):
            raise ValueError("Preflight duplicate attempt")
        if tuple(a.gate for a in self.attempts) != GATES:
            raise ValueError("Preflight must return each exact gate in order")
        for offset, attempt in enumerate(self.attempts):
            if any(
                getattr(attempt, field) != getattr(first, field)
                for field in (
                    "case_id",
                    "manifest_digest",
                    "rubric_version",
                    "run_id",
                    "producer_user_id",
                    "started_at",
                    "finished_at",
                )
            ):
                raise ValueError("Preflight batch identity differs")
            if (
                attempt.tool_version != TOOL_VERSION
                or attempt.policy_version != POLICY_VERSION
                or attempt.cost_usd != 0
                or attempt.provenance != "automation"
                or attempt.case_sequence != first.case_sequence + offset
            ):
                raise ValueError("Preflight execution identity differs")
            if offset >= 3 and (attempt.outcome != "inconclusive" or attempt.receipt is not None):
                raise ValueError("Unexecuted gates cannot carry passing receipts")
            if offset < 3 and attempt.receipt is None:
                raise ValueError("Executed metadata gates require receipts")
        return self
