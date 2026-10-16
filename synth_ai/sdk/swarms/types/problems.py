"""Typed run problems (`synth.problems.v1`): the reason a run is stuck or failed.

One `Problem` per root cause with the reason chain root first. Orchestra emits
the owner wire (`since_unix_ms`); the customer wire renders RFC 3339 `since` and
`last_seen`. See backend/.. fixes/problems-bubbling-design.md. The code set is
closed: a value outside it is refused at parse, so adding a code is a schema
version decision, never a silent string.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, JsonValue, StrictInt, model_validator

from synth_ai.types.base import Identifier, Model

PROBLEMS_SCHEMA = "synth.problems.v1"

ProblemCode = Literal[
    "provider_refused_prompt",
    "provider_rate_limited",
    "provider_unavailable",
    "provider_stream_truncated",
    "sandbox_exec_result_lost",
    "sandbox_unreachable",
    "sandbox_teardown_unverified",
    "actor_attempt_failed",
    "actor_silent",
    "task_stalled",
    "lease_expired_no_live_attempt",
    "scope_dormant",
    "cancel_not_terminal",
    "review_absent_auto_finalized",
    "dispatch_withheld_by_limits",
    "settlement_exhausted",
    "accounting_disagreement",
    "head_input_refused_repeatedly",
    "scope_quarantined",
    "owner_unavailable",
    "run_wall_clock_exceeded",
    "unclassified_fault",
]
ProblemSeverity = Literal["info", "warning", "error", "critical"]
ProblemState = Literal["open", "resolved"]
# `provider:<name>`, a closed service name, or `unknown` (which carries
# `origin_tried` in its chain link's cause).
Origin = Annotated[
    str,
    Field(
        strict=True,
        max_length=96,
        pattern=(
            r"^(provider:[A-Za-z0-9_.-]{1,64}|gateway|agents|orchestra|sublinear|"
            r"backend|daytona|tunnel|database|host|unknown)$"
        ),
    ),
]
Text = Annotated[str, Field(strict=True, min_length=1, max_length=1024)]


class EvidenceRef(Model):
    service: Annotated[str, Field(strict=True, min_length=1, max_length=96)]
    kind: Annotated[str, Field(strict=True, min_length=1, max_length=64)]
    ref: Annotated[str, Field(strict=True, min_length=1, max_length=256)]


class ReasonLink(Model):
    service: Origin
    code: Annotated[str, Field(strict=True, min_length=1, max_length=160)]
    stage: Annotated[str, Field(strict=True, min_length=1, max_length=96)]
    # Typed detail. An object by construction; a null cause is a defect.
    cause: dict[str, JsonValue]


class Problem(Model):
    id: Identifier
    code: ProblemCode
    severity: ProblemSeverity
    origin: Origin
    headline: Text
    reason_chain: tuple[ReasonLink, ...] = Field(min_length=1, max_length=16)
    since: AwareDatetime
    last_seen: AwareDatetime
    count: StrictInt = Field(ge=1)
    evidence_refs: tuple[EvidenceRef, ...] = Field(max_length=16)
    retryable: bool
    in_doubt: bool
    next: Text
    owner_hint: Text
    state: ProblemState

    @model_validator(mode="after")
    def validate_times(self) -> Self:
        if self.last_seen < self.since:
            raise ValueError("last_seen precedes since")
        return self
