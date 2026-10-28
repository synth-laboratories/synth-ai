"""Typed run problems (`synth.problems.v1`/`v2`): why a run is stuck or failed.

One `Problem` per root cause with the reason chain root first. Orchestra emits
the owner wire (`since_unix_ms`); the customer wire renders RFC 3339 `since` and
`last_seen`. Known codes and origins retain typed values; a future response code
is exposed as UnknownValue and a future origin as UnknownOrigin, never
interpreted as a successful outcome and never a decode failure.

Versions (SYN-4551): a client that sends `X-Synth-Accept-Problems:
synth.problems.v2` may receive origins outside the v1 set (`docker` today).
Every other caller receives v1 values only: the backend maps `docker` to `host`
(cause `sandbox_provider: docker`) and any other non-v1 origin to `unknown`
(cause `origin_tried`), so an SDK with the strict v1 origin pattern still
decodes every body.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import (
    AwareDatetime,
    Field,
    GetCoreSchemaHandler,
    JsonValue,
    StrictInt,
    model_validator,
)
from pydantic_core import CoreSchema, core_schema

from synth_ai.types.base import ResponseBase, ResponseIdentifier
from synth_ai.types.unknown import UnknownValue

PROBLEMS_SCHEMA = "synth.problems.v1"
PROBLEMS_SCHEMA_V2 = "synth.problems.v2"
#: Request header a client sends to receive v2 origins.
ACCEPT_PROBLEMS_HEADER = "X-Synth-Accept-Problems"

KnownProblemCode = Literal[
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
ProblemCode = KnownProblemCode | UnknownValue
ProblemSeverity = Literal["info", "warning", "error", "critical"]
ProblemState = Literal["open", "resolved"]
V1_ORIGIN_PATTERN = (
    r"^(provider:[A-Za-z0-9_.-]{1,64}|gateway|agents|orchestra|sublinear|"
    r"backend|daytona|tunnel|database|host|unknown)$"
)
# `provider:<name>`, a closed service name, or `unknown` (which carries
# `origin_tried` in its chain link's cause). `docker` is v2 only.
KNOWN_ORIGIN_PATTERN = (
    r"^(provider:[A-Za-z0-9_.-]{1,64}|gateway|agents|orchestra|sublinear|"
    r"backend|daytona|docker|tunnel|database|host|unknown)$"
)
KnownOrigin = Annotated[str, Field(strict=True, max_length=96, pattern=KNOWN_ORIGIN_PATTERN)]


class UnknownOrigin(UnknownValue):
    """A future origin token this SDK does not interpret (bounded, never free text)."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(
                strict=True,
                min_length=1,
                max_length=96,
                pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$",
            ),
        )


Origin = KnownOrigin | UnknownOrigin
Text = Annotated[str, Field(strict=True, min_length=1, max_length=1024)]


class EvidenceRef(ResponseBase):
    service: Annotated[str, Field(strict=True, min_length=1, max_length=96)]
    kind: Annotated[str, Field(strict=True, min_length=1, max_length=64)]
    ref: Annotated[str, Field(strict=True, min_length=1, max_length=256)]


class ReasonLink(ResponseBase):
    service: Origin
    code: Annotated[str, Field(strict=True, min_length=1, max_length=160)]
    stage: Annotated[str, Field(strict=True, min_length=1, max_length=96)]
    # Typed detail. An object by construction; a null cause is a defect.
    cause: dict[str, JsonValue]


class Problem(ResponseBase):
    id: ResponseIdentifier
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
