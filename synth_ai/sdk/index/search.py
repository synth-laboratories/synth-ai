"""Shared fast/deep Search intent, lifecycle and exact-reference contracts.

See sibling docs/drafts/synth-index-api-design-2026-09-12.md and pricing-economics.
Private selection is explicit and never an access grant. Token limits are checked
again by the pinned encoder; UTF-8 byte bounds here do not claim token equivalence.
"""

import re
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    AnyHttpUrl,
    AwareDatetime,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)
from pydantic.types import JsonValue

from .artifacts import ArtifactDigest
from .contracts import (
    ContributionKind,
    ContributionReference,
    Identifier,
    IndexContract,
    ResearchArea,
    WorkflowStage,
    require_unique,
)

EMPTY_SEARCH_RESPONSE = "No matching evidence was found."
INLINE_CITATION = re.compile(r"\[([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\]")
MAX_SEARCH_CITATIONS = 10


class SearchScope(IndexContract):
    visibility: Literal["public", "private"] = "public"
    collection_ids: tuple[Identifier, ...] = Field(default=(), max_length=16)


class SearchFilters(IndexContract):
    kinds: tuple[ContributionKind, ...] = Field(default=(), max_length=7)
    research_areas: tuple[ResearchArea, ...] = Field(default=(), max_length=11)
    workflow_stages: tuple[WorkflowStage, ...] = Field(default=(), max_length=6)
    tags_any: tuple[Identifier, ...] = Field(default=(), max_length=10)
    tags_all: tuple[Identifier, ...] = Field(default=(), max_length=10)

    @model_validator(mode="after")
    def check_unique_filters(self) -> Self:
        for name in type(self).model_fields:
            require_unique(getattr(self, name), name)
        return self


class SearchContent(IndexContract):
    max_results: Annotated[StrictInt, Field(ge=1, le=10)] = 5
    max_excerpts_per_result: Annotated[StrictInt, Field(ge=0, le=2)] = 2
    relative_score_floor: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            allow_inf_nan=False,
            description=(
                "After ColBERT, drop hits below this fraction of the top MaxSim. "
                "Omit for 0.85; 0 fills max_results."
            ),
        ),
    ] = 0.85

    @field_validator("relative_score_floor", mode="before")
    @classmethod
    def reject_bool_relative_score_floor(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("relative_score_floor must be a number")
        return value


class SearchMode(StrEnum):
    FAST = "fast"
    DEEP = "deep"


class SearchExecutionLimits(IndexContract):
    """Backend-authored Deep ceilings; POST /search waits at most 90 seconds.

    Durable ``searches.create`` and ``searches.wait`` honor the full execution
    deadline. See backend's synth-index/fast-deep-search-contract.md.
    """

    deadline_seconds: Annotated[StrictInt, Field(ge=1, le=300)] = 180
    max_model_turns: Annotated[StrictInt, Field(ge=1, le=24)] = 16
    max_searches: Annotated[StrictInt, Field(ge=1, le=48)] = 32
    max_reads: Annotated[StrictInt, Field(ge=1, le=96)] = 64
    max_active_context_tokens: Annotated[StrictInt, Field(ge=1024, le=32_768)] = 32_768
    max_inference_tokens: Annotated[StrictInt, Field(ge=1)] | None = None
    max_cost_usd_micros: Annotated[StrictInt, Field(ge=1)] | None = None


class SearchBillingConstraints(IndexContract):
    """Per-request retail ceiling; organization wallet consent is also required."""

    allow_wallet: bool = False
    max_charge_cents: Annotated[StrictInt, Field(ge=0, le=1_000_000)] | None = None


class SearchSpec(IndexContract):
    query: Annotated[str, Field(min_length=1, max_length=8192)]
    mode: SearchMode = SearchMode.FAST
    scope: SearchScope = Field(default_factory=SearchScope)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    content: SearchContent = Field(default_factory=SearchContent)
    limits: SearchExecutionLimits | None = None
    billing: SearchBillingConstraints = Field(default_factory=SearchBillingConstraints)

    @field_validator("query")
    @classmethod
    def check_query_bytes(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 8192 or "\x00" in value:
            raise ValueError("query must fit 8192 UTF-8 bytes and contain no NUL")
        return value

    @model_validator(mode="after")
    def check_mode_options(self) -> Self:
        if self.mode == SearchMode.FAST and self.limits is not None:
            raise ValueError("execution limits are supported only for deep search")
        return self


class ContentsSpec(IndexContract):
    references: tuple[ContributionReference, ...] = Field(min_length=1, max_length=10)
    search_id: Identifier | None = None
    max_bytes: Annotated[StrictInt, Field(ge=1, le=65_536)] = 65_536

    @model_validator(mode="after")
    def check_unique_references(self) -> Self:
        require_unique(
            tuple((item.contribution_id, item.revision_id) for item in self.references),
            "references",
        )
        return self


class SearchExcerpt(IndexContract):
    """One citation: an exact byte span of one asset of one sealed revision.

    ``asset_digest_sha256`` names the bytes the span belongs to, bound by the
    backend from the revision's sealed descriptor. With the revision on the
    enclosing hit and the parser version on the enclosing result, a citation can
    be resolved to exact bytes or re-verified later. It is optional here only
    because the same shape carries retrieval evidence inside the backend.
    """

    model_config = ConfigDict(str_strip_whitespace=False)
    asset_id: Identifier
    chunk_id: Identifier
    text: Annotated[str, Field(min_length=1, max_length=2048)]
    start_byte: Annotated[StrictInt, Field(ge=0)]
    end_byte: Annotated[StrictInt, Field(gt=0)]
    asset_digest_sha256: ArtifactDigest | None = None

    @model_validator(mode="after")
    def check_source_span(self) -> Self:
        size = len(self.text.encode("utf-8"))
        if size > 2048 or self.end_byte - self.start_byte != size:
            raise ValueError("excerpt must be an exact source span of at most 2 KiB")
        return self


class SearchHit(IndexContract):
    id: Identifier
    reference: ContributionReference
    title: Annotated[str, Field(min_length=1, max_length=200)]
    url: AnyHttpUrl
    author_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=32)
    published_at: AwareDatetime
    highlights: tuple[SearchExcerpt, ...] = Field(default=(), max_length=2)
    limitations: Annotated[str, Field(min_length=1, max_length=2048)]
    assessment_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def check_citations_are_bound(self) -> Self:
        if any(excerpt.asset_digest_sha256 is None for excerpt in self.highlights):
            raise ValueError("delivered excerpts must cite an exact asset digest")
        return self


class SearchSettlementOutcome(StrEnum):
    COMPLETE = "complete"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    USABLE_PARTIAL = "usable_partial"
    INFRASTRUCTURE_STOPPED = "infrastructure_stopped"
    NO_DELIVERABLE = "no_deliverable"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    CANCELLED = "cancelled"
    ACCESS_REVOKED = "access_revoked"


class SearchUsage(IndexContract):
    """Server-authored successful logical search receipt; never contributor earnings."""

    mode: SearchMode = SearchMode.FAST
    billing_scope: Literal["public", "private"]
    logical_units: Annotated[StrictInt, Field(ge=1, le=1)] = 1
    price_version: Identifier | None = "synth.index.fast.v2"
    amount_cents: Annotated[StrictInt, Field(ge=0)]
    receipt_id: Identifier
    search_calls: Annotated[StrictInt, Field(ge=1)] = 1
    read_calls: Annotated[StrictInt, Field(ge=0)] = 0
    input_tokens: Annotated[StrictInt, Field(ge=0)] = 0
    output_tokens: Annotated[StrictInt, Field(ge=0)] = 0
    # Operator-only diagnostics are absent from ordinary customer responses.
    inference_cost_usd_micros: Annotated[StrictInt, Field(ge=0)] | None = None
    funding_source: Literal["none", "promo_credit", "deep_beta", "wallet"] = "none"
    settlement_outcome: SearchSettlementOutcome | None = None
    retail_amount_cents: Annotated[StrictInt, Field(ge=0)] | None = None
    allowance_units_consumed: Annotated[StrictInt, Field(ge=0, le=1)] = 0
    allowance_value_cents: Annotated[StrictInt, Field(ge=0)] = 0
    wallet_debit_cents: Annotated[StrictInt, Field(ge=0)] = 0

    @model_validator(mode="after")
    def check_published_rate(self) -> Self:
        if self.mode == SearchMode.FAST:
            # v1 is historical only; new public and private FAST both cost 5 cents.
            if self.price_version == "synth.index.fast.v2":
                expected = 5
            elif self.price_version == "synth.index.fast.v1":
                expected = 0 if self.billing_scope == "public" else 5
            else:
                expected = None
            if expected is None or self.amount_cents != expected or self.read_calls:
                raise ValueError("fast usage must use its published rate and no deep tool reads")
        elif self.price_version in {"synth.index.fast.v1", "synth.index.fast.v2"}:
            raise ValueError("deep usage cannot use the fast-search price version")
        elif self.price_version is None and self.amount_cents:
            raise ValueError("unpriced deep usage cannot report a charged amount")
        return self


class SearchExecutionVersions(IndexContract):
    """Exact implementation identities used by one result.

    An identity the execution did not use is omitted, never ``null``.
    """

    corpus_generation: Identifier
    ranker_version: Identifier
    parser_version: Identifier
    taxonomy_version: Identifier
    reranker_version: Identifier | None = None
    policy_version: Identifier | None = None
    agent_version: Identifier | None = None
    model_version: (
        Annotated[
            str,
            Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_./-]{0,255}$"),
        ]
        | None
    ) = None


class SearchPartialReason(StrEnum):
    DEADLINE_EXCEEDED = "deadline_exceeded"
    TOKEN_BUDGET_EXHAUSTED = "token_budget_exhausted"
    COST_BUDGET_EXHAUSTED = "cost_budget_exhausted"
    TOOL_LIMIT_EXHAUSTED = "tool_limit_exhausted"
    EVIDENCE_REVOKED = "evidence_revoked"


class SearchState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SearchEventKind(StrEnum):
    CREATED = "created"
    CLAIMED = "claimed"
    TOOL_COMPLETED = "tool_completed"
    CHECKPOINTED = "checkpointed"
    CANCELLATION_REQUESTED = "cancellation_requested"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SearchEvent(IndexContract):
    """Reconnectable public execution event; no private model reasoning."""

    search_id: Identifier
    sequence: Annotated[StrictInt, Field(ge=1)]
    kind: SearchEventKind
    operation_id: Identifier | None = None
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: AwareDatetime


class SearchEventPage(IndexContract):
    search_id: Identifier
    events: tuple[SearchEvent, ...] = Field(default=(), max_length=200)
    next_after: Annotated[StrictInt, Field(ge=0)]


class SearchFailure(IndexContract):
    code: Identifier
    retryable: bool


class Search(IndexContract):
    """Authorized lifecycle snapshot; a client wait timeout does not mutate it."""

    search_id: Identifier
    spec: SearchSpec
    state: SearchState
    requested_mode: SearchMode
    effective_mode: SearchMode | None = None
    cancellation_requested: bool = False
    result_available: bool = False
    failure: SearchFailure | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def check_lifecycle(self) -> Self:
        if self.requested_mode != self.spec.mode:
            raise ValueError("search snapshot mode must match its specification")
        if self.effective_mode is not None and self.effective_mode != self.requested_mode:
            raise ValueError("search snapshot cannot silently substitute a mode")
        if self.updated_at < self.created_at:
            raise ValueError("search update cannot precede creation")
        if self.state == SearchState.COMPLETED and (
            self.effective_mode is None or not self.result_available
        ):
            raise ValueError("completed search requires an available result")
        if self.state == SearchState.FAILED and self.failure is None:
            raise ValueError("failed search requires a typed failure")
        if self.state != SearchState.FAILED and self.failure is not None:
            raise ValueError("only a failed search carries a failure")
        return self


class SearchCancellation(IndexContract):
    search_id: Identifier
    state: SearchState
    cancellation_requested: Literal[True] = True


class SearchWaitTimeoutError(TimeoutError):
    """A local wait ended without cancelling or resubmitting the durable Search."""

    def __init__(self, search_id: str) -> None:
        self.search_id = search_id
        super().__init__(f"Search {search_id} is still running")


class SearchExecutionFailedError(RuntimeError):
    """The durable worker recorded a typed terminal failure."""

    def __init__(self, search_id: str, failure: SearchFailure) -> None:
        self.search_id = search_id
        self.code = failure.code
        self.retryable = failure.retryable
        super().__init__(f"Search {search_id} failed: {failure.code}")


class SearchExecutionCancelledError(RuntimeError):
    """The durable Search acknowledged cancellation before producing a result."""

    def __init__(self, search_id: str) -> None:
        self.search_id = search_id
        super().__init__(f"Search {search_id} was cancelled")


class SearchResult(IndexContract):
    """Delivered answer text and cited Contribution IDs, never ranked hits."""

    search_id: Identifier
    request_id: Identifier
    requested_mode: SearchMode
    effective_mode: SearchMode
    status: Literal["completed", "partial"]
    partial_reason: SearchPartialReason | None
    corpus_generation: Identifier
    ranker_version: Identifier
    parser_version: Identifier
    taxonomy_version: Identifier
    execution_versions: SearchExecutionVersions
    response: Annotated[str, Field(min_length=1, max_length=16_384)]
    citations: tuple[ContributionReference, ...] = Field(max_length=MAX_SEARCH_CITATIONS)
    usage: SearchUsage

    @model_validator(mode="after")
    def check_citations(self) -> Self:
        inline = tuple(dict.fromkeys(INLINE_CITATION.findall(self.response)))
        listed = tuple(item.contribution_id for item in self.citations)
        if inline != listed:
            raise ValueError(
                "inline citations must equal listed citations in first-appearance order"
            )
        if not self.citations and self.response != EMPTY_SEARCH_RESPONSE:
            raise ValueError("an uncited response must be the fixed abstention")
        if self.requested_mode != self.effective_mode:
            raise ValueError("requested and effective mode must match")
        if self.usage.mode != self.effective_mode:
            raise ValueError("usage mode must match the effective search mode")
        if (self.status == "partial") != (self.partial_reason is not None):
            raise ValueError("partial status and partial reason must agree")
        if any(
            getattr(self.execution_versions, name) != getattr(self, name)
            for name in (
                "corpus_generation",
                "ranker_version",
                "parser_version",
                "taxonomy_version",
            )
        ):
            raise ValueError("execution versions must match canonical result versions")
        if self.status == "partial" and self.effective_mode != SearchMode.DEEP:
            raise ValueError("only deep search may return a partial result")
        return self


class ContentsItem(IndexContract):
    """Delivered text bound to the exact asset bytes it was rendered from."""

    model_config = ConfigDict(str_strip_whitespace=False)
    reference: ContributionReference
    status: Literal["available", "unavailable"]
    asset_id: Identifier | None = None
    text: Annotated[str, Field(max_length=65_536)] | None = None
    asset_digest_sha256: ArtifactDigest | None = None
    logical_path: Annotated[str, Field(min_length=1, max_length=1024)] | None = None
    start_byte: Annotated[StrictInt, Field(ge=0)] | None = None
    end_byte: Annotated[StrictInt, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def check_delivery(self) -> Self:
        located = (
            self.asset_id,
            self.text,
            self.asset_digest_sha256,
            self.logical_path,
            self.start_byte,
            self.end_byte,
        )
        if self.status == "available":
            if (
                self.asset_id is None
                or self.text is None
                or self.asset_digest_sha256 is None
                or self.logical_path is None
                or self.start_byte is None
                or self.end_byte is None
            ):
                raise ValueError(
                    "available contents require an exact asset, digest, locator and text"
                )
            if self.end_byte - self.start_byte != len(self.text.encode("utf-8")):
                raise ValueError("contents span must match the delivered bytes")
        elif any(value is not None for value in located):
            raise ValueError("unavailable contents cannot disclose asset metadata or text")
        return self


class ContentsResult(IndexContract):
    request_id: Identifier
    items: tuple[ContentsItem, ...] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def check_total_bytes(self) -> Self:
        require_unique(
            tuple(
                (item.reference.contribution_id, item.reference.revision_id) for item in self.items
            ),
            "contents references",
        )
        if sum(len((item.text or "").encode("utf-8")) for item in self.items) > 65_536:
            raise ValueError("contents response exceeds 64 KiB total rendered text")
        return self
