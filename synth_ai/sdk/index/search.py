"""Bounded fast-search intent and exact-reference contents contracts.

See sibling docs/drafts/synth-index-api-design-2026-09-12.md and pricing-economics.
Private selection is explicit and never an access grant. Token limits are checked
again by the pinned encoder; UTF-8 byte bounds here do not claim token equivalence.
"""

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


class SearchMode(StrEnum):
    FAST = "fast"
    DEEP = "deep"


class SearchExecutionLimits(IndexContract):
    """Caller ceilings for deep execution; server policy may only tighten them."""

    deadline_seconds: Annotated[StrictInt, Field(ge=1, le=90)] = 90
    max_model_turns: Annotated[StrictInt, Field(ge=1, le=6)] = 6
    max_searches: Annotated[StrictInt, Field(ge=1, le=8)] = 8
    max_reads: Annotated[StrictInt, Field(ge=1, le=12)] = 12
    max_active_context_tokens: Annotated[
        StrictInt, Field(ge=1024, le=32_768)
    ] = 32_768
    max_inference_tokens: Annotated[StrictInt, Field(ge=1)] | None = None
    max_cost_usd_micros: Annotated[StrictInt, Field(ge=1)] | None = None


class SearchSpec(IndexContract):
    query: Annotated[str, Field(min_length=1, max_length=8192)]
    mode: SearchMode = SearchMode.FAST
    scope: SearchScope = Field(default_factory=SearchScope)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    content: SearchContent = Field(default_factory=SearchContent)
    limits: SearchExecutionLimits | None = None

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


class SearchUsage(IndexContract):
    """Server-authored successful logical search receipt; never contributor earnings."""

    mode: SearchMode = SearchMode.FAST
    billing_scope: Literal["public", "private"]
    logical_units: Annotated[StrictInt, Field(ge=1, le=1)] = 1
    price_version: Identifier | None = "synth.index.fast.v1"
    amount_cents: Annotated[StrictInt, Field(ge=0)]
    receipt_id: Identifier
    search_calls: Annotated[StrictInt, Field(ge=1)] = 1
    read_calls: Annotated[StrictInt, Field(ge=0)] = 0
    input_tokens: Annotated[StrictInt, Field(ge=0)] = 0
    output_tokens: Annotated[StrictInt, Field(ge=0)] = 0
    inference_cost_usd_micros: Annotated[StrictInt, Field(ge=0)] = 0

    @model_validator(mode="after")
    def check_published_rate(self) -> Self:
        if self.mode == SearchMode.FAST:
            expected = 0 if self.billing_scope == "public" else 5
            if (
                self.price_version != "synth.index.fast.v1"
                or self.amount_cents != expected
                or self.read_calls
                or self.input_tokens
                or self.output_tokens
                or self.inference_cost_usd_micros
            ):
                raise ValueError(
                    "fast usage must use its published rate and no deep inference"
                )
        elif self.price_version == "synth.index.fast.v1":
            raise ValueError("deep usage cannot use the fast-search price version")
        elif self.price_version is None and self.amount_cents:
            raise ValueError("unpriced deep usage cannot report a charged amount")
        return self


class SearchExecutionVersions(IndexContract):
    corpus_generation: Identifier
    ranker_version: Identifier
    parser_version: Identifier
    taxonomy_version: Identifier
    reranker_version: Identifier | None = None
    policy_version: Identifier | None = None
    agent_version: Identifier | None = None
    model_version: Identifier | None = None


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
    """Reconnectable execution event; never private model reasoning."""

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


class SearchResult(IndexContract):
    search_id: Identifier
    request_id: Identifier
    requested_mode: SearchMode
    effective_mode: SearchMode
    status: Literal["completed", "partial"] = "completed"
    partial_reason: SearchPartialReason | None = None
    unresolved_evidence_needs: tuple[
        Annotated[str, Field(min_length=1, max_length=512)], ...
    ] = Field(default=(), max_length=16)
    corpus_generation: Identifier
    ranker_version: Identifier
    parser_version: Identifier
    taxonomy_version: Identifier
    execution_versions: SearchExecutionVersions
    results: tuple[SearchHit, ...] = Field(max_length=10)
    usage: SearchUsage

    @model_validator(mode="after")
    def check_grouped_references(self) -> Self:
        require_unique(
            tuple(hit.reference.contribution_id for hit in self.results),
            "result contributions",
        )
        require_unique(tuple(hit.id for hit in self.results), "hit IDs")
        if self.requested_mode != self.effective_mode:
            raise ValueError("requested and effective mode must match")
        if self.usage.mode != self.effective_mode:
            raise ValueError("usage mode must match the effective search mode")
        if self.status == "completed" and (
            self.partial_reason is not None or self.unresolved_evidence_needs
        ):
            raise ValueError("completed search cannot carry partial outcome fields")
        if self.status == "partial" and self.partial_reason is None:
            raise ValueError("partial search requires a typed partial reason")
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


class PublicSearchResult(IndexContract):
    """Receipt-free anonymous search result from the public Index boundary."""

    request_id: Identifier
    mode: Literal["fast"] = "fast"
    status: Literal["completed"] = "completed"
    corpus_generation: Identifier
    ranker_version: Identifier
    parser_version: Identifier
    taxonomy_version: Identifier
    results: tuple[SearchHit, ...] = Field(max_length=10)
    amount_cents: Literal[0] = 0

    @model_validator(mode="after")
    def check_grouped_references(self) -> Self:
        require_unique(
            tuple(hit.reference.contribution_id for hit in self.results),
            "result contributions",
        )
        require_unique(tuple(hit.id for hit in self.results), "hit IDs")
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
        if self.status == "available" and any(value is None for value in located):
            raise ValueError(
                "available contents require an exact asset, digest, locator and text"
            )
        if self.status == "unavailable" and any(value is not None for value in located):
            raise ValueError("unavailable contents cannot disclose asset metadata or text")
        if self.status == "available" and self.end_byte - self.start_byte != len(
            self.text.encode("utf-8")
        ):
            raise ValueError("contents span must match the delivered bytes")
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
