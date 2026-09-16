"""Bounded fast-search intent and exact-reference contents contracts.

See sibling docs/drafts/synth-index-api-design-2026-09-12.md and pricing-economics.
Private selection is explicit and never an access grant. Token limits are checked
again by the pinned encoder; UTF-8 byte bounds here do not claim token equivalence.
"""

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


class SearchSpec(IndexContract):
    query: Annotated[str, Field(min_length=1, max_length=8192)]
    mode: Literal["fast"] = "fast"
    scope: SearchScope = Field(default_factory=SearchScope)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    content: SearchContent = Field(default_factory=SearchContent)

    @field_validator("query")
    @classmethod
    def check_query_bytes(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 8192 or "\x00" in value:
            raise ValueError("query must fit 8192 UTF-8 bytes and contain no NUL")
        return value


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


class SearchUsage(IndexContract):
    """Server-authored successful logical search receipt; never contributor earnings."""

    billing_scope: Literal["public", "private"]
    logical_units: Annotated[StrictInt, Field(ge=1, le=1)] = 1
    price_version: Literal["synth.index.fast.v1"] = "synth.index.fast.v1"
    amount_cents: Annotated[StrictInt, Field(ge=0, le=5)]
    receipt_id: Identifier

    @model_validator(mode="after")
    def check_published_rate(self) -> Self:
        expected = 0 if self.billing_scope == "public" else 5
        if self.amount_cents != expected:
            raise ValueError("successful public search is free; private search is five cents")
        return self


class SearchResult(IndexContract):
    search_id: Identifier
    request_id: Identifier
    mode: Literal["fast"] = "fast"
    status: Literal["completed"] = "completed"
    corpus_generation: Identifier
    ranker_version: Identifier
    parser_version: Identifier
    taxonomy_version: Identifier
    results: tuple[SearchHit, ...] = Field(max_length=10)
    usage: SearchUsage

    @property
    def abstained(self) -> bool:
        """No supported answer: a real outcome, not an error or an outage.

        The backend marks an empty result list ``X-Index-Result-State:
        abstained``; failures always arrive as typed errors instead.
        """
        return not self.results

    @model_validator(mode="after")
    def check_grouped_references(self) -> Self:
        require_unique(
            tuple(hit.reference.contribution_id for hit in self.results),
            "result contributions",
        )
        require_unique(tuple(hit.id for hit in self.results), "hit IDs")
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

    @property
    def abstained(self) -> bool:
        """No supported answer: a real outcome, not an error or an outage.

        The backend marks an empty result list ``X-Index-Result-State:
        abstained``; failures always arrive as typed errors instead.
        """
        return not self.results

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
