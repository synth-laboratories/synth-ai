"""Search intent and canonical results. See backend SDK index_contract.md."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    StrictInt,
    field_validator,
    model_serializer,
    model_validator,
)

from synth_ai.errors import OperationFailedError
from synth_ai.types.base import Identifier, Model, ResponseModel
from synth_ai.types.money import Money
from synth_ai.types.usage import UsageSummary

from .annotations import ContributionAnnotations
from .contents import ContentRef
from .retention import RetentionMode, SearchRetention, validate_retention_observation

Mode = Literal["fast", "deep"]


class SearchFilters(Model):
    tags_any: tuple[Identifier, ...] = Field(default=(), max_length=10)
    tags_all: tuple[Identifier, ...] = Field(default=(), max_length=10)
    tags_none: tuple[Identifier, ...] = Field(default=(), max_length=10)

    @model_validator(mode="after")
    def validate_filters(self) -> Self:
        for values in (self.tags_any, self.tags_all, self.tags_none):
            if len(values) != len(set(values)):
                raise ValueError("duplicate filter values")
        if set(self.tags_all) & set(self.tags_none):
            raise ValueError("required and excluded tags overlap")
        return self


class PublicSearchOptions(Model):
    mode: Mode = "fast"
    max_results: StrictInt = Field(default=5, ge=1, le=10)
    filters: SearchFilters = Field(default_factory=SearchFilters)


class KeyedSearchOptions(PublicSearchOptions):
    maximum_charge: Money
    collection_ids: tuple[Identifier, ...] = Field(default=(), max_length=16)
    # None omits the field: the organization's default mode applies server-side.
    retention: RetentionMode | None = None
    deadline_seconds: StrictInt | None = Field(default=None, ge=1, le=600)

    @model_serializer(mode="wrap")
    def omit_absent_retention(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data: dict[str, object] = handler(self)
        if data.get("retention") is None:
            data.pop("retention", None)
        if data.get("deadline_seconds") is None:
            data.pop("deadline_seconds", None)
        return data

    @model_validator(mode="after")
    def validate_charge_and_collections(self) -> Self:
        if self.deadline_seconds is not None and self.mode != "deep":
            raise ValueError("deadline_seconds is supported only for deep search")
        if self.maximum_charge.coefficient < 0:
            raise ValueError("maximum charge must be nonnegative")
        if len(self.collection_ids) != len(set(self.collection_ids)):
            raise ValueError("duplicate collections")
        return self


Query = Annotated[str, Field(strict=True, min_length=1, max_length=8192)]


def validate_query_text(value: str) -> str:
    if not value.strip() or "\x00" in value or len(value.encode("utf-8")) > 8192:
        raise ValueError("query must be nonblank, contain no NUL and fit 8192 UTF-8 bytes")
    return value


class PublicSearchRequest(Model):
    query: Query
    options: PublicSearchOptions

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        return validate_query_text(value)


class KeyedSearchRequest(PublicSearchRequest):
    options: KeyedSearchOptions


class SearchCitation(Model):
    reference: ContentRef
    label: str = Field(min_length=1, max_length=256)
    annotations: ContributionAnnotations | None = None

    @model_validator(mode="after")
    def validate_annotations(self) -> Self:
        if self.annotations is not None and (
            self.annotations.contribution_id != self.reference.contribution_id
            or self.annotations.revision_id != self.reference.revision_id
        ):
            raise ValueError("citation annotations belong to another contribution revision")
        return self


class SearchFailure(Model):
    code: Identifier
    retryable: bool


class SearchResponse(ResponseModel):
    search_id: Identifier | None
    mode: Mode
    status: Literal["completed", "partial", "failed", "cancelled"]
    response: str | None = Field(min_length=1, max_length=16384)
    citations: tuple[SearchCitation, ...] = Field(max_length=10)
    warnings: tuple[Annotated[str, Field(max_length=512)], ...] = Field(max_length=16)
    completeness: Literal["complete", "partial", "unknown"]
    failure: SearchFailure | None
    usage: UsageSummary
    output_ready: bool
    accounting_final: bool
    cleanup_complete: bool
    retention: SearchRetention | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        validate_retention_observation(self.retention, self.metadata.observed_at)
        if self.retention is not None and (self.response is not None or self.citations):
            if self.retention.content_unreadable:
                raise ValueError("unavailable retained content cannot be disclosed")
            expiry = self.retention.content_expires_at or self.retention.expires_at
            if expiry is not None and expiry <= self.metadata.observed_at:
                raise ValueError("expired content cannot be disclosed")
        if self.status == "failed" and self.failure is None:
            raise ValueError("failed result requires a failure")
        if self.status != "failed" and self.failure is not None:
            raise ValueError("failure only belongs to failed result")
        if self.status in {"completed", "partial"} and self.response is None:
            raise ValueError("delivered result requires response text")
        if self.citations and not self.output_ready:
            raise ValueError("undelivered result cannot expose citations")
        if self.accounting_final != (self.usage.financial_state == "final"):
            raise ValueError("result accounting finality must match its receipt")
        labels = tuple(citation.label for citation in self.citations)
        if len(set(labels)) != len(labels):
            raise ValueError("citation labels must be unique")
        return self

    def raise_for_failure(self) -> None:
        """Opt into raising; the original result remains inspectable."""
        if self.status in {"failed", "cancelled"}:
            raise OperationFailedError(
                "Search did not complete successfully",
                code=self.failure.code if self.failure else "search_cancelled",
                request_id=self.metadata.request_id,
            )


class SearchSnapshot(ResponseModel):
    search_id: Identifier
    mode: Mode
    state: Literal["queued", "running", "completed", "partial", "failed", "cancelled"]
    result_available: bool
    cancellation_requested: bool
    execution_closed: bool
    accounting_final: bool
    cleanup_complete: bool
    retention: SearchRetention | None = None
    # Present only when the create response carries a FAST zdr_strict result.
    inline_result: SearchResponse | None = None

    @model_validator(mode="after")
    def validate_retention(self) -> Self:
        validate_retention_observation(self.retention, self.metadata.observed_at)
        if (
            self.result_available
            and self.retention is not None
            and self.retention.content_unreadable
        ):
            raise ValueError("unavailable content cannot advertise a readable result")
        return self


class SearchEvent(Model):
    event_id: Identifier
    search_id: Identifier
    sequence: StrictInt = Field(ge=1)
    kind: Literal[
        "created",
        "running",
        "completed",
        "partial",
        "failed",
        "cancelled",
        "usage_updated",
        "unknown",
    ]
    original_kind: str | None = Field(default=None, max_length=128)
    usage: UsageSummary | None = None

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if (self.kind == "unknown") != (self.original_kind is not None):
            raise ValueError("unknown event requires its original discriminant")
        if (self.kind == "usage_updated") != (self.usage is not None):
            raise ValueError("only usage_updated carries usage")
        return self
