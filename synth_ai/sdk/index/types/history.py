"""Bounded private history and deletion receipts. See backend SDK index_contract.md."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import AwareDatetime, Field, StrictInt, field_validator, model_validator

from synth_ai.types.base import Cursor, Digest, Identifier, Model, Page, ResponseModel

from .retention import SearchRetention
from .search import Mode, Query, validate_query_text


class SearchHistoryFilter(Model):
    query: str = Field(default="", max_length=200, repr=False)
    created_after: AwareDatetime | None = None
    created_before: AwareDatetime | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if "\x00" in value or len(value.encode("utf-8")) > 800:
            raise ValueError("history query must contain no NUL and fit 800 UTF-8 bytes")
        return value

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if (
            self.created_after is not None
            and self.created_before is not None
            and self.created_after >= self.created_before
        ):
            raise ValueError("history window must have increasing timestamps")
        return self


class SearchHistoryRequest(Model):
    limit: StrictInt = Field(ge=1, le=200)
    cursor: Cursor | None = None
    filter: SearchHistoryFilter = Field(default_factory=SearchHistoryFilter)


class SearchHistoryEntry(Model):
    search_id: Identifier
    query: Query = Field(repr=False)
    mode: Mode
    created_at: AwareDatetime
    archive_digest_sha256: Digest
    retention: SearchRetention

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        return validate_query_text(value)

    @model_validator(mode="after")
    def validate_history(self) -> Self:
        if self.retention.mode != "retained" or not self.retention.history_available:
            raise ValueError("history entries require available customer-retained content")
        if self.retention.archived_at is None or self.created_at > self.retention.archived_at:
            raise ValueError("history archival must follow search creation")
        return self


class SearchHistoryPage(Page[SearchHistoryEntry]):
    filter: SearchHistoryFilter

    @model_validator(mode="after")
    def validate_page(self) -> Self:
        if self.complete != (self.next_cursor is None):
            raise ValueError("history completeness must match its continuation cursor")
        if not self.items and self.next_cursor is not None:
            raise ValueError("empty history page cannot advertise a continuation")
        identifiers = tuple(item.search_id for item in self.items)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("history page cannot duplicate search identities")
        for item in self.items:
            expires_at = item.retention.content_expires_at
            archived_at = item.retention.archived_at
            if expires_at is None or expires_at <= self.metadata.observed_at:
                raise ValueError("history page cannot advertise expired content")
            if archived_at is None or archived_at > self.metadata.observed_at:
                raise ValueError("history page cannot advertise future archival")
            if (
                self.filter.created_after is not None
                and item.created_at <= self.filter.created_after
            ):
                raise ValueError("history entry falls before its selected window")
            if (
                self.filter.created_before is not None
                and item.created_at >= self.filter.created_before
            ):
                raise ValueError("history entry falls after its selected window")
            if self.filter.query.casefold() not in item.query.casefold():
                raise ValueError("history entry does not match its selected query")
        return self


class SearchHistoryDeletion(Model):
    expected_archive_digest_sha256: Digest


class SearchHistoryDeletionReceipt(ResponseModel):
    search_id: Identifier
    archive_digest_sha256: Digest
    state: Literal["deleted", "already_deleted"]
    history_available: Literal[False]
    cleanup_complete: bool
