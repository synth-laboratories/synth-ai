"""Typed clients for notebook discovery and the Human Artifactory (`SYN-6067`).

A Human Artifactory record is one typed, revisioned decision input for a notebook
or run: a preference (how to rank outcomes), a hypothesis (what to test) or a
finding (what was learned). Each write appends an immutable revision.

`origin=human` (or `author_kind=human`), and any revision of an `origin=human`
record, needs a Console session or a personal API key. Automation keys, legacy
keys and MCP OAuth grants are refused with
`ArtifactoryOriginRequiresHumanCredentialError` (403).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._generated.models import (
    ArtifactoryFinding,
    ArtifactoryHypothesis,
    ArtifactoryPreference,
    ArtifactoryPreferenceAggregationMode,
    ArtifactoryRecord,
    ArtifactoryRecordCategory,
    ArtifactoryRecordList,
    ArtifactoryRecordOrigin,
    ArtifactoryReference,
    ArtifactoryRevision,
    ArtifactoryRevisionAuthorKind,
    NotebookPage,
    NotebookSummary,
)
from ._pagination import paginate
from .errors import ArtifactoryPayloadError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator

    from pydantic import JsonValue

    from ._blocking import Runner
    from ._transport import Transport

ArtifactoryCategory = ArtifactoryRecordCategory
ArtifactoryOrigin = ArtifactoryRecordOrigin
ArtifactoryAuthorKind = ArtifactoryRevisionAuthorKind
AggregationMode = ArtifactoryPreferenceAggregationMode
ArtifactoryPayload = ArtifactoryPreference | ArtifactoryHypothesis | ArtifactoryFinding

__all__ = [
    "AggregationMode",
    "ArtifactoryAuthorKind",
    "ArtifactoryCategory",
    "ArtifactoryClient",
    "ArtifactoryFinding",
    "ArtifactoryHypothesis",
    "ArtifactoryOrigin",
    "ArtifactoryPayload",
    "ArtifactoryPreference",
    "ArtifactoryRecord",
    "ArtifactoryRecordList",
    "ArtifactoryReference",
    "ArtifactoryRevision",
    "AsyncArtifactory",
    "AsyncNotebooks",
    "NotebookPage",
    "NotebookSummary",
    "NotebooksClient",
    "payload_category",
]


def payload_category(payload: ArtifactoryPayload) -> ArtifactoryCategory:
    """Return the record category that a typed payload model belongs to."""
    if isinstance(payload, ArtifactoryPreference):
        return ArtifactoryCategory.PREFERENCE
    if isinstance(payload, ArtifactoryHypothesis):
        return ArtifactoryCategory.HYPOTHESIS
    if isinstance(payload, ArtifactoryFinding):
        return ArtifactoryCategory.FINDING
    raise ArtifactoryPayloadError(
        "payload must be ArtifactoryPreference, ArtifactoryHypothesis or ArtifactoryFinding"
    )


def _payload_field(payload: ArtifactoryPayload) -> dict[str, JsonValue]:
    return {payload_category(payload).value: payload.model_dump(mode="json", exclude_unset=True)}


class AsyncNotebooks:
    """Async client for discovering the notebooks in the caller's organization."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def list_notebooks(self, *, limit: int = 20, cursor: str | None = None) -> NotebookPage:
        """Read one page of the organization's notebooks, newest activity first."""
        return await self._transport.request(
            "listNotebooks", NotebookPage, query={"limit": limit, "cursor": cursor}
        )

    async def iter_notebooks(
        self, *, limit: int = 20, cursor: str | None = None
    ) -> AsyncGenerator[NotebookSummary, None]:
        """Yield every notebook in the organization across cursor pages."""
        async for item in paginate(
            lambda next_cursor: self.list_notebooks(limit=limit, cursor=next_cursor),
            cursor=cursor,
        ):
            yield item


class AsyncArtifactory:
    """Async client for typed Human Artifactory records."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def create_record(
        self,
        notebook_id: str,
        title: str,
        payload: ArtifactoryPayload,
        *,
        origin: ArtifactoryOrigin | str = ArtifactoryOrigin.AGENT,
        run_id: str | None = None,
        author_kind: ArtifactoryAuthorKind | str | None = None,
        pinned: bool | None = None,
        summary: str | None = None,
        idempotency_key: str | None = None,
    ) -> ArtifactoryRecord:
        """Create a record; the category follows the payload type (preference, hypothesis, finding)."""
        body: dict[str, JsonValue] = {
            "title": title,
            "category": payload_category(payload).value,
            "origin": ArtifactoryOrigin(origin).value,
            **_payload_field(payload),
        }
        if run_id is not None:
            body["run_id"] = run_id
        if author_kind is not None:
            body["author_kind"] = ArtifactoryAuthorKind(author_kind).value
        if pinned is not None:
            body["pinned"] = pinned
        if summary is not None:
            body["summary"] = summary
        return await self._transport.request(
            "createArtifactoryRecord",
            ArtifactoryRecord,
            path={"notebook_id": notebook_id},
            body=body,
            idempotency_key=idempotency_key,
        )

    async def list_records(
        self,
        notebook_id: str,
        *,
        category: ArtifactoryCategory | str | None = None,
        origin: ArtifactoryOrigin | str | None = None,
        run_id: str | None = None,
        pinned_only: bool = False,
    ) -> ArtifactoryRecordList:
        """List a notebook's records, optionally filtered by category, origin, run or pinned."""
        return await self._transport.request(
            "listArtifactoryRecords",
            ArtifactoryRecordList,
            path={"notebook_id": notebook_id},
            query={
                "category": ArtifactoryCategory(category).value if category is not None else None,
                "origin": ArtifactoryOrigin(origin).value if origin is not None else None,
                "run_id": run_id,
                "pinned": "true" if pinned_only else None,
            },
        )

    async def resolve_preference(
        self, notebook_id: str, *, run_id: str | None = None
    ) -> ArtifactoryRecord:
        """Resolve the preference that governs a notebook or run; human-pinned ones win."""
        return await self._transport.request(
            "resolveEffectivePreference",
            ArtifactoryRecord,
            path={"notebook_id": notebook_id},
            query={"run_id": run_id},
        )

    async def get_record(self, record_id: str) -> ArtifactoryRecord:
        """Read a record with its full revision history."""
        return await self._transport.request(
            "getArtifactoryRecord", ArtifactoryRecord, path={"record_id": record_id}
        )

    async def revise_record(
        self,
        record_id: str,
        *,
        payload: ArtifactoryPayload | None = None,
        title: str | None = None,
        summary: str | None = None,
        pinned: bool | None = None,
        author_kind: ArtifactoryAuthorKind | str | None = None,
        expected_revision: int | None = None,
        expected_supersedes_digest: str | None = None,
        idempotency_key: str | None = None,
    ) -> ArtifactoryRecord:
        """Append revision r+1; pass expected_revision to fence out concurrent writers."""
        body: dict[str, JsonValue] = {} if payload is None else _payload_field(payload)
        if title is not None:
            body["title"] = title
        if summary is not None:
            body["summary"] = summary
        if pinned is not None:
            body["pinned"] = pinned
        if author_kind is not None:
            body["author_kind"] = ArtifactoryAuthorKind(author_kind).value
        if expected_revision is not None:
            body["expected_revision"] = expected_revision
        if expected_supersedes_digest is not None:
            body["expected_supersedes_digest"] = expected_supersedes_digest
        return await self._transport.request(
            "reviseArtifactoryRecord",
            ArtifactoryRecord,
            path={"record_id": record_id},
            body=body,
            idempotency_key=idempotency_key,
        )


class NotebooksClient:
    """Synchronous delegate for discovering the notebooks in the caller's organization."""

    def __init__(self, runner: Runner, inner: AsyncNotebooks) -> None:
        self._runner = runner
        self._inner = inner

    def list_notebooks(self, *, limit: int = 20, cursor: str | None = None) -> NotebookPage:
        """Read one page of the organization's notebooks, newest activity first."""
        return self._runner.call(self._inner.list_notebooks(limit=limit, cursor=cursor))

    def iter_notebooks(
        self, *, limit: int = 20, cursor: str | None = None
    ) -> Iterator[NotebookSummary]:
        """Yield every notebook in the organization across cursor pages."""
        return self._runner.iterate(self._inner.iter_notebooks(limit=limit, cursor=cursor))


class ArtifactoryClient:
    """Synchronous delegate for typed Human Artifactory records."""

    def __init__(self, runner: Runner, inner: AsyncArtifactory) -> None:
        self._runner = runner
        self._inner = inner

    def create_record(
        self,
        notebook_id: str,
        title: str,
        payload: ArtifactoryPayload,
        *,
        origin: ArtifactoryOrigin | str = ArtifactoryOrigin.AGENT,
        run_id: str | None = None,
        author_kind: ArtifactoryAuthorKind | str | None = None,
        pinned: bool | None = None,
        summary: str | None = None,
        idempotency_key: str | None = None,
    ) -> ArtifactoryRecord:
        """Create a record; the category follows the payload type (preference, hypothesis, finding)."""
        return self._runner.call(
            self._inner.create_record(
                notebook_id,
                title,
                payload,
                origin=origin,
                run_id=run_id,
                author_kind=author_kind,
                pinned=pinned,
                summary=summary,
                idempotency_key=idempotency_key,
            )
        )

    def list_records(
        self,
        notebook_id: str,
        *,
        category: ArtifactoryCategory | str | None = None,
        origin: ArtifactoryOrigin | str | None = None,
        run_id: str | None = None,
        pinned_only: bool = False,
    ) -> ArtifactoryRecordList:
        """List a notebook's records, optionally filtered by category, origin, run or pinned."""
        return self._runner.call(
            self._inner.list_records(
                notebook_id,
                category=category,
                origin=origin,
                run_id=run_id,
                pinned_only=pinned_only,
            )
        )

    def resolve_preference(
        self, notebook_id: str, *, run_id: str | None = None
    ) -> ArtifactoryRecord:
        """Resolve the preference that governs a notebook or run; human-pinned ones win."""
        return self._runner.call(self._inner.resolve_preference(notebook_id, run_id=run_id))

    def get_record(self, record_id: str) -> ArtifactoryRecord:
        """Read a record with its full revision history."""
        return self._runner.call(self._inner.get_record(record_id))

    def revise_record(
        self,
        record_id: str,
        *,
        payload: ArtifactoryPayload | None = None,
        title: str | None = None,
        summary: str | None = None,
        pinned: bool | None = None,
        author_kind: ArtifactoryAuthorKind | str | None = None,
        expected_revision: int | None = None,
        expected_supersedes_digest: str | None = None,
        idempotency_key: str | None = None,
    ) -> ArtifactoryRecord:
        """Append revision r+1; pass expected_revision to fence out concurrent writers."""
        return self._runner.call(
            self._inner.revise_record(
                record_id,
                payload=payload,
                title=title,
                summary=summary,
                pinned=pinned,
                author_kind=author_kind,
                expected_revision=expected_revision,
                expected_supersedes_digest=expected_supersedes_digest,
                idempotency_key=idempotency_key,
            )
        )
