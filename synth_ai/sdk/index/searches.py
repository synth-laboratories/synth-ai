"""Keyed create/reattach/list/reconcile. See SDK index_contract.md."""

from __future__ import annotations

from synth_ai.contracts import operations
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.parameters import identifier, page_parameters, verified
from synth_ai.core.recovery import async_resolve_creation as recover_creation_async
from synth_ai.core.recovery import resolve_creation as recover_creation
from synth_ai.options import MutationOptions, ReadOptions
from synth_ai.types.base import Page, PageCursor
from synth_ai.types.identity import MutationIdentity
from synth_ai.types.receipts import CreationResolution

from .search_run import AsyncSearchRun, SearchRun
from .types.search import KeyedSearchOptions, KeyedSearchRequest, SearchSnapshot


class Searches:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def create(
        self, query: str, *, options: KeyedSearchOptions, mutation: MutationOptions | None = None
    ) -> SearchRun:
        """Create one durable owner resource; uncertain responses preserve mutation identity.

        See: backend/notes/specifications/sdk/index_contract.md."""
        snapshot = self._transport.request(
            operations.CREATE_SEARCH,
            SearchSnapshot,
            body=KeyedSearchRequest(query=query, options=options),
            options=mutation,
            validate=lambda received: verified(
                received,
                received.mode == options.mode,
                "Search admission changed the requested mode",
            ),
        )
        return SearchRun(self._transport, snapshot.search_id)

    def get(self, search_id: str, *, request: ReadOptions | None = None) -> SearchRun:
        """Read or reattach the exact authorized resource with fresh server state.

        See: backend/notes/specifications/sdk/index_contract.md."""
        search_id = identifier(search_id)
        self._transport.request(
            operations.GET_SEARCH,
            SearchSnapshot,
            path=f"/index/searches/{search_id}",
            options=request,
            validate=lambda received: verified(
                received,
                received.search_id == search_id,
                "Reattachment returned a different search",
            ),
        )
        return SearchRun(self._transport, search_id)

    def list(
        self, *, limit: int, cursor: PageCursor | None = None, request: ReadOptions | None = None
    ) -> Page[SearchSnapshot]:
        """Read one bounded authorized page; no implicit all-pages traversal.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = self._transport.request(
            operations.LIST_SEARCHES,
            Page[SearchSnapshot],
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda received: verified(
                received,
                not len(received.items) > limit,
                "Search page exceeded its requested limit",
            ),
        )
        return result

    def resolve_creation(
        self, identity: MutationIdentity, *, request: ReadOptions | None = None
    ) -> CreationResolution[SearchRun]:
        return recover_creation(
            self._transport,
            identity,
            operations.CREATE_SEARCH,
            "search",
            lambda resource_id: SearchRun(self._transport, resource_id),
            request,
        )


class AsyncSearches:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def create(
        self, query: str, *, options: KeyedSearchOptions, mutation: MutationOptions | None = None
    ) -> AsyncSearchRun:
        """Create one durable owner resource; uncertain responses preserve mutation identity.

        See: backend/notes/specifications/sdk/index_contract.md."""
        snapshot = await self._transport.request(
            operations.CREATE_SEARCH,
            SearchSnapshot,
            body=KeyedSearchRequest(query=query, options=options),
            options=mutation,
            validate=lambda received: verified(
                received,
                received.mode == options.mode,
                "Search admission changed the requested mode",
            ),
        )
        return AsyncSearchRun(self._transport, snapshot.search_id)

    async def get(self, search_id: str, *, request: ReadOptions | None = None) -> AsyncSearchRun:
        """Read or reattach the exact authorized resource with fresh server state.

        See: backend/notes/specifications/sdk/index_contract.md."""
        search_id = identifier(search_id)
        await self._transport.request(
            operations.GET_SEARCH,
            SearchSnapshot,
            path=f"/index/searches/{search_id}",
            options=request,
            validate=lambda received: verified(
                received,
                received.search_id == search_id,
                "Reattachment returned a different search",
            ),
        )
        return AsyncSearchRun(self._transport, search_id)

    async def list(
        self, *, limit: int, cursor: PageCursor | None = None, request: ReadOptions | None = None
    ) -> Page[SearchSnapshot]:
        """Read one bounded authorized page; no implicit all-pages traversal.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = await self._transport.request(
            operations.LIST_SEARCHES,
            Page[SearchSnapshot],
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda received: verified(
                received,
                not len(received.items) > limit,
                "Search page exceeded its requested limit",
            ),
        )
        return result

    async def resolve_creation(
        self, identity: MutationIdentity, *, request: ReadOptions | None = None
    ) -> CreationResolution[AsyncSearchRun]:
        async def attach(resource_id: str) -> AsyncSearchRun:
            return AsyncSearchRun(self._transport, resource_id)

        return await recover_creation_async(
            self._transport, identity, operations.CREATE_SEARCH, "search", attach, request
        )
