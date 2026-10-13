"""Read-only setup resources. See SDK index_contract.md."""

from __future__ import annotations

from synth_ai.contracts import operations
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.parameters import identifier, page_parameters, verified
from synth_ai.options import ReadOptions
from synth_ai.types.base import Page, PageCursor

from .types.account import Collection


class Collections:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def get(self, collection_id: str, *, request: ReadOptions | None = None) -> Collection:
        """Read or reattach the exact authorized resource with fresh server state.

        See: backend/notes/specifications/sdk/index_contract.md."""
        collection_id = identifier(collection_id)
        result = self._transport.request(
            operations.COLLECTION,
            Collection,
            path=f"/index/collections/{collection_id}",
            options=request,
            validate=lambda received: verified(
                received,
                received.collection_id == collection_id,
                "Resource identity or page bound mismatch",
            ),
        )
        return result

    def list(
        self, *, limit: int, cursor: PageCursor | None = None, request: ReadOptions | None = None
    ) -> Page[Collection]:
        """Read one bounded authorized page; no implicit all-pages traversal.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = self._transport.request(
            operations.COLLECTIONS,
            Page[Collection],
            path="/index/collections",
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda received: verified(
                received,
                not len(received.items) > limit,
                "Resource identity or page bound mismatch",
            ),
        )
        return result


class AsyncCollections:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def get(self, collection_id: str, *, request: ReadOptions | None = None) -> Collection:
        """Read or reattach the exact authorized resource with fresh server state.

        See: backend/notes/specifications/sdk/index_contract.md."""
        collection_id = identifier(collection_id)
        result = await self._transport.request(
            operations.COLLECTION,
            Collection,
            path=f"/index/collections/{collection_id}",
            options=request,
            validate=lambda received: verified(
                received,
                received.collection_id == collection_id,
                "Resource identity or page bound mismatch",
            ),
        )
        return result

    async def list(
        self, *, limit: int, cursor: PageCursor | None = None, request: ReadOptions | None = None
    ) -> Page[Collection]:
        """Read one bounded authorized page; no implicit all-pages traversal.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = await self._transport.request(
            operations.COLLECTIONS,
            Page[Collection],
            path="/index/collections",
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda received: verified(
                received,
                not len(received.items) > limit,
                "Resource identity or page bound mismatch",
            ),
        )
        return result
