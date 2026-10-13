"""Current operation summaries and bounded items. See SDK index_contract.md."""

from __future__ import annotations

from synth_ai.contracts import operations
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.parameters import identifier, page_parameters, verified
from synth_ai.options import ReadOptions
from synth_ai.types.base import Page, PageCursor
from synth_ai.types.usage import UsageOperation, UsageSummary


class Usage:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def get(self, operation_id: str, *, request: ReadOptions | None = None) -> UsageSummary:
        """Read or reattach the exact authorized resource with fresh server state.

        See: backend/notes/specifications/sdk/index_contract.md."""
        operation_id = identifier(operation_id)
        result = self._transport.request(
            operations.USAGE,
            UsageSummary,
            path=f"/index/usage/{operation_id}",
            options=request,
            validate=lambda received: verified(
                received,
                received.operation_id == operation_id,
                "Resource identity or page bound mismatch",
            ),
        )
        return result

    def operations(
        self,
        operation_id: str,
        *,
        limit: int,
        cursor: PageCursor | None = None,
        request: ReadOptions | None = None,
    ) -> Page[UsageOperation]:
        """Read bounded usage items; clients do not reconstruct authoritative totals.

        See: backend/notes/specifications/sdk/index_contract.md."""
        operation_id = identifier(operation_id)
        result = self._transport.request(
            operations.USAGE_ITEMS,
            Page[UsageOperation],
            path=f"/index/usage/{operation_id}/operations",
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda received: verified(
                received,
                not len(received.items) > limit,
                "Resource identity or page bound mismatch",
            ),
        )
        return result


class AsyncUsage:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def get(self, operation_id: str, *, request: ReadOptions | None = None) -> UsageSummary:
        """Read or reattach the exact authorized resource with fresh server state.

        See: backend/notes/specifications/sdk/index_contract.md."""
        operation_id = identifier(operation_id)
        result = await self._transport.request(
            operations.USAGE,
            UsageSummary,
            path=f"/index/usage/{operation_id}",
            options=request,
            validate=lambda received: verified(
                received,
                received.operation_id == operation_id,
                "Resource identity or page bound mismatch",
            ),
        )
        return result

    async def operations(
        self,
        operation_id: str,
        *,
        limit: int,
        cursor: PageCursor | None = None,
        request: ReadOptions | None = None,
    ) -> Page[UsageOperation]:
        """Read bounded usage items; clients do not reconstruct authoritative totals.

        See: backend/notes/specifications/sdk/index_contract.md."""
        operation_id = identifier(operation_id)
        result = await self._transport.request(
            operations.USAGE_ITEMS,
            Page[UsageOperation],
            path=f"/index/usage/{operation_id}/operations",
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda received: verified(
                received,
                not len(received.items) > limit,
                "Resource identity or page bound mismatch",
            ),
        )
        return result
