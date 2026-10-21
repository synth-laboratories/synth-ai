"""Read-only setup resources. See SDK index_contract.md."""

from __future__ import annotations

from synth_ai.contracts import operations
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.options import ReadOptions

from .types.account import IndexAccount


class Account:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def get(self, *, request: ReadOptions | None = None) -> IndexAccount:
        """Read or reattach the exact authorized resource with fresh server state.

        See: the public contract snapshots in contracts/"""
        result = self._transport.request(
            operations.ACCOUNT, IndexAccount, path="/index/account", options=request
        )
        return result


class AsyncAccount:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def get(self, *, request: ReadOptions | None = None) -> IndexAccount:
        """Read or reattach the exact authorized resource with fresh server state.

        See: the public contract snapshots in contracts/"""
        result = await self._transport.request(
            operations.ACCOUNT, IndexAccount, path="/index/account", options=request
        )
        return result
