"""Read-only account policy; see backend ACCOUNT_POLICY_ENGINE.md."""

from synth_ai.contracts.operations import ACCOUNT_ENTITLEMENTS
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.options import ReadOptions
from synth_ai.types.account import AccountEntitlements


class AccountClient:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def entitlements(self, *, request: ReadOptions | None = None) -> AccountEntitlements:
        """Read this organization's current access conditions and observed limits."""
        return self._transport.request(ACCOUNT_ENTITLEMENTS, AccountEntitlements, options=request)


class AsyncAccountClient:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def entitlements(self, *, request: ReadOptions | None = None) -> AccountEntitlements:
        """Read this organization's current access conditions and observed limits."""
        return await self._transport.request(
            ACCOUNT_ENTITLEMENTS, AccountEntitlements, options=request
        )
