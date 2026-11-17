"""Async public client composition."""

from __future__ import annotations

from types import TracebackType

import httpx

from ._transport import RetryPolicy, Transport
from .account import AsyncAccount
from .auth import credentials
from .billing import AsyncBilling
from .diagnostics import DiagnosticHook
from .environments import Environment, resolve_base_url
from .swarms import AsyncSwarms


class AsyncSynthClient:
    """Access Swarms, account and billing through the public API.

    Use as a context manager to close client resources. API keys and base URLs
    can be passed explicitly or resolved by the documented authentication flow.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        environment: Environment | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 30,
        retries: int = 2,
        retry_policy: RetryPolicy | None = None,
        diagnostic_hook: DiagnosticHook | None = None,
    ) -> None:
        """Create a client with a request timeout in seconds and bounded retries."""
        url, token = credentials(resolve_base_url(base_url, environment), api_key)
        self._transport = Transport(
            url,
            token,
            client=http_client,
            timeout=timeout,
            retries=retries,
            retry_policy=retry_policy,
            diagnostic_hook=diagnostic_hook,
        )
        self.swarms = AsyncSwarms(self._transport)
        self.account = AsyncAccount(self._transport)
        self.billing = AsyncBilling(self._transport)

    async def close(self) -> None:
        """Close the client transport and owned network resources."""
        await self._transport.close()

    async def __aenter__(self) -> AsyncSynthClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()
