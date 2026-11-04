"""Async public client composition."""

from __future__ import annotations

import os
from typing import Any

import httpx

from ._transport import Transport
from .account import AsyncAccount
from .auth import credentials
from .billing import AsyncBilling
from .swarms import AsyncSwarms


class AsyncSynthClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        project_id: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 30,
        retries: int = 2,
    ) -> None:
        url, token = credentials(base_url, api_key)
        self._transport = Transport(
            url, token, client=http_client, timeout=timeout, retries=retries
        )
        self.swarms = AsyncSwarms(self._transport, project_id or os.getenv("SYNTH_PROJECT_ID"))
        self.account = AsyncAccount(self._transport)
        self.billing = AsyncBilling(self._transport)

    async def close(self) -> None:
        await self._transport.close()

    async def __aenter__(self) -> AsyncSynthClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
