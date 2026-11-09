"""Synchronous facade over one dedicated async client loop."""

from __future__ import annotations

from types import TracebackType

import httpx

from ._blocking import Runner
from ._sync import AccountClient, BillingClient, SwarmsClient
from ._transport import RetryPolicy
from .aclient import AsyncSynthClient
from .diagnostics import DiagnosticHook


class SynthClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 30,
        retries: int = 2,
        retry_policy: RetryPolicy | None = None,
        diagnostic_hook: DiagnosticHook | None = None,
    ) -> None:
        self._async = AsyncSynthClient(
            api_key,
            base_url=base_url,
            http_client=http_client,
            timeout=timeout,
            retries=retries,
            retry_policy=retry_policy,
            diagnostic_hook=diagnostic_hook,
        )
        self._runner = Runner()
        self.swarms = SwarmsClient(self._runner, self._async.swarms)
        self.account = AccountClient(self._runner, self._async.account)
        self.billing = BillingClient(self._runner, self._async.billing)

    def close(self) -> None:
        if self._runner.closed:
            return
        try:
            self._runner.call(self._async.close())
        finally:
            self._runner.close()

    def __enter__(self) -> SynthClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
