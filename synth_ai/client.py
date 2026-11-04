"""Synchronous facade over one dedicated async client loop."""

from __future__ import annotations

from typing import Any

from ._blocking import Runner
from ._sync import AccountClient, BillingClient, SwarmsClient
from .aclient import AsyncSynthClient


class SynthClient:
    def __init__(self, api_key: str | None = None, **options: Any) -> None:
        self._async = AsyncSynthClient(api_key, **options)
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

    def __exit__(self, *args: Any) -> None:
        self.close()
