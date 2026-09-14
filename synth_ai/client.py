"""Python-only front-door SDK clients."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from synth_ai.core.auth.credentials import resolve_api_credential
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base

if TYPE_CHECKING:
    from synth_ai.core.http.async_transport import AsyncHttpTransport
    from synth_ai.core.http.transport import HttpTransport
    from synth_ai.sdk.index.client import AsyncIndexAPI, IndexAPI
    from synth_ai.sdk.messaging import AsyncMessagingClient, MessagingClient
    from synth_ai.sdk.optimizers import AsyncOptimizersClient, OptimizersClient
    from synth_ai.sdk.research import AsyncResearchClient
    from synth_ai.sdk.research.facade import ResearchClient


def _resolve_api_key(api_key: str | None) -> str:
    return resolve_api_credential(api_key).value


def _resolve_base_url(base_url: str | None) -> str:
    return normalize_backend_base(base_url or BACKEND_URL_BASE)


class SynthClient:
    """Sync client for Managed Research and hosted optimizers.

    Use ``research`` for hosted projects, swarms, and Factory lifecycles. That
    Use ``optimizers`` for hosted training model discovery and saved LoRAs.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        allow_legacy_intern_sessions: bool = False,
    ) -> None:
        self.api_key = _resolve_api_key(api_key)
        self.base_url = _resolve_base_url(base_url)
        self.timeout_seconds = timeout_seconds
        self.allow_legacy_intern_sessions = allow_legacy_intern_sessions
        self._research_client: ResearchClient | None = None
        self._optimizers_client: OptimizersClient | None = None
        self._index_api: IndexAPI | None = None
        self._index_transport: HttpTransport | None = None
        self._messaging_client: MessagingClient | None = None

    @property
    def index(self) -> IndexAPI:
        """Unreleased fast research search and exact-reference contents namespace."""
        if self._index_api is None:
            from synth_ai.core.http.transport import HttpTransport
            from synth_ai.sdk.index.client import IndexAPI

            self._index_transport = HttpTransport(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout_seconds=self.timeout_seconds,
            )
            self._index_api = IndexAPI(self._index_transport)
        return self._index_api

    @property
    def messaging(self) -> MessagingClient:
        """Typed threads, history and explicit Workshop device grants."""
        if self._messaging_client is None:
            from synth_ai.sdk.messaging import MessagingClient

            self._messaging_client = MessagingClient(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout_seconds=self.timeout_seconds,
            )
        return self._messaging_client

    @property
    def research(self) -> ResearchClient:
        """Research hero namespace (projects, swarms, and factories)."""
        if self._research_client is None:
            from synth_ai.sdk.research.facade import ResearchClient

            self._research_client = ResearchClient(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout_seconds=self.timeout_seconds,
                allow_legacy_intern_sessions=self.allow_legacy_intern_sessions,
            )
        return self._research_client

    def close(self) -> None:
        """Close all lazily opened SDK transports."""
        if self._index_transport is not None:
            self._index_transport.close()
            self._index_transport = None
            self._index_api = None
        if self._messaging_client is not None:
            self._messaging_client.close()
            self._messaging_client = None
        if self._research_client is not None:
            self._research_client.close()
            self._research_client = None
        if self._optimizers_client is not None:
            self._optimizers_client.close()
            self._optimizers_client = None

    @property
    def optimizers(self) -> OptimizersClient:
        """Hosted training models and searchable saved-LoRA lineage."""
        if self._optimizers_client is None:
            from synth_ai.sdk.optimizers import OptimizersClient

            self._optimizers_client = OptimizersClient(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout_seconds=self.timeout_seconds,
            )
        return self._optimizers_client

    def __enter__(self) -> SynthClient:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class AsyncSynthClient:
    """Async client for Managed Research and hosted optimizers."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        allow_legacy_intern_sessions: bool = False,
    ) -> None:
        self.api_key = _resolve_api_key(api_key)
        self.base_url = _resolve_base_url(base_url)
        self.timeout_seconds = timeout_seconds
        self.allow_legacy_intern_sessions = allow_legacy_intern_sessions
        self._async_research_client: AsyncResearchClient | None = None
        self._async_optimizers_client: AsyncOptimizersClient | None = None
        self._index_api: AsyncIndexAPI | None = None
        self._index_transport: AsyncHttpTransport | None = None
        self._async_messaging_client: AsyncMessagingClient | None = None

    @property
    def index(self) -> AsyncIndexAPI:
        """Unreleased asynchronous fast search and exact-reference contents."""
        if self._index_api is None:
            from synth_ai.core.http.async_transport import AsyncHttpTransport
            from synth_ai.sdk.index.client import AsyncIndexAPI

            self._index_transport = AsyncHttpTransport(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout_seconds=self.timeout_seconds,
            )
            self._index_api = AsyncIndexAPI(self._index_transport)
        return self._index_api

    @property
    def messaging(self) -> AsyncMessagingClient:
        """Asynchronous threads, history and explicit Workshop device grants."""
        if self._async_messaging_client is None:
            from synth_ai.sdk.messaging import AsyncMessagingClient

            self._async_messaging_client = AsyncMessagingClient(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout_seconds=self.timeout_seconds,
            )
        return self._async_messaging_client

    @property
    def research(self) -> AsyncResearchClient:
        """Native asynchronous Research namespace."""
        if self._async_research_client is None:
            from synth_ai.sdk.research import AsyncResearchClient

            self._async_research_client = AsyncResearchClient(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout_seconds=self.timeout_seconds,
                allow_legacy_intern_sessions=self.allow_legacy_intern_sessions,
            )
        return self._async_research_client

    @property
    def async_research(self) -> AsyncResearchClient:
        """Deprecated alias for :attr:`research`."""
        warnings.warn(
            "AsyncSynthClient.async_research is deprecated; use .research.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.research

    async def close(self) -> None:
        """Close all asynchronous Research transports."""
        if self._index_transport is not None:
            await self._index_transport.close()
            self._index_transport = None
            self._index_api = None
        if self._async_messaging_client is not None:
            await self._async_messaging_client.close()
            self._async_messaging_client = None
        if self._async_research_client is not None:
            await self._async_research_client.close()
            self._async_research_client = None
        if self._async_optimizers_client is not None:
            await self._async_optimizers_client.close()
            self._async_optimizers_client = None

    @property
    def optimizers(self) -> AsyncOptimizersClient:
        """Native asynchronous hosted optimizer namespace."""
        if self._async_optimizers_client is None:
            from synth_ai.sdk.optimizers import AsyncOptimizersClient

            self._async_optimizers_client = AsyncOptimizersClient(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout_seconds=self.timeout_seconds,
            )
        return self._async_optimizers_client

    async def __aenter__(self) -> AsyncSynthClient:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        await self.close()


__all__ = [
    "AsyncSynthClient",
    "SynthClient",
]
