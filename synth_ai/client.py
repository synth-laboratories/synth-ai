"""Python-only front-door SDK clients."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from synth_ai.core.auth.credentials import resolve_api_credential
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base
from synth_ai.sdk.index.timeouts import INDEX_TRANSPORT_TIMEOUT_SECONDS

if TYPE_CHECKING:
    from synth_ai.core.http.async_transport import AsyncHttpTransport
    from synth_ai.core.http.transport import HttpTransport
    from synth_ai.pools import PoolClient
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
    """Synchronous entry point for Index, Research, Messaging and hosted optimizers.

    Namespaces and their transports are created lazily. Closing the client closes
    all transports that it opened. Factory APIs remain available for compatibility.

    Example:
        ```python
        from synth_ai import SynthClient

        with SynthClient(api_key="YOUR_SYNTH_API_KEY") as client:
            index = client.index
        ```
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        """Configure credentials, endpoint and transport timeouts without making requests.

        Args:
            api_key: Explicit Synth API credential; None uses the configured credential resolver.
            base_url: Backend URL override; None uses the SDK's configured backend base URL.
            timeout_seconds: Transport timeout in seconds. None uses 30 seconds for
                Research, Messaging and Optimizers, and the Index transport default for Index.
        """
        self.api_key = _resolve_api_key(api_key)
        self.base_url = _resolve_base_url(base_url)
        self.timeout_seconds = 30.0 if timeout_seconds is None else timeout_seconds
        self._index_timeout_seconds = (
            INDEX_TRANSPORT_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )
        self._research_client: ResearchClient | None = None
        self._optimizers_client: OptimizersClient | None = None
        self._index_api: IndexAPI | None = None
        self._index_transport: HttpTransport | None = None
        self._messaging_client: MessagingClient | None = None

    @property
    def index(self) -> IndexAPI:
        """Return the lazily initialized Index namespace.

        Returns:
            IndexAPI using this client's credential, backend URL and Index timeout.
        """
        if self._index_api is None:
            from synth_ai.core.http.transport import HttpTransport
            from synth_ai.sdk.index.client import IndexAPI

            self._index_transport = HttpTransport(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout_seconds=self._index_timeout_seconds,
            )
            self._index_api = IndexAPI(self._index_transport)
        return self._index_api

    @property
    def messaging(self) -> MessagingClient:
        """Return the lazily initialized Messaging namespace.

        Returns:
            MessagingClient for typed threads, history and explicit Workshop device grants.
        """
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
        """Return the lazily initialized Research namespace.

        Returns:
            ResearchClient for hosted projects and swarms, with compatibility Factory APIs.
        """
        if self._research_client is None:
            from synth_ai.sdk.research.facade import ResearchClient

            self._research_client = ResearchClient(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout_seconds=self.timeout_seconds,
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
        """Return the lazily initialized hosted Optimizers namespace.

        Returns:
            OptimizersClient for hosted training model discovery and saved-LoRA lineage.
        """
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
    """Asynchronous entry point with lazily initialized SDK namespaces.

    Example:
        ```python
        from synth_ai import AsyncSynthClient

        async def inspect_namespace():
            async with AsyncSynthClient(api_key="YOUR_SYNTH_API_KEY") as client:
                index = client.index
        ```
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        """Configure credentials, endpoint and transport timeouts without making requests.

        Args:
            api_key: Explicit Synth API credential; None uses the configured credential resolver.
            base_url: Backend URL override; None uses the SDK's configured backend base URL.
            timeout_seconds: Transport timeout in seconds. None uses 30 seconds for
                Research, Messaging and Optimizers, and the Index transport default for Index.
        """
        self.api_key = _resolve_api_key(api_key)
        self.base_url = _resolve_base_url(base_url)
        self.timeout_seconds = 30.0 if timeout_seconds is None else timeout_seconds
        self._index_timeout_seconds = (
            INDEX_TRANSPORT_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )
        self._async_research_client: AsyncResearchClient | None = None
        self._async_optimizers_client: AsyncOptimizersClient | None = None
        self._index_api: AsyncIndexAPI | None = None
        self._index_transport: AsyncHttpTransport | None = None
        self._pool_client: PoolClient | None = None
        self._async_messaging_client: AsyncMessagingClient | None = None

    @property
    def index(self) -> AsyncIndexAPI:
        """Return the lazily initialized asynchronous Index namespace.

        Returns:
            AsyncIndexAPI using this client's credential, backend URL and Index timeout.
        """
        if self._index_api is None:
            from synth_ai.core.http.async_transport import AsyncHttpTransport
            from synth_ai.sdk.index.client import AsyncIndexAPI

            self._index_transport = AsyncHttpTransport(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout_seconds=self._index_timeout_seconds,
            )
            self._index_api = AsyncIndexAPI(self._index_transport)
        return self._index_api

    @property
    def messaging(self) -> AsyncMessagingClient:
        """Return the lazily initialized asynchronous Messaging namespace.

        Returns:
            AsyncMessagingClient for threads, history and explicit Workshop device grants.
        """
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
        """Return the lazily initialized asynchronous Research namespace.

        Returns:
            AsyncResearchClient using this client's credential, backend URL and timeout.
        """
        if self._async_research_client is None:
            from synth_ai.sdk.research import AsyncResearchClient

            self._async_research_client = AsyncResearchClient(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout_seconds=self.timeout_seconds,
            )
        return self._async_research_client

    @property
    def pools(self) -> PoolClient:
        """Backend-owned pool, deployment and interactive lease operations.

        See: evals/docs/handoffs/EVAL_EXECUTION_STREAMING_DELIVERY_PLAN_2026-09-10.md §3.
        Uses the canonical synth-containers client and this client's existing
        backend credential. Requires the optional ``synth-ai[pools]`` extra.
        Requests always pass through hosted admission and resource ownership.
        """
        if self._pool_client is None:
            from synth_ai.pools import PoolClient

            self._pool_client = PoolClient(
                api_key=self.api_key,
                backend_url=self.base_url,
                timeout_seconds=self.timeout_seconds,
            )
        return self._pool_client

    @property
    def async_research(self) -> AsyncResearchClient:
        """Deprecated alias for :attr:`research`.

        Returns:
            The same AsyncResearchClient instance as research, after a deprecation warning.
        """
        warnings.warn(
            "AsyncSynthClient.async_research is deprecated; use .research.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.research

    async def close(self) -> None:
        """Close all asynchronous Research transports."""
        try:
            if self._index_transport is not None:
                await self._index_transport.close()
                self._index_transport = None
                self._index_api = None
        finally:
            try:
                if self._async_messaging_client is not None:
                    await self._async_messaging_client.close()
                    self._async_messaging_client = None
            finally:
                try:
                    if self._async_research_client is not None:
                        await self._async_research_client.close()
                        self._async_research_client = None
                finally:
                    try:
                        if self._async_optimizers_client is not None:
                            await self._async_optimizers_client.close()
                            self._async_optimizers_client = None
                    finally:
                        if self._pool_client is not None:
                            await self._pool_client.aclose()
                            self._pool_client = None

    @property
    def optimizers(self) -> AsyncOptimizersClient:
        """Return the lazily initialized asynchronous hosted Optimizers namespace.

        Returns:
            AsyncOptimizersClient using this client's credential, backend URL and timeout.
        """
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
