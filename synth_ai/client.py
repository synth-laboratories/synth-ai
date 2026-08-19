"""Python-only front-door SDK clients."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from synth_ai.core.auth.credentials import resolve_api_credential
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base

if TYPE_CHECKING:
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
