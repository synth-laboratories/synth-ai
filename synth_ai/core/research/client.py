"""Public Research client implementations over the shared core transport."""

from __future__ import annotations

from typing import TYPE_CHECKING

from synth_ai.core.auth.credentials import ApiCredential, resolve_api_credential
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.research.factories import AsyncFactoriesAPI, FactoriesAPI
from synth_ai.core.research.projects import AsyncProjectsAPI, ProjectsAPI
from synth_ai.core.research.swarms import AsyncSwarmsAPI, SwarmsAPI
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base

if TYPE_CHECKING:
    from synth_ai.core.research.economics import (
        AsyncEconomicsAPI,
        AsyncLimitsAPI,
        EconomicsAPI,
        LimitsAPI,
    )


class Client:
    """Synchronous typed client for projects and swarms."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._credential = resolve_api_credential(api_key)
        self._transport = HttpTransport(
            base_url=normalize_backend_base(base_url or BACKEND_URL_BASE),
            headers=self._credential.authorization_headers(),
            timeout_seconds=timeout_seconds,
        )
        self.projects = ProjectsAPI(self._transport)
        self.swarms = SwarmsAPI(self._transport)
        self.factories = FactoriesAPI(self._transport)
        self._economics: EconomicsAPI | None = None
        self._limits: LimitsAPI | None = None

    @property
    def economics(self) -> EconomicsAPI:
        """Advanced economics operations, loaded only when explicitly accessed."""
        if self._economics is None:
            from synth_ai.core.research.economics import EconomicsAPI

            self._economics = EconomicsAPI(self._transport)
        return self._economics

    @property
    def limits(self) -> LimitsAPI:
        """Advanced organization limits, loaded only when explicitly accessed."""
        if self._limits is None:
            from synth_ai.core.research.economics import LimitsAPI

            self._limits = LimitsAPI(self._transport)
        return self._limits

    @property
    def credential(self) -> ApiCredential:
        return self._credential

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class AsyncClient:
    """Native asynchronous typed client with sync surface parity."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._credential = resolve_api_credential(api_key)
        self._transport = AsyncHttpTransport(
            base_url=normalize_backend_base(base_url or BACKEND_URL_BASE),
            headers=self._credential.authorization_headers(),
            timeout_seconds=timeout_seconds,
        )
        self.projects = AsyncProjectsAPI(self._transport)
        self.swarms = AsyncSwarmsAPI(self._transport)
        self.factories = AsyncFactoriesAPI(self._transport)
        self._economics: AsyncEconomicsAPI | None = None
        self._limits: AsyncLimitsAPI | None = None

    @property
    def economics(self) -> AsyncEconomicsAPI:
        """Advanced async economics operations, loaded only on access."""
        if self._economics is None:
            from synth_ai.core.research.economics import AsyncEconomicsAPI

            self._economics = AsyncEconomicsAPI(self._transport)
        return self._economics

    @property
    def limits(self) -> AsyncLimitsAPI:
        """Advanced async organization limits, loaded only on access."""
        if self._limits is None:
            from synth_ai.core.research.economics import AsyncLimitsAPI

            self._limits = AsyncLimitsAPI(self._transport)
        return self._limits

    @property
    def credential(self) -> ApiCredential:
        return self._credential

    async def close(self) -> None:
        await self._transport.close()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()


ResearchClient = Client
AsyncResearchClient = AsyncClient


__all__ = ["AsyncClient", "AsyncResearchClient", "Client", "ResearchClient"]
