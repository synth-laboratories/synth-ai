"""Public Research client implementations over the shared core transport."""

from __future__ import annotations

from synth_ai.core.auth.credentials import ApiCredential, resolve_api_credential
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.research.economics import (
    AsyncResearchEconomicsAPI,
    AsyncResearchLimitsAPI,
    ResearchEconomicsAPI,
    ResearchLimitsAPI,
)
from synth_ai.core.research.projects import AsyncResearchProjectsAPI, ResearchProjectsAPI
from synth_ai.core.research.swarms import AsyncResearchSwarmsAPI, ResearchSwarmsAPI
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base


class ResearchClient:
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
        self.projects = ResearchProjectsAPI(self._transport)
        self.swarms = ResearchSwarmsAPI(self._transport)
        self.economics = ResearchEconomicsAPI(self._transport)
        self.limits = ResearchLimitsAPI(self._transport)

    @property
    def credential(self) -> ApiCredential:
        return self._credential

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> ResearchClient:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class AsyncResearchClient:
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
        self.projects = AsyncResearchProjectsAPI(self._transport)
        self.swarms = AsyncResearchSwarmsAPI(self._transport)
        self.economics = AsyncResearchEconomicsAPI(self._transport)
        self.limits = AsyncResearchLimitsAPI(self._transport)

    @property
    def credential(self) -> ApiCredential:
        return self._credential

    async def close(self) -> None:
        await self._transport.close()

    async def __aenter__(self) -> AsyncResearchClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()


__all__ = ["AsyncResearchClient", "ResearchClient"]
