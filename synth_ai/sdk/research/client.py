"""Public Research client implementations over the shared core transport."""

from __future__ import annotations

from typing import TYPE_CHECKING

from synth_ai.core.auth.credentials import ApiCredential, resolve_api_credential
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.utils.urls import BACKEND_URL_BASE, normalize_backend_base
from synth_ai.sdk.research.environments import AsyncEnvironmentsAPI, EnvironmentsAPI
from synth_ai.sdk.research.factories import AsyncFactoriesAPI, FactoriesAPI
from synth_ai.sdk.research.image_releases import (
    AsyncImageReleasesAPI,
    ImageReleasesAPI,
)
from synth_ai.sdk.research.projects import AsyncProjectsAPI, ProjectsAPI
from synth_ai.sdk.research.research_intern import (
    AsyncResearchInternAPI,
    ResearchInternAPI,
)
from synth_ai.sdk.research.swarms import AsyncSwarmsAPI, SwarmsAPI
from synth_ai.sdk.research.traces import AsyncResearchTracesAPI, ResearchTracesAPI
from synth_ai.sdk.research.visuals import AsyncVisualsAPI, VisualsAPI

if TYPE_CHECKING:
    from synth_ai.sdk.research.economics import (
        AsyncEconomicsAPI,
        AsyncLimitsAPI,
        EconomicsAPI,
        LimitsAPI,
    )


class Client:
    """Synchronous typed client over the Research namespaces.

    Mounts projects, intern, swarms, factories, environments, image_releases,
    traces, and visuals; economics and limits load lazily.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        allow_legacy_intern_sessions: bool = False,
    ) -> None:
        self._credential = resolve_api_credential(api_key)
        self._transport = HttpTransport(
            base_url=normalize_backend_base(base_url or BACKEND_URL_BASE),
            headers=self._credential.authorization_headers(),
            timeout_seconds=timeout_seconds,
        )
        self.projects = ProjectsAPI(self._transport)
        self.intern = ResearchInternAPI(
            self._transport,
            allow_legacy_intern_sessions=allow_legacy_intern_sessions,
        )
        self.swarms = SwarmsAPI(self._transport)
        self.factories = FactoriesAPI(self._transport)
        self.environments = EnvironmentsAPI(self._transport)
        self.image_releases = ImageReleasesAPI(self._transport)
        self.traces = ResearchTracesAPI(self._transport)
        self.visuals = VisualsAPI(self._transport)
        self._economics: EconomicsAPI | None = None
        self._limits: LimitsAPI | None = None

    @property
    def transport(self) -> HttpTransport:
        """The shared typed HTTP transport every namespace routes through."""
        return self._transport

    @property
    def economics(self) -> EconomicsAPI:
        """Advanced economics operations, loaded only when explicitly accessed."""
        if self._economics is None:
            from synth_ai.sdk.research.economics import EconomicsAPI

            self._economics = EconomicsAPI(self._transport)
        return self._economics

    @property
    def limits(self) -> LimitsAPI:
        """Advanced organization limits, loaded only when explicitly accessed."""
        if self._limits is None:
            from synth_ai.sdk.research.economics import LimitsAPI

            self._limits = LimitsAPI(self._transport)
        return self._limits

    @property
    def credential(self) -> ApiCredential:
        """Return the resolved API credential used by this client.

        Returns:
            The API credential that supplied authorization headers for the transport.
        """
        return self._credential

    def close(self) -> None:
        """Close the underlying HTTP transport."""
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
        allow_legacy_intern_sessions: bool = False,
    ) -> None:
        self._credential = resolve_api_credential(api_key)
        self._transport = AsyncHttpTransport(
            base_url=normalize_backend_base(base_url or BACKEND_URL_BASE),
            headers=self._credential.authorization_headers(),
            timeout_seconds=timeout_seconds,
        )
        self.projects = AsyncProjectsAPI(self._transport)
        self.intern = AsyncResearchInternAPI(
            self._transport,
            allow_legacy_intern_sessions=allow_legacy_intern_sessions,
        )
        self.swarms = AsyncSwarmsAPI(self._transport)
        self.factories = AsyncFactoriesAPI(self._transport)
        self.environments = AsyncEnvironmentsAPI(self._transport)
        self.image_releases = AsyncImageReleasesAPI(self._transport)
        self.traces = AsyncResearchTracesAPI(self._transport)
        self.visuals = AsyncVisualsAPI(self._transport)
        self._economics: AsyncEconomicsAPI | None = None
        self._limits: AsyncLimitsAPI | None = None

    @property
    def transport(self) -> AsyncHttpTransport:
        """The shared typed async HTTP transport every namespace routes through."""
        return self._transport

    @property
    def economics(self) -> AsyncEconomicsAPI:
        """Advanced async economics operations, loaded only on access."""
        if self._economics is None:
            from synth_ai.sdk.research.economics import AsyncEconomicsAPI

            self._economics = AsyncEconomicsAPI(self._transport)
        return self._economics

    @property
    def limits(self) -> AsyncLimitsAPI:
        """Advanced async organization limits, loaded only on access."""
        if self._limits is None:
            from synth_ai.sdk.research.economics import AsyncLimitsAPI

            self._limits = AsyncLimitsAPI(self._transport)
        return self._limits

    @property
    def credential(self) -> ApiCredential:
        """Return the resolved API credential used by this client.

        Returns:
            The API credential that supplied authorization headers for the transport.
        """
        return self._credential

    async def close(self) -> None:
        """Close the underlying HTTP transport."""
        await self._transport.close()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()


# Public SynthClient().research facade is synth_ai.sdk.research.facade.ResearchClient.
# This module is the typed HTTP transport client only.

__all__ = ["AsyncClient", "Client"]
