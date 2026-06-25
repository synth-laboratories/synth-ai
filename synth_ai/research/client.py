"""``SynthClient().research`` namespace (alpha)."""

from __future__ import annotations

import warnings
from typing import Any

from synth_ai.managed_research.sdk.client import ManagedResearchClient
from synth_ai.managed_research.sdk.tag import TagAPI
from synth_ai.research.factories import ResearchFactoriesAPI
from synth_ai.research.limits import ResearchLimitsAPI
from synth_ai.research.projects import ResearchProjectsAPI
from synth_ai.research.runs import ResearchRunsAPI
from synth_ai.research.secrets import ResearchSecretsAPI


class ResearchClient:
    """Public Research API under ``SynthClient``.

    One ``ManagedResearchClient`` session backs all hero namespaces.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self._session: ManagedResearchClient | None = None
        self._factories: ResearchFactoriesAPI | None = None
        self._projects: ResearchProjectsAPI | None = None
        self._runs: ResearchRunsAPI | None = None
        self._limits: ResearchLimitsAPI | None = None
        self._secrets: ResearchSecretsAPI | None = None
        self._tag: TagAPI | None = None

    def _open_session(self) -> ManagedResearchClient:
        if self._session is None:
            self._session = ManagedResearchClient(
                api_key=self.api_key,
                backend_base=self.base_url,
                timeout_seconds=self.timeout_seconds,
            )
        return self._session

    @property
    def session(self) -> ManagedResearchClient:
        """Advanced: wire session backing hero namespaces (eval harness interim)."""
        return self._open_session()

    @property
    def backing_client(self) -> ManagedResearchClient:
        warnings.warn(
            "research.backing_client is deprecated; use research.session instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._open_session()

    @property
    def factories(self) -> ResearchFactoriesAPI:
        if self._factories is None:
            self._factories = ResearchFactoriesAPI(self._open_session())
        return self._factories

    @property
    def projects(self) -> ResearchProjectsAPI:
        if self._projects is None:
            self._projects = ResearchProjectsAPI(self._open_session())
        return self._projects

    @property
    def runs(self) -> ResearchRunsAPI:
        if self._runs is None:
            self._runs = ResearchRunsAPI(self._open_session())
        return self._runs

    @property
    def limits(self) -> ResearchLimitsAPI:
        if self._limits is None:
            self._limits = ResearchLimitsAPI(self._open_session())
        return self._limits

    @property
    def secrets(self) -> ResearchSecretsAPI:
        if self._secrets is None:
            self._secrets = ResearchSecretsAPI(self._open_session())
        return self._secrets

    @property
    def tag(self) -> TagAPI:
        warnings.warn(
            "client.research.tag is deprecated; use client.research.factories.tag instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._tag is None:
            self._tag = self._open_session().tag
        return self._tag

    def get_limits(self) -> dict[str, Any]:
        warnings.warn(
            "research.get_limits() is deprecated; use research.limits.get() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.limits.get()

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
        self._session = None
        self._factories = None
        self._projects = None
        self._runs = None
        self._limits = None
        self._secrets = None
        self._tag = None


__all__ = ["ResearchClient"]
