"""``SynthClient().research`` namespace (alpha)."""

from __future__ import annotations

import warnings
from typing import Any

from synth_ai.managed_research.sdk.client import ManagedResearchClient
from synth_ai.managed_research.sdk.tag import TagAPI
from synth_ai.research.economics import ResearchEconomicsAPI
from synth_ai.research.efforts import ResearchEffortsAPI
from synth_ai.research.factories import ResearchFactoriesAPI
from synth_ai.research.hosted_artifacts import ResearchHostedArtifactsAPI
from synth_ai.research.limits import ResearchLimitsAPI
from synth_ai.research.projects import ResearchProjectsAPI
from synth_ai.research.secrets import ResearchSecretsAPI
from synth_ai.research.swarms import ResearchSwarmsAPI
from synth_ai.research.visuals import ResearchVisualsAPI


class ResearchClient:
    """Managed Research entrypoint on ``SynthClient``.

    SMR (Managed Research) is the product umbrella; its capabilities are
    Managed Factories (``factories``) and Managed Swarms (``swarms``) — swarms
    launch directly or are composed by factories. Namespaces also cover
    projects, limits, economics, secrets, and Factory Tag (``factories.tag``).

    Example:
        >>> client = SynthClient()
        >>> research = client.research
        >>> research.limits.get()
        >>> research.projects.create({"name": "demo", "work_mode": "standard"})
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
        self._efforts: ResearchEffortsAPI | None = None
        self._projects: ResearchProjectsAPI | None = None
        self._swarms: ResearchSwarmsAPI | None = None
        self._limits: ResearchLimitsAPI | None = None
        self._economics: ResearchEconomicsAPI | None = None
        self._secrets: ResearchSecretsAPI | None = None
        self._hosted_artifacts: ResearchHostedArtifactsAPI | None = None
        self._visuals: ResearchVisualsAPI | None = None
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
        """Low-level session client (advanced integrations and eval harnesses only).

        Prefer hero namespaces (``projects``, ``swarms``, ``limits``) for new code.
        """
        return self._open_session()

    @property
    def backing_client(self) -> ManagedResearchClient:
        """Deprecated alias for :attr:`session`."""
        warnings.warn(
            "research.backing_client is deprecated; use research.session instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._open_session()

    @property
    def factories(self) -> ResearchFactoriesAPI:
        """Factory domain APIs (Tag at ``factories.tag``)."""
        if self._factories is None:
            self._factories = ResearchFactoriesAPI(self._open_session())
        return self._factories

    @property
    def efforts(self) -> ResearchEffortsAPI:
        """Graduate runs into persistent Efforts (``efforts.proposals`` for proposals)."""
        if self._efforts is None:
            self._efforts = ResearchEffortsAPI(self._open_session())
        return self._efforts

    @property
    def projects(self) -> ResearchProjectsAPI:
        """Create and configure Managed Research projects."""
        if self._projects is None:
            self._projects = ResearchProjectsAPI(self._open_session())
        return self._projects

    @property
    def swarms(self) -> ResearchSwarmsAPI:
        """Launch Managed Swarms and open swarm-scoped readout handles."""
        if self._swarms is None:
            self._swarms = ResearchSwarmsAPI(self._open_session())
        return self._swarms

    @property
    def runs(self) -> ResearchSwarmsAPI:
        """Deprecated alias for :attr:`swarms` (the public noun is Swarm)."""
        warnings.warn(
            "research.runs is deprecated; use research.swarms instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.swarms

    @property
    def limits(self) -> ResearchLimitsAPI:
        """Read org limits and allowance before launching work."""
        if self._limits is None:
            self._limits = ResearchLimitsAPI(self._open_session())
        return self._limits

    @property
    def economics(self) -> ResearchEconomicsAPI:
        """Read authoritative org entitlements and project economics."""
        if self._economics is None:
            self._economics = ResearchEconomicsAPI(self._open_session())
        return self._economics

    @property
    def secrets(self) -> ResearchSecretsAPI:
        """Manage project secret refs for providers and repos."""
        if self._secrets is None:
            self._secrets = ResearchSecretsAPI(self._open_session())
        return self._secrets

    @property
    def hosted_artifacts(self) -> ResearchHostedArtifactsAPI:
        """Open Research hosted artifact operator API (read, promote, review)."""
        if self._hosted_artifacts is None:
            self._hosted_artifacts = ResearchHostedArtifactsAPI(self._open_session())
        return self._hosted_artifacts

    @property
    def visuals(self) -> ResearchVisualsAPI:
        """Publish and browse first-class blob-backed Synth Visuals."""
        if self._visuals is None:
            self._visuals = ResearchVisualsAPI(self._open_session())
        return self._visuals

    @property
    def tag(self) -> TagAPI:
        """Deprecated — use ``factories.tag`` instead."""
        warnings.warn(
            "client.research.tag is deprecated; use client.research.factories.tag instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._tag is None:
            self._tag = self._open_session().tag
        return self._tag

    def get_limits(self) -> dict[str, Any]:
        """Return the legacy raw payload; use typed ``limits.get_typed()`` instead."""
        warnings.warn(
            "research.get_limits() is deprecated; use research.limits.get_typed() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._open_session().get_limits()

    def close(self) -> None:
        """Close the underlying HTTP session and cached namespace clients."""
        if self._session is not None:
            self._session.close()
        self._session = None
        self._factories = None
        self._efforts = None
        self._projects = None
        self._swarms = None
        self._limits = None
        self._economics = None
        self._secrets = None
        self._hosted_artifacts = None
        self._visuals = None
        self._tag = None


__all__ = ["ResearchClient"]
