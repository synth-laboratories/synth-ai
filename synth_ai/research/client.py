"""``SynthClient().research`` namespace (alpha)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from synth_ai.core.research.client import ResearchClient as CoreResearchClient
from synth_ai.core.research.factories import FactoriesAPI
from synth_ai.core.research.environments import EnvironmentsAPI
from synth_ai.core.research.projects import ResearchProjectsAPI
from synth_ai.core.research.swarms import ResearchSwarmsAPI

if TYPE_CHECKING:
    from synth_ai.core.research.advanced import (
        ManagedResearchClient,
        ResearchAdvancedAPI,
    )


class Client:
    """Research entrypoint on ``SynthClient``.

    Obtain via ``SynthClient().research``. The three hero namespaces are
    projects, swarms, and factories.

    Example:
        >>> client = SynthClient()
        >>> research = client.research
        >>> project = research.projects.create(request)
        >>> swarm = research.swarms.create(project.project_id, request=launch)
        >>> swarm.wait()
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
        self._core = CoreResearchClient(
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        self._session: ManagedResearchClient | None = None
        self._advanced: ResearchAdvancedAPI | None = None

    def _open_session(self) -> ManagedResearchClient:
        if self._session is None:
            from synth_ai.core.research.advanced import open_advanced_session

            self._session = open_advanced_session(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout_seconds=self.timeout_seconds,
            )
        return self._session

    @property
    def advanced(self) -> ResearchAdvancedAPI:
        """Explicitly unstable operator capabilities outside the hero workflow."""
        if self._advanced is None:
            from synth_ai.core.research.advanced import ResearchAdvancedAPI

            self._advanced = ResearchAdvancedAPI(
                open_session=self._open_session,
                limits=self._core.limits,
                economics=self._core.economics,
            )
        return self._advanced

    @property
    def factories(self) -> FactoriesAPI:
        """Stable Factory lifecycle and typed Efforts."""
        return self._core.factories

    @property
    def environments(self) -> EnvironmentsAPI:
        """Versioned runtime declarations and deterministic preflight."""
        return self._core.environments

    @property
    def projects(self) -> ResearchProjectsAPI:
        """Create and configure Research projects through the core client."""
        return self._core.projects

    @property
    def swarms(self) -> ResearchSwarmsAPI:
        """Launch and control typed Research swarms."""
        return self._core.swarms

    @property
    def runs(self) -> ResearchSwarmsAPI:
        """Deprecated alias for :attr:`swarms`."""
        warnings.warn(
            "research.runs is deprecated; use research.swarms instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._core.swarms

    def __getattr__(self, name: str) -> object:
        compatibility = {
            "session": "session",
            "backing_client": "session",
            "efforts": "efforts",
            "limits": "limits",
            "economics": "economics",
            "secrets": "secrets",
            "hosted_artifacts": "artifacts",
            "visuals": "visuals",
            "images": "images",
            "tag": "tag",
        }
        target = compatibility.get(name)
        if target is not None:
            warnings.warn(
                f"research.{name} is deprecated; use research.advanced.{target}.",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(self.advanced, target)
        if name == "get_limits":
            warnings.warn(
                "research.get_limits is deprecated; use research.advanced.limits.retrieve.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.advanced.limits.retrieve
        raise AttributeError(name)

    def close(self) -> None:
        """Close the underlying HTTP session and cached namespace clients."""
        self._core.close()
        if self._session is not None:
            self._session.close()
        self._session = None
        self._advanced = None


ResearchClient = Client


__all__ = ["Client", "ResearchClient"]
