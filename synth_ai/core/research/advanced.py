"""Explicitly unstable Research capabilities awaiting typed graduation."""

from __future__ import annotations

from collections.abc import Callable

from synth_ai.core.research._legacy.sdk.client import ManagedResearchClient
from synth_ai.core.research._legacy.sdk.images import ImagesAPI
from synth_ai.core.research._legacy.sdk.tag import TagAPI
from synth_ai.core.research.advanced_factories import (
    ResearchFactoriesAPI as AdvancedFactoriesAPI,
)
from synth_ai.core.research.artifacts import ResearchHostedArtifactsAPI
from synth_ai.core.research.economics import ResearchEconomicsAPI, ResearchLimitsAPI
from synth_ai.core.research.efforts import ResearchEffortsAPI
from synth_ai.core.research.secrets import ResearchSecretsAPI
from synth_ai.core.research.visuals import ResearchVisualsAPI


def open_advanced_session(
    *,
    api_key: str,
    base_url: str,
    timeout_seconds: float,
) -> ManagedResearchClient:
    """Construct the compatibility session only after advanced access."""
    return ManagedResearchClient(
        api_key=api_key,
        backend_base=base_url,
        timeout_seconds=timeout_seconds,
    )


class ResearchAdvancedAPI:
    """Unstable operator surface, deliberately separated from the hero loop."""

    def __init__(
        self,
        *,
        open_session: Callable[[], ManagedResearchClient],
        limits: ResearchLimitsAPI,
        economics: ResearchEconomicsAPI,
    ) -> None:
        self._open_session = open_session
        self.limits = limits
        self.economics = economics
        self._efforts: ResearchEffortsAPI | None = None
        self._factories: AdvancedFactoriesAPI | None = None
        self._secrets: ResearchSecretsAPI | None = None
        self._artifacts: ResearchHostedArtifactsAPI | None = None
        self._visuals: ResearchVisualsAPI | None = None

    @property
    def session(self) -> ManagedResearchClient:
        """Low-level operator client; capabilities may change before 1.0."""
        return self._open_session()

    @property
    def efforts(self) -> ResearchEffortsAPI:
        if self._efforts is None:
            self._efforts = ResearchEffortsAPI(self._open_session())
        return self._efforts

    @property
    def factories(self) -> AdvancedFactoriesAPI:
        """Operator Factory projections not yet admitted to the stable contract."""
        if self._factories is None:
            self._factories = AdvancedFactoriesAPI(self._open_session())
        return self._factories

    @property
    def secrets(self) -> ResearchSecretsAPI:
        if self._secrets is None:
            self._secrets = ResearchSecretsAPI(self._open_session())
        return self._secrets

    @property
    def artifacts(self) -> ResearchHostedArtifactsAPI:
        if self._artifacts is None:
            self._artifacts = ResearchHostedArtifactsAPI(self._open_session())
        return self._artifacts

    @property
    def visuals(self) -> ResearchVisualsAPI:
        if self._visuals is None:
            self._visuals = ResearchVisualsAPI(self._open_session())
        return self._visuals

    @property
    def images(self) -> ImagesAPI:
        return self._open_session().images

    @property
    def tag(self) -> TagAPI:
        return self._open_session().tag


__all__ = ["ResearchAdvancedAPI"]
