"""``client.research.projects`` — alpha project surface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List

from synth_ai.managed_research.models.types import SmrProjectSetup, SmrRunnableProjectRequest
from synth_ai.managed_research.sdk.client import ManagedResearchClient
from synth_ai.research.models import ResearchCreateProjectResult, ResearchProject


class ResearchProjectsAPI:
    """Public Research project methods (alpha must-have)."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def create(
        self,
        request: SmrRunnableProjectRequest | Mapping[str, Any] | dict[str, Any],
    ) -> ResearchCreateProjectResult:
        """Create a runnable Research project (canonical name)."""
        return self.create_runnable(request)

    def create_runnable(
        self,
        request: SmrRunnableProjectRequest | Mapping[str, Any] | dict[str, Any],
    ) -> ResearchCreateProjectResult:
        return ResearchCreateProjectResult.from_wire(self._session.create_runnable_project(request))

    def list(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
    ) -> List[ResearchProject]:
        return self._session.projects.list(
            include_archived=include_archived,
            limit=limit,
        )

    def get(self, project_id: str) -> ResearchProject:
        return self._session.projects.get(project_id)

    def setup(self, project_id: str) -> SmrProjectSetup:
        return self._session.setup.get(project_id)

    def prepare_setup(self, project_id: str) -> SmrProjectSetup:
        return self._session.setup.prepare(project_id)

    def start_onboarding(self, project_id: str) -> dict[str, Any]:
        return self._session.setup.start_onboarding(project_id)

    def complete_onboarding_step(
        self,
        project_id: str,
        *,
        step: str,
        status: str,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._session.setup.complete_onboarding_step(
            project_id,
            step=step,
            status=status,
            detail=detail,
        )

    def dry_run_onboarding(self, project_id: str) -> dict[str, Any]:
        return self._session.setup.dry_run_onboarding(project_id)

    def onboarding_status(self, project_id: str) -> dict[str, Any]:
        return self._session.setup.get_onboarding_status(project_id)


__all__ = ["ResearchProjectsAPI"]
