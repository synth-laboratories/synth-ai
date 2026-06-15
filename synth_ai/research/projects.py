"""``client.research.projects`` — alpha project surface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List

from synth_ai.managed_research.models.types import SmrRunnableProjectRequest
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


__all__ = ["ResearchProjectsAPI"]
