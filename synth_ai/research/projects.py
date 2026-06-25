"""``client.research.projects`` — alpha project surface."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, List

from synth_ai.managed_research.models.types import SmrProjectSetup, SmrRunnableProjectRequest
from synth_ai.managed_research.sdk.client import ManagedResearchClient
from synth_ai.research.models import ResearchCreateProjectResult, ResearchProject
from synth_ai.research.project_namespaces import (
    ResearchProjectsCodeAPI,
    ResearchProjectsGitAPI,
    ResearchProjectsMilestonesAPI,
    ResearchProjectsObjectivesAPI,
    ResearchProjectsReposAPI,
    ResearchProjectsRunsAPI,
    ResearchProjectsSetupAPI,
    ResearchProjectsWorkspaceAPI,
)


class ResearchProjectsAPI:
    """Public Research project methods (alpha must-have)."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session
        self._setup: ResearchProjectsSetupAPI | None = None
        self._workspace: ResearchProjectsWorkspaceAPI | None = None
        self._repos: ResearchProjectsReposAPI | None = None
        self._git: ResearchProjectsGitAPI | None = None
        self._code: ResearchProjectsCodeAPI | None = None
        self._objectives: ResearchProjectsObjectivesAPI | None = None
        self._milestones: ResearchProjectsMilestonesAPI | None = None
        self._runs: ResearchProjectsRunsAPI | None = None

    @property
    def setup(self) -> ResearchProjectsSetupAPI:
        if self._setup is None:
            self._setup = ResearchProjectsSetupAPI(self._session)
        return self._setup

    @property
    def workspace(self) -> ResearchProjectsWorkspaceAPI:
        if self._workspace is None:
            self._workspace = ResearchProjectsWorkspaceAPI(self._session)
        return self._workspace

    @property
    def repos(self) -> ResearchProjectsReposAPI:
        if self._repos is None:
            self._repos = ResearchProjectsReposAPI(self._session)
        return self._repos

    @property
    def git(self) -> ResearchProjectsGitAPI:
        if self._git is None:
            self._git = ResearchProjectsGitAPI(self._session)
        return self._git

    @property
    def code(self) -> ResearchProjectsCodeAPI:
        if self._code is None:
            self._code = ResearchProjectsCodeAPI(self._session)
        return self._code

    @property
    def objectives(self) -> ResearchProjectsObjectivesAPI:
        if self._objectives is None:
            self._objectives = ResearchProjectsObjectivesAPI(self._session)
        return self._objectives

    @property
    def milestones(self) -> ResearchProjectsMilestonesAPI:
        if self._milestones is None:
            self._milestones = ResearchProjectsMilestonesAPI(self._session)
        return self._milestones

    @property
    def runs(self) -> ResearchProjectsRunsAPI:
        if self._runs is None:
            self._runs = ResearchProjectsRunsAPI(self._session)
        return self._runs

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

    def update(
        self,
        project_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._session.projects.patch(project_id, dict(payload), **kwargs)

    def archive(self, project_id: str) -> dict[str, Any]:
        return self._session.projects.archive(project_id)

    def unarchive(self, project_id: str) -> dict[str, Any]:
        return self._session.projects.unarchive(project_id)

    def setup_state(self, project_id: str) -> SmrProjectSetup:
        warnings.warn(
            "projects.setup_state is deprecated; use projects.setup.get instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.setup.get(project_id)

    def prepare_setup(self, project_id: str) -> SmrProjectSetup:
        warnings.warn(
            "projects.prepare_setup is deprecated; use projects.setup.prepare instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.setup.prepare(project_id)

    def start_onboarding(self, project_id: str) -> dict[str, Any]:
        return self.setup.start_onboarding(project_id)

    def complete_onboarding_step(
        self,
        project_id: str,
        *,
        step: str,
        status: str,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.setup.complete_onboarding_step(
            project_id,
            step=step,
            status=status,
            detail=detail,
        )

    def dry_run_onboarding(self, project_id: str) -> dict[str, Any]:
        return self.setup.dry_run_onboarding(project_id)

    def onboarding_status(self, project_id: str) -> dict[str, Any]:
        return self.setup.onboarding_status(project_id)


__all__ = ["ResearchProjectsAPI"]
