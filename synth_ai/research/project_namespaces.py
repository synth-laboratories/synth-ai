"""Nested project namespaces on ``client.research.projects``."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from os import PathLike
from typing import Any, List

from synth_ai.managed_research.models.project_workspace import ProjectWorkspaceProjection
from synth_ai.managed_research.models.types import (
    SmrProjectSetup,
    WorkspaceInputsState,
    WorkspaceUploadResult,
)
from synth_ai.managed_research.sdk.client import ManagedResearchClient


class ResearchProjectsSetupAPI:
    """Prepare projects for launch (onboarding + runnable setup)."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def get(self, project_id: str) -> SmrProjectSetup:
        """Return current setup/onboarding state for a project."""
        return self._session.setup.get(project_id)

    def prepare(self, project_id: str) -> SmrProjectSetup:
        """Run setup steps required before ``runs.check_preflight`` succeeds."""
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


class ResearchProjectsWorkspaceAPI:
    """Upload and download project workspace inputs and archives."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def get(self, project_id: str) -> ProjectWorkspaceProjection:
        return self._session.projects.get_workspace(project_id)

    def upload(
        self,
        project_id: str,
        files: Iterable[Mapping[str, object]],
    ) -> WorkspaceUploadResult:
        return self._session.workspace_inputs.upload_files(project_id, files)

    def upload_directory(
        self,
        project_id: str,
        directory: str | PathLike[str],
    ) -> WorkspaceUploadResult:
        return self._session.workspace_inputs.upload_directory(project_id, directory)

    def download(
        self,
        project_id: str,
        destination: str,
    ) -> dict[str, Any]:
        return self._session.projects.download_workspace_archive(project_id, destination)

    def inputs(self, project_id: str) -> WorkspaceInputsState:
        return self._session.workspace_inputs.get(project_id)

    def upload_url(self, project_id: str) -> dict[str, Any]:
        return self._session.get_workspace_upload_url(project_id)

    def confirm_push(
        self,
        project_id: str,
        *,
        commit_sha: str,
        archive_key: str,
    ) -> dict[str, Any]:
        return self._session.workspace_confirm_push(
            project_id,
            commit_sha=commit_sha,
            archive_key=archive_key,
        )


class ResearchProjectsReposAPI:
    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def attach(
        self,
        project_id: str,
        url: str,
        *,
        default_branch: str | None = None,
        commit_sha: str | None = None,
    ) -> dict[str, Any]:
        return self._session.workspace_inputs.attach_source_repo(
            project_id,
            url,
            default_branch=default_branch,
            commit_sha=commit_sha,
        )


class ResearchProjectsGitAPI:
    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def get(self, project_id: str) -> dict[str, Any]:
        return self._session.projects.get_git(project_id)

    def connect(
        self,
        project_id: str,
        *,
        provider: str | None = None,
        repo_url: str | None = None,
        branch: str | None = None,
        auth_ref: str | None = None,
        sync_policy: Mapping[str, Any] | None = None,
    ) -> Any:
        return self._session.projects.connect_git_source(
            project_id,
            provider=provider,
            repo_url=repo_url,
            branch=branch,
            auth_ref=auth_ref,
            sync_policy=sync_policy,
        )


class ResearchProjectsCodeAPI:
    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def download(
        self,
        project_id: str,
        destination: str,
    ) -> dict[str, Any]:
        return self._session.projects.download_code(project_id, destination)


class ResearchProjectsObjectivesAPI:
    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def list_directed_effort_outcomes(
        self,
        project_id: str,
        *,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        return self._session.list_directed_effort_outcomes(
            project_id,
            run_id=run_id,
            limit=limit,
        )

    def list(
        self,
        project_id: str,
        *,
        kind: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        return self._session.list_objectives(
            project_id,
            kind=kind,
            run_id=run_id,
            limit=limit,
        )

    def get_status(
        self,
        project_id: str,
        objective_id: str,
        *,
        kind: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._session.get_objective_status(
            project_id,
            objective_id,
            kind=kind,
            **kwargs,
        )

    def get_progress(
        self,
        project_id: str,
        objective_id: str,
        *,
        kind: str | None = None,
    ) -> dict[str, Any]:
        return self._session.get_objective_progress(
            project_id,
            objective_id,
            kind=kind,
        )


class ResearchProjectsMilestonesAPI:
    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def list(
        self,
        project_id: str,
        *,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        return self._session.list_milestones(
            project_id,
            run_id=run_id,
            limit=limit,
        )


class ResearchProjectsRunsAPI:
    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def list(
        self,
        project_id: str,
        *,
        active_only: bool = False,
        **kwargs: Any,
    ) -> List[dict[str, Any]]:
        return self._session.runs.list(project_id, active_only=active_only, **kwargs)


__all__ = [
    "ResearchProjectsCodeAPI",
    "ResearchProjectsGitAPI",
    "ResearchProjectsMilestonesAPI",
    "ResearchProjectsObjectivesAPI",
    "ResearchProjectsReposAPI",
    "ResearchProjectsRunsAPI",
    "ResearchProjectsSetupAPI",
    "ResearchProjectsWorkspaceAPI",
]
