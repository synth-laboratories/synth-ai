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
        """Start the onboarding workflow for a new project."""
        return self._session.setup.start_onboarding(project_id)

    def complete_onboarding_step(
        self,
        project_id: str,
        *,
        step: str,
        status: str,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Mark an onboarding step complete or failed.

        Args:
            project_id: Project identifier.
            step: Onboarding step name.
            status: Step status (for example ``"complete"``).
            detail: Optional structured detail payload.
        """
        return self._session.setup.complete_onboarding_step(
            project_id,
            step=step,
            status=status,
            detail=detail,
        )

    def dry_run_onboarding(self, project_id: str) -> dict[str, Any]:
        """Validate onboarding prerequisites without mutating project state."""
        return self._session.setup.dry_run_onboarding(project_id)

    def onboarding_status(self, project_id: str) -> dict[str, Any]:
        """Return onboarding progress and blocking issues."""
        return self._session.setup.get_onboarding_status(project_id)


class ResearchProjectsWorkspaceAPI:
    """Upload and download project workspace inputs and archives."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def get(self, project_id: str) -> ProjectWorkspaceProjection:
        """Return the current workspace projection for a project."""
        return self._session.projects.get_workspace(project_id)

    def upload(
        self,
        project_id: str,
        files: Iterable[Mapping[str, object]],
    ) -> WorkspaceUploadResult:
        """Upload workspace files from in-memory file descriptors.

        Args:
            project_id: Project identifier.
            files: Iterable of file mappings (path, content, metadata).
        """
        return self._session.workspace_inputs.upload_files(project_id, files)

    def upload_directory(
        self,
        project_id: str,
        directory: str | PathLike[str],
    ) -> WorkspaceUploadResult:
        """Upload an entire local directory into the project workspace."""
        return self._session.workspace_inputs.upload_directory(project_id, directory)

    def download(
        self,
        project_id: str,
        destination: str,
    ) -> dict[str, Any]:
        """Download the project workspace archive to a local path."""
        return self._session.projects.download_workspace_archive(project_id, destination)

    def inputs(self, project_id: str) -> WorkspaceInputsState:
        """Return workspace input state (uploaded files, git linkage)."""
        return self._session.workspace_inputs.get(project_id)

    def upload_url(self, project_id: str) -> dict[str, Any]:
        """Mint a presigned URL for direct workspace archive upload."""
        return self._session.get_workspace_upload_url(project_id)

    def confirm_push(
        self,
        project_id: str,
        *,
        commit_sha: str,
        archive_key: str,
    ) -> dict[str, Any]:
        """Confirm a git push completed and attach the uploaded archive.

        Args:
            project_id: Project identifier.
            commit_sha: Git commit SHA for the pushed workspace.
            archive_key: Storage key returned from ``upload_url``.
        """
        return self._session.workspace_confirm_push(
            project_id,
            commit_sha=commit_sha,
            archive_key=archive_key,
        )


class ResearchProjectsReposAPI:
    """Attach external source repositories to a project workspace."""

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
        """Attach a git repository as a workspace input source.

        Args:
            project_id: Project identifier.
            url: Repository clone URL.
            default_branch: Branch to track when no commit is pinned.
            commit_sha: Optional pinned commit SHA.
        """
        return self._session.workspace_inputs.attach_source_repo(
            project_id,
            url,
            default_branch=default_branch,
            commit_sha=commit_sha,
        )


class ResearchProjectsGitAPI:
    """Project git source connection and metadata."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def get(self, project_id: str) -> dict[str, Any]:
        """Return git metadata for the project workspace."""
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
        """Connect or update the project's git code source."""
        return self._session.projects.connect_git_source(
            project_id,
            provider=provider,
            repo_url=repo_url,
            branch=branch,
            auth_ref=auth_ref,
            sync_policy=sync_policy,
        )


class ResearchProjectsCodeAPI:
    """Download project code archives."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def download(
        self,
        project_id: str,
        destination: str,
    ) -> dict[str, Any]:
        """Download the project code archive to a local path."""
        return self._session.projects.download_code(project_id, destination)


class ResearchProjectsObjectivesAPI:
    """Directed effort outcomes and objective status for a project."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def list_directed_effort_outcomes(
        self,
        project_id: str,
        *,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        """List directed effort outcomes (legacy alias for directed objectives)."""
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
        """List objectives for a project, optionally scoped to a run."""
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
        """Return status fields for a single objective."""
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
        """Return progress metrics for a single objective."""
        return self._session.get_objective_progress(
            project_id,
            objective_id,
            kind=kind,
        )


class ResearchProjectsMilestonesAPI:
    """Project and run-scoped milestones."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def list(
        self,
        project_id: str,
        *,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        """List milestones for a project, optionally filtered to a run."""
        return self._session.list_milestones(
            project_id,
            run_id=run_id,
            limit=limit,
        )


class ResearchProjectsRunsAPI:
    """List runs belonging to a project."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def list(
        self,
        project_id: str,
        *,
        active_only: bool = False,
        **kwargs: Any,
    ) -> List[dict[str, Any]]:
        """List runs for a project.

        Args:
            project_id: Project identifier.
            active_only: When ``True``, return only non-terminal runs.
        """
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
