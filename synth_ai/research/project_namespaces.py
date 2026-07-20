"""Nested project namespaces on ``client.research.projects``."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
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


def _require_git_mapping(payload: object, *, label: str) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} payload must be an object")
    return {str(key): value for key, value in payload.items()}


@dataclass(frozen=True)
class GitCommitRow:
    """One commit summary row from the project git server."""

    sha: str
    summary: str = ""
    author_name: str = ""
    author_email: str = ""
    authored_at: str | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> GitCommitRow:
        """Build a typed commit summary from a git-server response."""
        mapping = _require_git_mapping(payload, label="git commit")
        authored_at = mapping.get("authored_at")
        return cls(
            sha=str(mapping.get("sha") or ""),
            summary=str(mapping.get("summary") or ""),
            author_name=str(mapping.get("author_name") or ""),
            author_email=str(mapping.get("author_email") or ""),
            authored_at=str(authored_at) if authored_at is not None else None,
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class GitBranchRow:
    """One unmerged-branch summary row from the project git server."""

    name: str
    head_commit_sha: str = ""
    last_authored_at: str | None = None
    author_name: str | None = None
    summary: str | None = None
    merged_into_default: bool | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> GitBranchRow:
        """Build a typed branch summary from a git-server response."""
        mapping = _require_git_mapping(payload, label="git branch")

        def _opt_str(key: str) -> str | None:
            value = mapping.get(key)
            return str(value) if value is not None else None

        merged = mapping.get("merged_into_default")
        return cls(
            name=str(mapping.get("name") or ""),
            head_commit_sha=str(mapping.get("head_commit_sha") or ""),
            last_authored_at=_opt_str("last_authored_at"),
            author_name=_opt_str("author_name"),
            summary=_opt_str("summary"),
            merged_into_default=bool(merged) if merged is not None else None,
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class GitRepoStatus:
    """Project git repo status: HEAD, recent commits, unmerged branches, tree."""

    project_id: str
    branch: str = ""
    default_branch: str = ""
    head_commit_sha: str = ""
    recent_commits: tuple[GitCommitRow, ...] = ()
    unmerged_branches: tuple[GitBranchRow, ...] = ()
    tree_paths: tuple[str, ...] = ()
    tree_truncated: bool = False
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> GitRepoStatus:
        """Build a typed repository status projection from a git-server response."""
        mapping = _require_git_mapping(payload, label="git status")
        return cls(
            project_id=str(mapping.get("project_id") or ""),
            branch=str(mapping.get("branch") or ""),
            default_branch=str(mapping.get("default_branch") or ""),
            head_commit_sha=str(mapping.get("head_commit_sha") or ""),
            recent_commits=tuple(
                GitCommitRow.from_wire(item) for item in list(mapping.get("recent_commits") or [])
            ),
            unmerged_branches=tuple(
                GitBranchRow.from_wire(item)
                for item in list(mapping.get("unmerged_branches") or [])
            ),
            tree_paths=tuple(str(item) for item in list(mapping.get("tree_paths") or [])),
            tree_truncated=bool(mapping.get("tree_truncated")),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class GitFileContent:
    """One file read from the project git server (utf-8 or base64 content)."""

    project_id: str
    ref: str
    path: str
    content: str = ""
    encoding: str = "utf-8"
    size_bytes: int = 0
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> GitFileContent:
        mapping = _require_git_mapping(payload, label="git file")
        return cls(
            project_id=str(mapping.get("project_id") or ""),
            ref=str(mapping.get("ref") or ""),
            path=str(mapping.get("path") or ""),
            content=str(mapping.get("content") or ""),
            encoding=str(mapping.get("encoding") or "utf-8"),
            size_bytes=int(mapping.get("size_bytes") or 0),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class GitPullRequestRow:
    """One GitHub pull-request row (raw GitHub payload preserved in ``raw``)."""

    number: int
    title: str = ""
    state: str = ""
    html_url: str | None = None
    user_login: str | None = None
    head_ref: str | None = None
    base_ref: str | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> GitPullRequestRow:
        mapping = _require_git_mapping(payload, label="pull request")
        user = mapping.get("user")
        head = mapping.get("head")
        base = mapping.get("base")

        def _ref(value: object) -> str | None:
            if isinstance(value, Mapping):
                ref = value.get("ref")
                return str(ref) if ref is not None else None
            return None

        html_url = mapping.get("html_url")
        return cls(
            number=int(mapping.get("number") or 0),
            title=str(mapping.get("title") or ""),
            state=str(mapping.get("state") or ""),
            html_url=str(html_url) if html_url is not None else None,
            user_login=(
                str(user.get("login")) if isinstance(user, Mapping) and user.get("login") else None
            ),
            head_ref=_ref(head),
            base_ref=_ref(base),
            raw=dict(mapping),
        )


class ResearchProjectsGitPullRequestsAPI:
    """Read pull requests on the project's bound GitHub repo."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def list(
        self,
        project_id: str,
        *,
        repo: str | None = None,
        state: str = "open",
        limit: int = 30,
    ) -> tuple[GitPullRequestRow, ...]:
        """List pull requests (``state`` in open/closed/all, ``limit`` 1-100).

        Backend route: ``GET /smr/projects/{project_id}/git/pull-requests``.
        """
        params: dict[str, Any] = {"state": str(state), "limit": int(limit)}
        if repo is not None:
            params["repo"] = str(repo)
        payload = _require_git_mapping(
            self._session._request_json(
                "GET",
                f"/smr/projects/{project_id}/git/pull-requests",
                params=params,
            ),
            label="pull requests",
        )
        return tuple(
            GitPullRequestRow.from_wire(item) for item in list(payload.get("pull_requests") or [])
        )


class ResearchProjectsGitAPI:
    """Project git source connection, metadata, and read-only git-server views.

    Read bindings cover the mounted GET routes only: status, tree, file, diff,
    and pull-request listing. Branch/commit creation, push, file writes, and PR
    create/comment are POST routes and are intentionally not bound here.
    """

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session
        self._pull_requests: ResearchProjectsGitPullRequestsAPI | None = None

    @property
    def pull_requests(self) -> ResearchProjectsGitPullRequestsAPI:
        """Read pull requests on the project's bound repo."""
        if self._pull_requests is None:
            self._pull_requests = ResearchProjectsGitPullRequestsAPI(self._session)
        return self._pull_requests

    def status(
        self,
        project_id: str,
        *,
        branch: str | None = None,
        max_commits: int = 20,
        max_tree_entries: int = 200,
        max_unmerged_branches: int = 20,
    ) -> GitRepoStatus:
        """Read repo status: HEAD, recent commits, unmerged branches, tree.

        Backend route: ``GET /smr/projects/{project_id}/git/status``.
        """
        params: dict[str, Any] = {
            "max_commits": int(max_commits),
            "max_tree_entries": int(max_tree_entries),
            "max_unmerged_branches": int(max_unmerged_branches),
        }
        if branch is not None:
            params["branch"] = str(branch)
        payload = self._session._request_json(
            "GET",
            f"/smr/projects/{project_id}/git/status",
            params=params,
        )
        return GitRepoStatus.from_wire(payload)

    def tree(
        self,
        project_id: str,
        *,
        ref: str | None = None,
        path_prefix: str | None = None,
    ) -> tuple[str, ...]:
        """List tree entry paths at a ref, optionally under a path prefix.

        Backend route: ``GET /smr/projects/{project_id}/git/tree``.
        """
        params: dict[str, Any] = {}
        if ref is not None:
            params["ref"] = str(ref)
        if path_prefix is not None:
            params["path_prefix"] = str(path_prefix)
        payload = _require_git_mapping(
            self._session._request_json(
                "GET",
                f"/smr/projects/{project_id}/git/tree",
                params=params or None,
            ),
            label="git tree",
        )
        return tuple(str(item) for item in list(payload.get("entries") or []))

    def file(
        self,
        project_id: str,
        path: str,
        *,
        ref: str | None = None,
    ) -> GitFileContent:
        """Read one file blob at a ref.

        Backend route: ``GET /smr/projects/{project_id}/git/file``.
        """
        params: dict[str, Any] = {"path": str(path)}
        if ref is not None:
            params["ref"] = str(ref)
        payload = self._session._request_json(
            "GET",
            f"/smr/projects/{project_id}/git/file",
            params=params,
        )
        return GitFileContent.from_wire(payload)

    def diff(
        self,
        project_id: str,
        *,
        base_ref: str,
        head_ref: str,
        path: str | None = None,
    ) -> dict[str, Any]:
        """Read the diff between two refs (optionally scoped to one path).

        Backend route: ``GET /smr/projects/{project_id}/git/diff``.
        """
        params: dict[str, Any] = {"base_ref": str(base_ref), "head_ref": str(head_ref)}
        if path is not None:
            params["path"] = str(path)
        return dict(
            _require_git_mapping(
                self._session._request_json(
                    "GET",
                    f"/smr/projects/{project_id}/git/diff",
                    params=params,
                ),
                label="git diff",
            )
        )

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
    "GitBranchRow",
    "GitCommitRow",
    "GitFileContent",
    "GitPullRequestRow",
    "GitRepoStatus",
    "ResearchProjectsCodeAPI",
    "ResearchProjectsGitAPI",
    "ResearchProjectsGitPullRequestsAPI",
    "ResearchProjectsMilestonesAPI",
    "ResearchProjectsObjectivesAPI",
    "ResearchProjectsReposAPI",
    "ResearchProjectsRunsAPI",
    "ResearchProjectsSetupAPI",
    "ResearchProjectsWorkspaceAPI",
]
