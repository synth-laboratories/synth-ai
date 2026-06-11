"""Project repo namespace for the flatter noun-first SDK surface."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from synth_ai.managed_research.errors import UnsupportedProvider
from synth_ai.managed_research.models.types import Repository
from synth_ai.managed_research.sdk._base import _ClientNamespace


def _github_repo_name(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"}:
        raise UnsupportedProvider(
            "Only unauthenticated GitHub http(s) repositories are supported in v1."
        )
    host = parsed.hostname or ""
    if host.lower() != "github.com":
        raise UnsupportedProvider("Only GitHub repositories are supported in v1.")
    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        raise ValueError("GitHub repository URL must include owner and repo.")
    return "/".join(parts[:2])


class ReposAPI(_ClientNamespace):
    def list(self, project_id: str) -> list[Repository]:
        return [
            Repository.from_wire(item)
            for item in self._client.list_project_external_repositories(project_id)
        ]

    def attach(
        self,
        project_id: str,
        *,
        url: str | None = None,
        role: str = "dependency",
        branch: str | None = None,
        github_repo: str | None = None,
        visibility: str = "public",
        metadata: dict[str, Any] | None = None,
    ) -> Repository:
        repo_url = url or (f"https://github.com/{github_repo}" if github_repo else None)
        if not repo_url:
            raise ValueError("url is required")
        name = _github_repo_name(repo_url)
        payload = {"provider": "github", "visibility": visibility, **dict(metadata or {})}
        return Repository.from_wire(
            self._client.create_project_external_repository(
                project_id,
                name=name,
                url=repo_url,
                default_branch=branch,
                role=role,
                metadata=payload,
            )
        )

    def detach(self, project_id: str, *, repo_id: str) -> dict[str, Any]:
        return self._client.delete_project_external_repository(project_id, repo_id)

    def detach_binding(self, project_id: str, *, github_repo: str) -> dict[str, Any]:
        return self._client.detach_project_repo(project_id, repo=github_repo)

    def list_bindings(self, project_id: str) -> list[dict[str, Any]]:
        return self._client.list_project_repo_bindings(project_id)


__all__ = ["ReposAPI"]
