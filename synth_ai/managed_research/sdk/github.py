"""Org-scoped GitHub namespace for the flatter noun-first SDK surface."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.sdk._base import _ClientNamespace


class GithubAPI(_ClientNamespace):
    def status(self) -> dict[str, Any]:
        return self._client.get_github_status()

    def start_oauth(
        self,
        *,
        redirect_uri: str | None = None,
    ) -> dict[str, Any]:
        return self._client.start_github_oauth(redirect_uri=redirect_uri)

    def list_repos(
        self,
        *,
        page: int | None = None,
        per_page: int | None = None,
    ) -> list[dict[str, Any]]:
        return self._client.list_github_repos(page=page, per_page=per_page)

    def disconnect(self) -> dict[str, Any]:
        return self._client.disconnect_github()


__all__ = ["GithubAPI"]
