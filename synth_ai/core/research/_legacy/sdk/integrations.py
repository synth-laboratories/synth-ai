"""Integration-oriented SDK namespace."""

from __future__ import annotations

from synth_ai.core.research._legacy.models.local_execution_profile import LocalPublicationReadiness
from synth_ai.core.research._legacy.sdk._base import _ClientNamespace


class IntegrationsAPI(_ClientNamespace):
    """Integration helpers for first-party local and hosted flows."""

    def register_local_github_credential(
        self,
        project_id: str,
        *,
        repo: str,
        access_token: str,
        pr_write_enabled: bool = True,
    ) -> dict[str, object]:
        return self._client.register_local_github_credential(
            project_id,
            repo=repo,
            access_token=access_token,
            pr_write_enabled=pr_write_enabled,
        )

    def register_local_github_repo_credential(
        self,
        *,
        repo: str,
        access_token: str,
        pr_write_enabled: bool = True,
    ) -> dict[str, object]:
        return self._client.register_local_github_repo_credential(
            repo=repo,
            access_token=access_token,
            pr_write_enabled=pr_write_enabled,
        )

    def get_local_publication_readiness(
        self,
        project_id: str,
        *,
        repo: str | None = None,
        pr_write_enabled: bool = True,
    ) -> LocalPublicationReadiness:
        return self._client.get_local_publication_readiness(
            project_id,
            repo=repo,
            pr_write_enabled=pr_write_enabled,
        )


__all__ = ["IntegrationsAPI"]
