"""External-repository SDK namespace for Phase 3 resource surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synth_ai.managed_research.models.types import ExternalRepository, RunRepositoryMount
from synth_ai.managed_research.sdk._base import _ClientNamespace


class RepositoriesAPI(_ClientNamespace):
    def list_project(self, project_id: str) -> list[ExternalRepository]:
        return [
            ExternalRepository.from_wire(item)
            for item in self._client.list_project_external_repositories(project_id)
        ]

    def create_project(
        self,
        project_id: str,
        *,
        name: str,
        url: str,
        default_branch: str | None = None,
        role: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExternalRepository:
        return ExternalRepository.from_wire(
            self._client.create_project_external_repository(
                project_id,
                name=name,
                url=url,
                default_branch=default_branch,
                role=role,
                metadata=metadata,
            )
        )

    def patch_project(
        self,
        project_id: str,
        repository_id: str,
        *,
        url: str | None = None,
        default_branch: str | None = None,
        role: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExternalRepository:
        return ExternalRepository.from_wire(
            self._client.patch_project_external_repository(
                project_id,
                repository_id,
                url=url,
                default_branch=default_branch,
                role=role,
                metadata=metadata,
            )
        )

    def list_run_mounts(self, run_id: str) -> list[RunRepositoryMount]:
        return [
            RunRepositoryMount.from_wire(item)
            for item in self._client.list_run_repository_mounts(run_id)
        ]

    def create_run_mount(
        self,
        run_id: str,
        *,
        repository_id: str,
        mount_name: str | None = None,
    ) -> RunRepositoryMount:
        return RunRepositoryMount.from_wire(
            self._client.create_run_repository_mount(
                run_id,
                repository_id=repository_id,
                mount_name=mount_name,
            )
        )


__all__ = ["RepositoriesAPI"]
