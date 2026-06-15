"""Workspace-input SDK namespace."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from os import PathLike
from typing import Any

from synth_ai.managed_research.models.types import WorkspaceInputsState, WorkspaceUploadResult
from synth_ai.managed_research.sdk._base import _ClientNamespace


class WorkspaceInputsAPI(_ClientNamespace):
    def attach_source_repo(
        self,
        project_id: str,
        url: str,
        *,
        default_branch: str | None = None,
    ) -> dict[str, Any]:
        return self._client.attach_source_repo(project_id, url, default_branch=default_branch)

    def get(self, project_id: str) -> WorkspaceInputsState:
        return WorkspaceInputsState.from_wire(self._client.get_workspace_inputs(project_id))

    def upload_files(
        self,
        project_id: str,
        files: Iterable[Mapping[str, object]],
    ) -> WorkspaceUploadResult:
        return WorkspaceUploadResult.from_wire(
            self._client.upload_workspace_files(project_id, files)
        )

    def upload_directory(
        self,
        project_id: str,
        directory: str | PathLike[str],
    ) -> WorkspaceUploadResult:
        return WorkspaceUploadResult.from_wire(
            self._client.upload_workspace_directory(project_id, directory)
        )

    def upload_source_bundle(
        self,
        project_id: str,
        bundle_path: str | PathLike[str],
        *,
        path: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WorkspaceUploadResult:
        return WorkspaceUploadResult.from_wire(
            self._client.upload_workspace_source_bundle(
                project_id,
                bundle_path,
                path=path,
                metadata=metadata,
            )
        )


__all__ = ["WorkspaceInputsAPI"]
