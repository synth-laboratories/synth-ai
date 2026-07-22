"""Project-owned workspace-input operations over the shared core transport."""

from __future__ import annotations

from typing import Optional

from synth_ai.core.contracts.json_value import JsonObject
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.research.contracts.common import ProjectId
from synth_ai.core.research.contracts.workspaces import (
    ProjectWorkspaceInputs,
    WorkspaceFilesUploadReceipt,
    WorkspaceFilesUploadRequest,
    WorkspaceSourceRepositoryReceipt,
    WorkspaceSourceRepositorySpec,
)
from synth_ai.core.research.operations import research_operation


def _request(
    operation_id: str,
    path: str,
    *,
    body: Optional[JsonObject] = None,
) -> HttpRequest:
    return HttpRequest(
        research_operation(operation_id),
        path,
        body=body,
    )


class ProjectWorkspaceAPI:
    """Typed workspace bootstrap inputs for one Research project."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def retrieve(self, project_id: ProjectId) -> ProjectWorkspaceInputs:
        value = self._transport.execute(
            _request(
                "retrieve_project_workspace_inputs",
                f"/smr/projects/{project_id}/workspace-inputs",
            )
        )
        return ProjectWorkspaceInputs.from_wire(value)

    def set_source_repository(
        self,
        project_id: ProjectId,
        request: WorkspaceSourceRepositorySpec,
    ) -> WorkspaceSourceRepositoryReceipt:
        value = self._transport.execute(
            _request(
                "set_project_workspace_source_repository",
                f"/smr/projects/{project_id}/workspace-inputs/source-repo",
                body=request.to_wire(),
            )
        )
        return WorkspaceSourceRepositoryReceipt.from_wire(value)

    def upload_files(
        self,
        project_id: ProjectId,
        request: WorkspaceFilesUploadRequest,
    ) -> WorkspaceFilesUploadReceipt:
        value = self._transport.execute(
            _request(
                "upload_project_workspace_files",
                f"/smr/projects/{project_id}/workspace-inputs/files:upload",
                body=request.to_wire(),
            )
        )
        return WorkspaceFilesUploadReceipt.from_wire(value)


class AsyncProjectWorkspaceAPI:
    """Native-async peer of :class:`ProjectWorkspaceAPI`."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def retrieve(self, project_id: ProjectId) -> ProjectWorkspaceInputs:
        value = await self._transport.execute(
            _request(
                "retrieve_project_workspace_inputs",
                f"/smr/projects/{project_id}/workspace-inputs",
            )
        )
        return ProjectWorkspaceInputs.from_wire(value)

    async def set_source_repository(
        self,
        project_id: ProjectId,
        request: WorkspaceSourceRepositorySpec,
    ) -> WorkspaceSourceRepositoryReceipt:
        value = await self._transport.execute(
            _request(
                "set_project_workspace_source_repository",
                f"/smr/projects/{project_id}/workspace-inputs/source-repo",
                body=request.to_wire(),
            )
        )
        return WorkspaceSourceRepositoryReceipt.from_wire(value)

    async def upload_files(
        self,
        project_id: ProjectId,
        request: WorkspaceFilesUploadRequest,
    ) -> WorkspaceFilesUploadReceipt:
        value = await self._transport.execute(
            _request(
                "upload_project_workspace_files",
                f"/smr/projects/{project_id}/workspace-inputs/files:upload",
                body=request.to_wire(),
            )
        )
        return WorkspaceFilesUploadReceipt.from_wire(value)


__all__ = ["AsyncProjectWorkspaceAPI", "ProjectWorkspaceAPI"]
