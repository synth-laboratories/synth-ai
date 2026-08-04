"""Durable files attached to a Research project.

# See: openapi/research-v1.json
"""

from __future__ import annotations

from dataclasses import dataclass

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.sdk.research.contracts._wire import array_value, object_value, required_text
from synth_ai.sdk.research.contracts.common import ProjectId
from synth_ai.sdk.research.contracts.workspaces import (
    WorkspaceFilesUploadRequest,
    WorkspaceFileUpload,
    WorkspaceStoredFile,
)
from synth_ai.sdk.research.operations import research_operation

ProjectFile = WorkspaceStoredFile
ProjectFileUpload = WorkspaceFileUpload
ProjectFilesUploadRequest = WorkspaceFilesUploadRequest


def _request(
    operation_id: str,
    path: str,
    *,
    query: JsonObject | None = None,
    body: JsonObject | None = None,
) -> HttpRequest:
    return HttpRequest(
        research_operation(operation_id),
        path,
        query=query or {},
        body=body,
    )


def _non_negative_int(payload: JsonObject, name: str) -> int:
    value = payload.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ProjectFilesPage:
    """One cursor page of files that remain attached to a Project."""

    files: tuple[ProjectFile, ...]
    next_cursor: str | None

    @classmethod
    def from_wire(cls, value: JsonValue) -> ProjectFilesPage:
        payload = object_value(value, operation_id="list project files")
        raw_cursor = payload.get("next_cursor")
        if raw_cursor is not None and not isinstance(raw_cursor, str):
            raise ValueError("next_cursor must be a string when provided")
        return cls(
            files=tuple(
                ProjectFile.from_wire(item)
                for item in array_value(payload.get("files"), operation_id="project files")
            ),
            next_cursor=raw_cursor,
        )


@dataclass(frozen=True, slots=True)
class ProjectFilesUploadReceipt:
    """Receipt for files attached to a Project through the canonical file route."""

    project_id: ProjectId
    file_count: int
    bytes_uploaded: int
    uploaded_files: tuple[ProjectFile, ...]

    @classmethod
    def from_wire(cls, value: JsonValue) -> ProjectFilesUploadReceipt:
        payload = object_value(value, operation_id="upload project files")
        project_id = ProjectId(required_text(payload, "project_id"))
        files = tuple(
            ProjectFile.from_wire(item)
            for item in array_value(payload.get("uploaded_files"), operation_id="uploaded files")
        )
        file_count = _non_negative_int(payload, "file_count")
        if file_count != len(files):
            raise ValueError("project file upload count does not match uploaded files")
        if any(item.project_id != project_id for item in files):
            raise ValueError("project file upload response crossed its project boundary")
        return cls(
            project_id=project_id,
            file_count=file_count,
            bytes_uploaded=_non_negative_int(payload, "bytes_uploaded"),
            uploaded_files=files,
        )


class ProjectFilesAPI:
    """List and attach durable Project files at any point in its lifecycle."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(
        self,
        project_id: ProjectId,
        *,
        visibility: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> ProjectFilesPage:
        query: JsonObject = {"limit": limit}
        if visibility is not None:
            query["visibility"] = visibility
        if cursor is not None:
            query["cursor"] = cursor
        value = self._transport.execute(
            _request("list_project_files", f"/smr/projects/{project_id}/files", query=query)
        )
        return ProjectFilesPage.from_wire(value)

    def upload(
        self,
        project_id: ProjectId,
        request: ProjectFilesUploadRequest,
    ) -> ProjectFilesUploadReceipt:
        value = self._transport.execute(
            _request(
                "upload_project_files",
                f"/smr/projects/{project_id}/files",
                body=request.to_wire(),
            )
        )
        return ProjectFilesUploadReceipt.from_wire(value)


class AsyncProjectFilesAPI:
    """Native-async peer of :class:`ProjectFilesAPI`."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def list(
        self,
        project_id: ProjectId,
        *,
        visibility: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> ProjectFilesPage:
        query: JsonObject = {"limit": limit}
        if visibility is not None:
            query["visibility"] = visibility
        if cursor is not None:
            query["cursor"] = cursor
        value = await self._transport.execute(
            _request("list_project_files", f"/smr/projects/{project_id}/files", query=query)
        )
        return ProjectFilesPage.from_wire(value)

    async def upload(
        self,
        project_id: ProjectId,
        request: ProjectFilesUploadRequest,
    ) -> ProjectFilesUploadReceipt:
        value = await self._transport.execute(
            _request(
                "upload_project_files",
                f"/smr/projects/{project_id}/files",
                body=request.to_wire(),
            )
        )
        return ProjectFilesUploadReceipt.from_wire(value)


__all__ = [
    "AsyncProjectFilesAPI",
    "ProjectFile",
    "ProjectFileUpload",
    "ProjectFilesAPI",
    "ProjectFilesPage",
    "ProjectFilesUploadReceipt",
    "ProjectFilesUploadRequest",
]
