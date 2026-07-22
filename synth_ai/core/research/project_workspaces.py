"""Project-owned workspace-input operations over the shared core transport."""

from __future__ import annotations

import hashlib
import json
from typing import Optional, cast

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.errors import (
    RetryDirective,
    SynthError,
    SynthErrorCategory,
    SynthErrorCode,
    SynthFailure,
)
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.research.contracts._wire import object_value
from synth_ai.core.research.contracts.common import ProjectId
from synth_ai.core.research.contracts.workspaces import (
    ProjectWorkspaceInputs,
    WorkspaceFilesBatchUploadProgress,
    WorkspaceFilesBatchUploadReceipt,
    WorkspaceFilesBatchUploadRequest,
    WorkspaceFilesUploadReceipt,
    WorkspaceFilesUploadRequest,
    WorkspaceFileUpload,
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


def _upload_idempotency_key(
    project_id: ProjectId,
    files: tuple[WorkspaceFileUpload, ...],
) -> str:
    canonical = json.dumps(
        {
            "project_id": str(project_id),
            "files": [item.to_wire() for item in files],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return f"workspace-upload-v1:{digest}"


def _idempotent_upload_body(
    project_id: ProjectId,
    request: WorkspaceFilesUploadRequest,
) -> JsonObject:
    payload = request.to_wire()
    payload["idempotency_key"] = _upload_idempotency_key(project_id, request.files)
    return payload


def _confirm_push_body(
    *,
    commit_sha: str,
    archive_key: str | None,
    run_id: str | None,
) -> JsonObject:
    normalized_commit_sha = commit_sha.strip()
    if not normalized_commit_sha:
        raise ValueError("commit_sha must be non-empty")
    payload: JsonObject = {"commit_sha": normalized_commit_sha}
    if archive_key is not None:
        normalized_archive_key = archive_key.strip()
        if not normalized_archive_key:
            raise ValueError("archive_key must be non-empty when provided")
        payload["archive_key"] = normalized_archive_key
    if run_id is not None:
        normalized_run_id = run_id.strip()
        if not normalized_run_id:
            raise ValueError("run_id must be non-empty when provided")
        payload["run_id"] = normalized_run_id
    return payload


class WorkspaceBatchUploadError(SynthError):
    """Fail-loud composite error carrying exact committed-batch progress."""

    def __init__(
        self,
        progress: WorkspaceFilesBatchUploadProgress,
        cause: Exception,
    ) -> None:
        self.progress = progress
        self.cause = cause
        failure = cause.failure if isinstance(cause, SynthError) else None
        if failure is None:
            failure = SynthFailure(
                code=SynthErrorCode("workspace_batch_upload_failed"),
                category=SynthErrorCategory.OPERATION,
                operation="upload_project_workspace_files",
                request_id=None,
                correlation_id=None,
                retry=RetryDirective(retryable=False),
                status=None,
                detail=(
                    "workspace batch upload failed at batch "
                    f"{progress.failed_batch_index + 1}/{progress.total_batch_count}"
                ),
            )
        super().__init__(failure.detail, failure=failure)


def _batch_failure(
    *,
    project_id: ProjectId,
    request: WorkspaceFilesBatchUploadRequest,
    partitions: tuple[tuple[WorkspaceFileUpload, ...], ...],
    completed: list[WorkspaceFilesUploadReceipt],
    failed_batch_index: int,
    cause: Exception,
) -> WorkspaceBatchUploadError:
    return WorkspaceBatchUploadError(
        WorkspaceFilesBatchUploadProgress(
            project_id=project_id,
            requested_file_count=len(request.files),
            total_batch_count=len(partitions),
            completed_batches=tuple(completed),
            failed_batch_index=failed_batch_index,
            failed_paths=tuple(item.path for item in partitions[failed_batch_index]),
        ),
        cause,
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

    def confirm_push(
        self,
        project_id: ProjectId,
        *,
        commit_sha: str,
        archive_key: str | None = None,
        run_id: str | None = None,
    ) -> JsonObject:
        """Confirm an already-pushed workspace commit through project authority."""
        value = self._transport.execute(
            _request(
                "confirm_project_workspace_push",
                f"/smr/projects/{project_id}/workspace/confirm-push",
                body=_confirm_push_body(
                    commit_sha=commit_sha,
                    archive_key=archive_key,
                    run_id=run_id,
                ),
            )
        )
        return object_value(
            cast(JsonValue, value),
            operation_id="confirm_project_workspace_push",
        )

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
                body=_idempotent_upload_body(project_id, request),
            )
        )
        return WorkspaceFilesUploadReceipt.from_wire(value)

    def upload_batches(
        self,
        project_id: ProjectId,
        request: WorkspaceFilesBatchUploadRequest,
    ) -> WorkspaceFilesBatchUploadReceipt:
        partitions = request.partitions()
        completed: list[WorkspaceFilesUploadReceipt] = []
        for batch_index, files in enumerate(partitions):
            batch = WorkspaceFilesUploadRequest(files=files)
            try:
                completed.append(self.upload_files(project_id, batch))
            except Exception as error:
                raise _batch_failure(
                    project_id=project_id,
                    request=request,
                    partitions=partitions,
                    completed=completed,
                    failed_batch_index=batch_index,
                    cause=error,
                ) from error
        return WorkspaceFilesBatchUploadReceipt(
            project_id=project_id,
            requested_file_count=len(request.files),
            batches=tuple(completed),
        )


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

    async def confirm_push(
        self,
        project_id: ProjectId,
        *,
        commit_sha: str,
        archive_key: str | None = None,
        run_id: str | None = None,
    ) -> JsonObject:
        """Confirm an already-pushed workspace commit through project authority."""
        value = await self._transport.execute(
            _request(
                "confirm_project_workspace_push",
                f"/smr/projects/{project_id}/workspace/confirm-push",
                body=_confirm_push_body(
                    commit_sha=commit_sha,
                    archive_key=archive_key,
                    run_id=run_id,
                ),
            )
        )
        return object_value(
            cast(JsonValue, value),
            operation_id="confirm_project_workspace_push",
        )

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
                body=_idempotent_upload_body(project_id, request),
            )
        )
        return WorkspaceFilesUploadReceipt.from_wire(value)

    async def upload_batches(
        self,
        project_id: ProjectId,
        request: WorkspaceFilesBatchUploadRequest,
    ) -> WorkspaceFilesBatchUploadReceipt:
        partitions = request.partitions()
        completed: list[WorkspaceFilesUploadReceipt] = []
        for batch_index, files in enumerate(partitions):
            batch = WorkspaceFilesUploadRequest(files=files)
            try:
                completed.append(await self.upload_files(project_id, batch))
            except Exception as error:
                raise _batch_failure(
                    project_id=project_id,
                    request=request,
                    partitions=partitions,
                    completed=completed,
                    failed_batch_index=batch_index,
                    cause=error,
                ) from error
        return WorkspaceFilesBatchUploadReceipt(
            project_id=project_id,
            requested_file_count=len(request.files),
            batches=tuple(completed),
        )


__all__ = [
    "AsyncProjectWorkspaceAPI",
    "ProjectWorkspaceAPI",
    "WorkspaceBatchUploadError",
]
