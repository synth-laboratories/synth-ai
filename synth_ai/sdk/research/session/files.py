"""File-oriented SDK namespace for Phase 3 resource surfaces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from os import PathLike
from typing import Any, Literal

from synth_ai.sdk.research.contracts.types import (
    ResourceUploadResult,
    RunFileMount,
    RunOutputFile,
    StoredFile,
)
from synth_ai.sdk.research.session._base import _ClientNamespace


class FilesAPI(_ClientNamespace):
    def create_org(
        self,
        *,
        name: str,
        content: str,
        encoding: Literal["utf-8", "base64"] = "utf-8",
        content_type: str | None = None,
        sync_session_id: str | None = None,
        project_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an org-owned file, optionally bound to a session or project.

        # See: backend/app/api/v1/managed_research/files.py::SmrFileCreateRequest
        Unbound files retain a null project_id; they are not project StoredFiles.
        """
        if not name.strip() or len(name) > 512:
            raise ValueError("name must contain 1 through 512 characters")
        if encoding not in {"utf-8", "base64"}:
            raise ValueError("encoding must be utf-8 or base64")
        return self._client.create_org_file(
            name=name,
            content=content,
            encoding=encoding,
            content_type=content_type,
            sync_session_id=sync_session_id,
            project_id=project_id,
            metadata=dict(metadata or {}),
        )

    def list_project(
        self,
        project_id: str,
        *,
        visibility: str | None = None,
        limit: int | None = None,
    ) -> list[StoredFile]:
        return [
            StoredFile.from_wire(item)
            for item in self._client.list_project_files(
                project_id,
                visibility=visibility,
                limit=limit,
            )
        ]

    def create_project(
        self,
        project_id: str,
        files: Iterable[Mapping[str, Any]],
    ) -> ResourceUploadResult:
        return ResourceUploadResult.from_wire(self._client.create_project_files(project_id, files))

    def create_project_source_bundle(
        self,
        project_id: str,
        bundle_path: str | PathLike[str],
        *,
        path: str | None = None,
        visibility: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ResourceUploadResult:
        return ResourceUploadResult.from_wire(
            self._client.create_project_source_bundle(
                project_id,
                bundle_path,
                path=path,
                visibility=visibility,
                metadata=metadata,
            )
        )

    def get_project(self, project_id: str, file_id: str) -> StoredFile:
        return StoredFile.from_wire(self._client.get_project_file(project_id, file_id))

    def get_content(self, file_id: str) -> dict[str, Any]:
        return self._client.get_file_content(file_id)

    def list_run_mounts(self, run_id: str) -> list[RunFileMount]:
        return [RunFileMount.from_wire(item) for item in self._client.list_run_file_mounts(run_id)]

    def upload_run(
        self,
        run_id: str,
        files: Iterable[Mapping[str, Any]],
    ) -> ResourceUploadResult:
        return ResourceUploadResult.from_wire(self._client.upload_run_files(run_id, files))

    def upload_run_source_bundle(
        self,
        run_id: str,
        bundle_path: str | PathLike[str],
        *,
        path: str | None = None,
        visibility: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ResourceUploadResult:
        return ResourceUploadResult.from_wire(
            self._client.upload_run_source_bundle(
                run_id,
                bundle_path,
                path=path,
                visibility=visibility,
                metadata=metadata,
            )
        )

    def list_outputs(
        self,
        run_id: str,
        *,
        artifact_type: str | None = None,
        limit: int | None = None,
    ) -> list[RunOutputFile]:
        return [
            RunOutputFile.from_wire(item)
            for item in self._client._list_run_output_files(
                run_id,
                artifact_type=artifact_type,
                limit=limit,
            )
        ]

    def get_output_content(
        self,
        run_id: str,
        output_file_id: str,
        *,
        disposition: str = "inline",
    ) -> dict[str, Any]:
        return self._client._get_run_output_file_content(
            run_id,
            output_file_id,
            disposition=disposition,
        )


__all__ = ["FilesAPI"]
