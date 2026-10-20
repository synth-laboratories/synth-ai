"""Authorized project file reads and explicit uploads. See SDK_0230_IMPLEMENTATION.md."""

from __future__ import annotations

from .deprecated import artifact_alias

from synth_ai.contracts import swarms as operations
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.parameters import identifier, page_parameters, verified
from synth_ai.options import ReadOptions
from synth_ai.types.base import Page, PageCursor

from .file_uploads import AsyncUploads, Uploads
from .types.files import File


class Files:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport
        self.uploads = Uploads(transport)

    @artifact_alias
    def list(
        self,
        project_id: str,
        *,
        limit: int,
        cursor: PageCursor | None = None,
        request: ReadOptions | None = None,
    ) -> Page[File]:
        """Read a bounded page; uploaded does not necessarily mean ready for input."""
        project_id = identifier(project_id)
        return self._transport.request(
            operations.FILES,
            Page[File],
            path=f"/swarms/projects/{project_id}/files",
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda result: verified(
                result,
                len(result.items) <= limit
                and all(item.project_id == project_id for item in result.items),
                "File page scope or bound mismatch",
            ),
        )

    @artifact_alias
    def get(self, project_id: str, file_id: str, *, request: ReadOptions | None = None) -> File:
        """Read the exact file; only a ready immutable reference is a launch input."""
        project_id, file_id = identifier(project_id), identifier(file_id)
        return self._transport.request(
            operations.FILE,
            File,
            path=f"/swarms/projects/{project_id}/files/{file_id}",
            options=request,
            validate=lambda result: verified(
                result,
                result.project_id == project_id and result.file_id == file_id,
                "File identity mismatch",
            ),
        )


class AsyncFiles:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport
        self.uploads = AsyncUploads(transport)

    @artifact_alias
    async def list(
        self,
        project_id: str,
        *,
        limit: int,
        cursor: PageCursor | None = None,
        request: ReadOptions | None = None,
    ) -> Page[File]:
        """Read a bounded page; uploaded does not necessarily mean ready for input."""
        project_id = identifier(project_id)
        return await self._transport.request(
            operations.FILES,
            Page[File],
            path=f"/swarms/projects/{project_id}/files",
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda result: verified(
                result,
                len(result.items) <= limit
                and all(item.project_id == project_id for item in result.items),
                "File page scope or bound mismatch",
            ),
        )

    @artifact_alias
    async def get(
        self, project_id: str, file_id: str, *, request: ReadOptions | None = None
    ) -> File:
        """Read the exact file; only a ready immutable reference is a launch input."""
        project_id, file_id = identifier(project_id), identifier(file_id)
        return await self._transport.request(
            operations.FILE,
            File,
            path=f"/swarms/projects/{project_id}/files/{file_id}",
            options=request,
            validate=lambda result: verified(
                result,
                result.project_id == project_id and result.file_id == file_id,
                "File identity mismatch",
            ),
        )
