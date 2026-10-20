"""Preparation/transfer/finalization are distinct owner operations. See SDK_0230_IMPLEMENTATION.md."""

from __future__ import annotations

from .deprecated import artifact_alias

import time

from synth_ai.contracts import swarms as operations
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.parameters import identifier, verified
from synth_ai.core.recovery import async_resolve_creation as recover_creation_async
from synth_ai.core.recovery import resolve_creation as recover_creation
from synth_ai.core.transfer import async_transfer
from synth_ai.core.transfer import transfer as transfer_bytes
from synth_ai.errors import RequestTimeoutError
from synth_ai.options import MutationOptions, ReadOptions, TransferOptions
from synth_ai.types.identity import MutationIdentity
from synth_ai.types.receipts import CreationResolution
from synth_ai.types.uploads import TransferReceipt

from .types.files import (
    FileManifest,
    FileSource,
    FileUploadReceipt,
    FileUploadStatus,
)


class UploadSession:
    def __init__(self, transport: HttpTransport, snapshot: FileUploadStatus) -> None:
        self._transport = transport
        self.upload_id = snapshot.upload_id
        self._project_id = snapshot.project_id
        self._manifest = snapshot.manifest

    @artifact_alias
    def status(self, *, request: ReadOptions | None = None) -> FileUploadStatus:
        """Read fresh state; this does not cancel, restart or finalize work.

        See: SDK_0230_IMPLEMENTATION.md."""
        result = self._transport.request(
            operations.FILE_UPLOAD,
            FileUploadStatus,
            path=f"/swarms/uploads/{self.upload_id}",
            options=request,
            validate=lambda received: verified(
                received,
                not (
                    received.upload_id != self.upload_id
                    or received.project_id != self._project_id
                    or received.manifest != self._manifest
                ),
                "Upload status changed identity or immutable manifest",
            ),
        )
        return result

    @artifact_alias
    def transfer(self, source: FileSource, *, transfer: TransferOptions) -> TransferReceipt:
        """Send bounded exact bytes using a scoped grant; never submit or automatically retry. Restartable flows use explicit prepare/status/finalize identities.

        See: SDK_0230_IMPLEMENTATION.md."""
        started = time.monotonic()
        status = self.status(request=ReadOptions(timeout_seconds=transfer.timeout_seconds))
        remaining = transfer.timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise RequestTimeoutError("Upload status read exhausted transfer deadline")
        return transfer_bytes(
            status,
            source,
            transfer.model_copy(update={"timeout_seconds": remaining}),
            self._transport.transfer_hosts,
        )

    @artifact_alias
    def finalize(self, *, mutation: MutationOptions | None = None) -> FileUploadReceipt:
        """Ask the backend to seal verified bytes; resolve a lost acknowledgement before repeating.

        See: SDK_0230_IMPLEMENTATION.md."""
        result = self._transport.request(
            operations.FINALIZE_FILE_UPLOAD,
            FileUploadReceipt,
            path=f"/swarms/uploads/{self.upload_id}/finalize",
            options=mutation,
            validate=lambda received: verified(
                received,
                not (
                    received.upload_id != self.upload_id
                    or received.file.project_id != self._project_id
                    or received.file.manifest != self._manifest
                ),
                "Finalized file identity or digest mismatch",
            ),
        )
        return result


class Uploads:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    @artifact_alias
    def prepare(
        self,
        project_id: str,
        *,
        manifest: FileManifest,
        mutation: MutationOptions | None = None,
    ) -> UploadSession:
        """Prepare a scoped expiring transfer session for an immutable manifest.

        See: SDK_0230_IMPLEMENTATION.md."""
        project_id = identifier(project_id)
        result = self._transport.request(
            operations.PREPARE_FILE_UPLOAD,
            FileUploadStatus,
            path=f"/swarms/projects/{project_id}/uploads",
            body=manifest,
            options=mutation,
            validate=lambda received: verified(
                received,
                not (received.project_id != project_id or received.manifest != manifest),
                "Upload preparation changed project or manifest",
            ),
        )
        return UploadSession(self._transport, result)

    @artifact_alias
    def get(self, upload_id: str, *, request: ReadOptions | None = None) -> UploadSession:
        """Read or reattach the exact authorized resource with fresh server state.

        See: SDK_0230_IMPLEMENTATION.md."""
        upload_id = identifier(upload_id)
        result = self._transport.request(
            operations.FILE_UPLOAD,
            FileUploadStatus,
            path=f"/swarms/uploads/{upload_id}",
            options=request,
            validate=lambda received: verified(
                received, received.upload_id == upload_id, "Upload reattachment identity mismatch"
            ),
        )
        return UploadSession(self._transport, result)

    @artifact_alias
    def resolve_creation(
        self, identity: MutationIdentity, *, request: ReadOptions | None = None
    ) -> CreationResolution[UploadSession]:
        return recover_creation(
            self._transport,
            identity,
            operations.PREPARE_FILE_UPLOAD,
            "upload",
            lambda resource_id: self.get(resource_id, request=request),
            request,
        )


class AsyncUploadSession:
    def __init__(self, transport: AsyncHttpTransport, snapshot: FileUploadStatus) -> None:
        self._transport = transport
        self.upload_id = snapshot.upload_id
        self._project_id = snapshot.project_id
        self._manifest = snapshot.manifest

    @artifact_alias
    async def status(self, *, request: ReadOptions | None = None) -> FileUploadStatus:
        """Read fresh state; this does not cancel, restart or finalize work.

        See: SDK_0230_IMPLEMENTATION.md."""
        result = await self._transport.request(
            operations.FILE_UPLOAD,
            FileUploadStatus,
            path=f"/swarms/uploads/{self.upload_id}",
            options=request,
            validate=lambda received: verified(
                received,
                not (
                    received.upload_id != self.upload_id
                    or received.project_id != self._project_id
                    or received.manifest != self._manifest
                ),
                "Upload status changed identity or immutable manifest",
            ),
        )
        return result

    @artifact_alias
    async def transfer(self, source: FileSource, *, transfer: TransferOptions) -> TransferReceipt:
        """Send bounded exact bytes using a scoped grant; never submit or automatically retry. Restartable flows use explicit prepare/status/finalize identities.

        See: SDK_0230_IMPLEMENTATION.md."""
        started = time.monotonic()
        status = await self.status(request=ReadOptions(timeout_seconds=transfer.timeout_seconds))
        remaining = transfer.timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise RequestTimeoutError("Upload status read exhausted transfer deadline")
        return await async_transfer(
            status,
            source,
            transfer.model_copy(update={"timeout_seconds": remaining}),
            self._transport.transfer_hosts,
        )

    @artifact_alias
    async def finalize(self, *, mutation: MutationOptions | None = None) -> FileUploadReceipt:
        """Ask the backend to seal verified bytes; resolve a lost acknowledgement before repeating.

        See: SDK_0230_IMPLEMENTATION.md."""
        result = await self._transport.request(
            operations.FINALIZE_FILE_UPLOAD,
            FileUploadReceipt,
            path=f"/swarms/uploads/{self.upload_id}/finalize",
            options=mutation,
            validate=lambda received: verified(
                received,
                not (
                    received.upload_id != self.upload_id
                    or received.file.project_id != self._project_id
                    or received.file.manifest != self._manifest
                ),
                "Finalized file identity or digest mismatch",
            ),
        )
        return result


class AsyncUploads:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    @artifact_alias
    async def prepare(
        self,
        project_id: str,
        *,
        manifest: FileManifest,
        mutation: MutationOptions | None = None,
    ) -> AsyncUploadSession:
        """Prepare a scoped expiring transfer session for an immutable manifest.

        See: SDK_0230_IMPLEMENTATION.md."""
        project_id = identifier(project_id)
        result = await self._transport.request(
            operations.PREPARE_FILE_UPLOAD,
            FileUploadStatus,
            path=f"/swarms/projects/{project_id}/uploads",
            body=manifest,
            options=mutation,
            validate=lambda received: verified(
                received,
                not (received.project_id != project_id or received.manifest != manifest),
                "Upload preparation changed project or manifest",
            ),
        )
        return AsyncUploadSession(self._transport, result)

    @artifact_alias
    async def get(
        self, upload_id: str, *, request: ReadOptions | None = None
    ) -> AsyncUploadSession:
        """Read or reattach the exact authorized resource with fresh server state.

        See: SDK_0230_IMPLEMENTATION.md."""
        upload_id = identifier(upload_id)
        result = await self._transport.request(
            operations.FILE_UPLOAD,
            FileUploadStatus,
            path=f"/swarms/uploads/{upload_id}",
            options=request,
            validate=lambda received: verified(
                received, received.upload_id == upload_id, "Upload reattachment identity mismatch"
            ),
        )
        return AsyncUploadSession(self._transport, result)

    @artifact_alias
    async def resolve_creation(
        self, identity: MutationIdentity, *, request: ReadOptions | None = None
    ) -> CreationResolution[AsyncUploadSession]:
        async def attach(resource_id: str) -> AsyncUploadSession:
            return await self.get(resource_id, request=request)

        return await recover_creation_async(
            self._transport, identity, operations.PREPARE_FILE_UPLOAD, "upload", attach, request
        )
