"""Preparation/transfer/finalization are distinct owner operations. See SDK index_contract.md."""

from __future__ import annotations

import time

from synth_ai.contracts import operations
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

from .types.uploads import (
    ContributionUploadManifest,
    TransferReceipt,
    UploadReceipt,
    UploadSource,
    UploadStatus,
)


class UploadSession:
    def __init__(self, transport: HttpTransport, snapshot: UploadStatus) -> None:
        self._transport = transport
        self.upload_id = snapshot.upload_id
        self._contribution_id = snapshot.contribution_id
        self._manifest = snapshot.manifest

    def status(self, *, request: ReadOptions | None = None) -> UploadStatus:
        """Read fresh state; this does not cancel, restart or finalize work.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = self._transport.request(
            operations.GET_UPLOAD,
            UploadStatus,
            path=f"/index/uploads/{self.upload_id}",
            options=request,
            validate=lambda received: verified(
                received,
                not (
                    received.upload_id != self.upload_id
                    or received.contribution_id != self._contribution_id
                    or received.manifest != self._manifest
                ),
                "Upload status changed identity or immutable manifest",
            ),
        )
        return result

    def transfer(self, source: UploadSource, *, transfer: TransferOptions) -> TransferReceipt:
        """Send bounded exact bytes using a scoped grant; never submit or automatically retry. Restartable flows use explicit prepare/status/finalize identities.

        See: backend/notes/specifications/sdk/index_contract.md."""
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

    def finalize(self, *, mutation: MutationOptions | None = None) -> UploadReceipt:
        """Ask the backend to seal verified bytes; resolve a lost acknowledgement before repeating.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = self._transport.request(
            operations.FINALIZE_UPLOAD,
            UploadReceipt,
            path=f"/index/uploads/{self.upload_id}/finalize",
            options=mutation,
            validate=lambda received: verified(
                received,
                not (
                    received.upload_id != self.upload_id
                    or received.reference.contribution_id != self._contribution_id
                    or received.reference.digest_sha256 != self._manifest.digest_sha256
                ),
                "Finalized package identity or digest mismatch",
            ),
        )
        return result


class Uploads:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def prepare(
        self,
        contribution_id: str,
        *,
        manifest: ContributionUploadManifest,
        mutation: MutationOptions | None = None,
    ) -> UploadSession:
        """Prepare a scoped expiring transfer session for an immutable manifest.

        See: backend/notes/specifications/sdk/index_contract.md."""
        contribution_id = identifier(contribution_id)
        result = self._transport.request(
            operations.PREPARE_UPLOAD,
            UploadStatus,
            path=f"/index/contributions/{contribution_id}/uploads",
            body=manifest,
            options=mutation,
            validate=lambda received: verified(
                received,
                not (received.contribution_id != contribution_id or received.manifest != manifest),
                "Upload preparation changed contribution or manifest",
            ),
        )
        return UploadSession(self._transport, result)

    def get(self, upload_id: str, *, request: ReadOptions | None = None) -> UploadSession:
        """Read or reattach the exact authorized resource with fresh server state.

        See: backend/notes/specifications/sdk/index_contract.md."""
        upload_id = identifier(upload_id)
        result = self._transport.request(
            operations.GET_UPLOAD,
            UploadStatus,
            path=f"/index/uploads/{upload_id}",
            options=request,
            validate=lambda received: verified(
                received, received.upload_id == upload_id, "Upload reattachment identity mismatch"
            ),
        )
        return UploadSession(self._transport, result)

    def resolve_creation(
        self, identity: MutationIdentity, *, request: ReadOptions | None = None
    ) -> CreationResolution[UploadSession]:
        return recover_creation(
            self._transport,
            identity,
            operations.PREPARE_UPLOAD,
            "upload",
            lambda resource_id: self.get(resource_id, request=request),
            request,
        )


class AsyncUploadSession:
    def __init__(self, transport: AsyncHttpTransport, snapshot: UploadStatus) -> None:
        self._transport = transport
        self.upload_id = snapshot.upload_id
        self._contribution_id = snapshot.contribution_id
        self._manifest = snapshot.manifest

    async def status(self, *, request: ReadOptions | None = None) -> UploadStatus:
        """Read fresh state; this does not cancel, restart or finalize work.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = await self._transport.request(
            operations.GET_UPLOAD,
            UploadStatus,
            path=f"/index/uploads/{self.upload_id}",
            options=request,
            validate=lambda received: verified(
                received,
                not (
                    received.upload_id != self.upload_id
                    or received.contribution_id != self._contribution_id
                    or received.manifest != self._manifest
                ),
                "Upload status changed identity or immutable manifest",
            ),
        )
        return result

    async def transfer(self, source: UploadSource, *, transfer: TransferOptions) -> TransferReceipt:
        """Send bounded exact bytes using a scoped grant; never submit or automatically retry. Restartable flows use explicit prepare/status/finalize identities.

        See: backend/notes/specifications/sdk/index_contract.md."""
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

    async def finalize(self, *, mutation: MutationOptions | None = None) -> UploadReceipt:
        """Ask the backend to seal verified bytes; resolve a lost acknowledgement before repeating.

        See: backend/notes/specifications/sdk/index_contract.md."""
        result = await self._transport.request(
            operations.FINALIZE_UPLOAD,
            UploadReceipt,
            path=f"/index/uploads/{self.upload_id}/finalize",
            options=mutation,
            validate=lambda received: verified(
                received,
                not (
                    received.upload_id != self.upload_id
                    or received.reference.contribution_id != self._contribution_id
                    or received.reference.digest_sha256 != self._manifest.digest_sha256
                ),
                "Finalized package identity or digest mismatch",
            ),
        )
        return result


class AsyncUploads:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def prepare(
        self,
        contribution_id: str,
        *,
        manifest: ContributionUploadManifest,
        mutation: MutationOptions | None = None,
    ) -> AsyncUploadSession:
        """Prepare a scoped expiring transfer session for an immutable manifest.

        See: backend/notes/specifications/sdk/index_contract.md."""
        contribution_id = identifier(contribution_id)
        result = await self._transport.request(
            operations.PREPARE_UPLOAD,
            UploadStatus,
            path=f"/index/contributions/{contribution_id}/uploads",
            body=manifest,
            options=mutation,
            validate=lambda received: verified(
                received,
                not (received.contribution_id != contribution_id or received.manifest != manifest),
                "Upload preparation changed contribution or manifest",
            ),
        )
        return AsyncUploadSession(self._transport, result)

    async def get(
        self, upload_id: str, *, request: ReadOptions | None = None
    ) -> AsyncUploadSession:
        """Read or reattach the exact authorized resource with fresh server state.

        See: backend/notes/specifications/sdk/index_contract.md."""
        upload_id = identifier(upload_id)
        result = await self._transport.request(
            operations.GET_UPLOAD,
            UploadStatus,
            path=f"/index/uploads/{upload_id}",
            options=request,
            validate=lambda received: verified(
                received, received.upload_id == upload_id, "Upload reattachment identity mismatch"
            ),
        )
        return AsyncUploadSession(self._transport, result)

    async def resolve_creation(
        self, identity: MutationIdentity, *, request: ReadOptions | None = None
    ) -> CreationResolution[AsyncUploadSession]:
        async def attach(resource_id: str) -> AsyncUploadSession:
            return await self.get(resource_id, request=request)

        return await recover_creation_async(
            self._transport, identity, operations.PREPARE_UPLOAD, "upload", attach, request
        )
