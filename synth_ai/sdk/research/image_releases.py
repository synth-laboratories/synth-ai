"""Customer image-release APIs over the shared Research transport.

# See: testing/specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

import httpx

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.sdk.research.contracts.image_releases import (
    ActorRuntimeImageReleaseArchive,
    ActorRuntimeImageReleaseList,
    ImageRelease,
    ImageReleaseFinalize,
    ImageReleaseFinalizeRequest,
    ImageReleaseId,
    ImageReleaseUpload,
    ImageReleaseUploadRequest,
    RegistryActorRuntimeImageRegistration,
    RegistryActorRuntimeImageRegistrationRequest,
    RuntimeImageReleaseId,
    image_release_from_wire,
)
from synth_ai.sdk.research.operations import research_operation


def _request(
    operation_id: str,
    path: str,
    *,
    body: JsonObject | None = None,
) -> HttpRequest:
    return HttpRequest(research_operation(operation_id), path, body=body)


def _upload(
    value: object,
    *,
    request: ImageReleaseUploadRequest,
) -> ImageReleaseUpload:
    upload = ImageReleaseUpload.from_wire(cast(JsonValue, value))
    if upload.declaration != request.declaration:
        raise ValueError("image upload response changed its declaration")
    if upload.expires_in != request.expires_in:
        raise ValueError("image upload response changed its expiration")
    return upload


def _finalize(
    value: object,
    *,
    request: ImageReleaseFinalizeRequest,
) -> ImageReleaseFinalize:
    result = ImageReleaseFinalize.from_wire(cast(JsonValue, value))
    if result.release.declaration != request.declaration:
        raise ValueError("image finalize response changed its declaration")
    if result.upload_reconciliation.upload_id != request.upload_id:
        raise ValueError("image finalize response changed its upload identity")
    return result


def _archive(
    value: object,
    *,
    runtime_image_release_id: RuntimeImageReleaseId,
) -> ActorRuntimeImageReleaseArchive:
    result = ActorRuntimeImageReleaseArchive.from_wire(cast(JsonValue, value))
    if result.runtime_image_release.runtime_image_release_id != runtime_image_release_id:
        raise ValueError("image archive response changed its runtime identity")
    return result


def _retrieve(value: object, *, release_id: ImageReleaseId) -> ImageRelease:
    release = image_release_from_wire(cast(JsonValue, value))
    if release.release_id != release_id:
        raise ValueError("image retrieve response changed its release identity")
    return release


def _archive_path(
    archive_path: str | Path,
    *,
    request: ImageReleaseUploadRequest,
) -> Path:
    """Verify the exact bytes that the backend will materialize.

    The signed upload URL is intentionally the only direct-storage capability
    used here.  Authentication for the Research API is never forwarded to the
    object store.
    """
    path = Path(archive_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"image release archive is not a file: {path}")
    if path.stat().st_size != request.declaration.archive_size_bytes:
        raise ValueError("image release archive size does not match its declaration")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != request.declaration.archive_sha256:
        raise ValueError("image release archive digest does not match its declaration")
    return path


def _upload_timeout_seconds(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("upload_timeout_seconds must be a number")
    timeout_seconds = float(value)
    if not 1.0 <= timeout_seconds <= 7200.0:
        raise ValueError("upload_timeout_seconds must be between 1 and 7200")
    return timeout_seconds


def _verify_finalized_upload(
    finalized: ImageReleaseFinalize,
    *,
    upload: ImageReleaseUpload,
) -> ImageReleaseFinalize:
    if finalized.release.release_id != upload.release_id:
        raise ValueError("image finalize response changed its release identity")
    if finalized.upload_reconciliation.upload_id != upload.upload_id:
        raise ValueError("image finalize response changed its upload identity")
    return finalized


class ImageReleasesAPI:
    """Immutable uploads plus executable actor-image materializations."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def create_upload(
        self,
        request: ImageReleaseUploadRequest,
    ) -> ImageReleaseUpload:
        if not isinstance(request, ImageReleaseUploadRequest):
            raise ValueError("request must be ImageReleaseUploadRequest")
        value = self._transport.execute(
            _request(
                "create_image_release_upload",
                "/smr/v1/image-releases/upload-url",
                body=request.to_wire(),
            )
        )
        return _upload(value, request=request)

    def finalize(
        self,
        request: ImageReleaseFinalizeRequest,
    ) -> ImageReleaseFinalize:
        if not isinstance(request, ImageReleaseFinalizeRequest):
            raise ValueError("request must be ImageReleaseFinalizeRequest")
        value = self._transport.execute(
            _request(
                "finalize_image_release",
                "/smr/v1/image-releases/finalize",
                body=request.to_wire(),
            )
        )
        return _finalize(value, request=request)

    def register_registry_image(
        self,
        request: RegistryActorRuntimeImageRegistrationRequest,
    ) -> RegistryActorRuntimeImageRegistration:
        """Register an org-scoped digest already published to the Synth registry."""

        if not isinstance(request, RegistryActorRuntimeImageRegistrationRequest):
            raise ValueError("request must be RegistryActorRuntimeImageRegistrationRequest")
        value = self._transport.execute(
            _request(
                "register_customer_actor_registry_image",
                "/smr/v1/image-releases/register-registry",
                body=request.to_wire(),
            )
        )
        result = RegistryActorRuntimeImageRegistration.from_wire(cast(JsonValue, value))
        if (
            result.runtime_image_release.resolved_digest
            != request.declaration.image_manifest_digest
        ):
            raise ValueError("registry image response changed its manifest digest")
        return result

    def upload_archive(
        self,
        archive_path: str | Path,
        request: ImageReleaseUploadRequest,
        *,
        upload_timeout_seconds: float = 1800.0,
    ) -> ImageReleaseFinalize:
        """Upload and materialize an exact declared OCI archive.

        The local archive is validated before requesting a presigned URL.  A
        non-2xx object-store response is terminal: finalize is never attempted
        against bytes the SDK did not observe as successfully uploaded.
        """
        if not isinstance(request, ImageReleaseUploadRequest):
            raise ValueError("request must be ImageReleaseUploadRequest")
        path = _archive_path(archive_path, request=request)
        timeout_seconds = _upload_timeout_seconds(upload_timeout_seconds)
        upload = self.create_upload(request)
        if upload.upload_required:
            with (
                httpx.Client(timeout=timeout_seconds, follow_redirects=False) as client,
                path.open("rb") as archive,
            ):
                response = client.put(
                    upload.upload_url,
                    content=archive,
                    headers={"Content-Length": str(request.declaration.archive_size_bytes)},
                )
            if not response.is_success:
                raise RuntimeError(
                    f"image release archive upload failed with HTTP {response.status_code}"
                )
        return _verify_finalized_upload(
            self.finalize(
                ImageReleaseFinalizeRequest(
                    upload_id=upload.upload_id,
                    declaration=request.declaration,
                )
            ),
            upload=upload,
        )

    def list(self) -> ActorRuntimeImageReleaseList:
        value = self._transport.execute(
            _request(
                "list_customer_actor_images",
                "/smr/v1/image-releases",
            )
        )
        return ActorRuntimeImageReleaseList.from_wire(value)

    def archive(
        self,
        runtime_image_release_id: RuntimeImageReleaseId,
    ) -> ActorRuntimeImageReleaseArchive:
        runtime_image_release_id = RuntimeImageReleaseId(runtime_image_release_id)
        value = self._transport.execute(
            _request(
                "archive_customer_actor_image",
                f"/smr/v1/image-releases/{runtime_image_release_id}/archive",
            )
        )
        return _archive(value, runtime_image_release_id=runtime_image_release_id)

    def retrieve(self, release_id: ImageReleaseId) -> ImageRelease:
        release_id = ImageReleaseId(release_id)
        value = self._transport.execute(
            _request(
                "retrieve_image_release",
                f"/smr/v1/image-releases/{release_id}",
            )
        )
        return _retrieve(value, release_id=release_id)

    def get(self, release_id: ImageReleaseId) -> ImageRelease:
        """Retrieve one immutable image-release receipt.

        ``retrieve`` remains available for callers on the earlier core-client
        surface; ``get`` matches the lifecycle verb used by the session API.
        """
        return self.retrieve(release_id)


class AsyncImageReleasesAPI:
    """Native-async peer of :class:`ImageReleasesAPI`."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def create_upload(
        self,
        request: ImageReleaseUploadRequest,
    ) -> ImageReleaseUpload:
        if not isinstance(request, ImageReleaseUploadRequest):
            raise ValueError("request must be ImageReleaseUploadRequest")
        value = await self._transport.execute(
            _request(
                "create_image_release_upload",
                "/smr/v1/image-releases/upload-url",
                body=request.to_wire(),
            )
        )
        return _upload(value, request=request)

    async def finalize(
        self,
        request: ImageReleaseFinalizeRequest,
    ) -> ImageReleaseFinalize:
        if not isinstance(request, ImageReleaseFinalizeRequest):
            raise ValueError("request must be ImageReleaseFinalizeRequest")
        value = await self._transport.execute(
            _request(
                "finalize_image_release",
                "/smr/v1/image-releases/finalize",
                body=request.to_wire(),
            )
        )
        return _finalize(value, request=request)

    async def register_registry_image(
        self,
        request: RegistryActorRuntimeImageRegistrationRequest,
    ) -> RegistryActorRuntimeImageRegistration:
        """Register an org-scoped digest already published to the Synth registry."""

        if not isinstance(request, RegistryActorRuntimeImageRegistrationRequest):
            raise ValueError("request must be RegistryActorRuntimeImageRegistrationRequest")
        value = await self._transport.execute(
            _request(
                "register_customer_actor_registry_image",
                "/smr/v1/image-releases/register-registry",
                body=request.to_wire(),
            )
        )
        result = RegistryActorRuntimeImageRegistration.from_wire(cast(JsonValue, value))
        if (
            result.runtime_image_release.resolved_digest
            != request.declaration.image_manifest_digest
        ):
            raise ValueError("registry image response changed its manifest digest")
        return result

    async def upload_archive(
        self,
        archive_path: str | Path,
        request: ImageReleaseUploadRequest,
        *,
        upload_timeout_seconds: float = 1800.0,
    ) -> ImageReleaseFinalize:
        """Native-async upload and materialization of an exact OCI archive."""
        if not isinstance(request, ImageReleaseUploadRequest):
            raise ValueError("request must be ImageReleaseUploadRequest")
        path = _archive_path(archive_path, request=request)
        timeout_seconds = _upload_timeout_seconds(upload_timeout_seconds)
        upload = await self.create_upload(request)
        if upload.upload_required:
            async with httpx.AsyncClient(
                timeout=timeout_seconds,
                follow_redirects=False,
            ) as client:
                with path.open("rb") as archive:
                    response = await client.put(
                        upload.upload_url,
                        content=archive,
                        headers={"Content-Length": str(request.declaration.archive_size_bytes)},
                    )
            if not response.is_success:
                raise RuntimeError(
                    f"image release archive upload failed with HTTP {response.status_code}"
                )
        return _verify_finalized_upload(
            await self.finalize(
                ImageReleaseFinalizeRequest(
                    upload_id=upload.upload_id,
                    declaration=request.declaration,
                )
            ),
            upload=upload,
        )

    async def list(self) -> ActorRuntimeImageReleaseList:
        value = await self._transport.execute(
            _request(
                "list_customer_actor_images",
                "/smr/v1/image-releases",
            )
        )
        return ActorRuntimeImageReleaseList.from_wire(value)

    async def archive(
        self,
        runtime_image_release_id: RuntimeImageReleaseId,
    ) -> ActorRuntimeImageReleaseArchive:
        runtime_image_release_id = RuntimeImageReleaseId(runtime_image_release_id)
        value = await self._transport.execute(
            _request(
                "archive_customer_actor_image",
                f"/smr/v1/image-releases/{runtime_image_release_id}/archive",
            )
        )
        return _archive(value, runtime_image_release_id=runtime_image_release_id)

    async def retrieve(self, release_id: ImageReleaseId) -> ImageRelease:
        release_id = ImageReleaseId(release_id)
        value = await self._transport.execute(
            _request(
                "retrieve_image_release",
                f"/smr/v1/image-releases/{release_id}",
            )
        )
        return _retrieve(value, release_id=release_id)

    async def get(self, release_id: ImageReleaseId) -> ImageRelease:
        """Native-async peer of :meth:`ImageReleasesAPI.get`."""
        return await self.retrieve(release_id)


__all__ = ["AsyncImageReleasesAPI", "ImageReleasesAPI"]
