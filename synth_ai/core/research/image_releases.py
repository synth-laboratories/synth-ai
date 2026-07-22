"""Customer image-release APIs over the shared Research transport.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from typing import cast

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.research.contracts.image_releases import (
    ActorRuntimeImageReleaseArchive,
    ActorRuntimeImageReleaseList,
    ImageRelease,
    ImageReleaseFinalize,
    ImageReleaseFinalizeRequest,
    ImageReleaseId,
    ImageReleaseUpload,
    ImageReleaseUploadRequest,
    RuntimeImageReleaseId,
    image_release_from_wire,
)
from synth_ai.core.research.operations import research_operation


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
    if result.staging_cleanup.upload_id != request.upload_id:
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
        release_id = RuntimeImageReleaseId(runtime_image_release_id)
        value = self._transport.execute(
            _request(
                "archive_customer_actor_image",
                f"/smr/v1/image-releases/{release_id}/archive",
            )
        )
        return _archive(value, runtime_image_release_id=release_id)

    def retrieve(self, release_id: ImageReleaseId) -> ImageRelease:
        normalized_id = ImageReleaseId(release_id)
        value = self._transport.execute(
            _request(
                "retrieve_image_release",
                f"/smr/v1/image-releases/{normalized_id}",
            )
        )
        return _retrieve(value, release_id=normalized_id)


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
        release_id = RuntimeImageReleaseId(runtime_image_release_id)
        value = await self._transport.execute(
            _request(
                "archive_customer_actor_image",
                f"/smr/v1/image-releases/{release_id}/archive",
            )
        )
        return _archive(value, runtime_image_release_id=release_id)

    async def retrieve(self, release_id: ImageReleaseId) -> ImageRelease:
        normalized_id = ImageReleaseId(release_id)
        value = await self._transport.execute(
            _request(
                "retrieve_image_release",
                f"/smr/v1/image-releases/{normalized_id}",
            )
        )
        return _retrieve(value, release_id=normalized_id)


__all__ = ["AsyncImageReleasesAPI", "ImageReleasesAPI"]
