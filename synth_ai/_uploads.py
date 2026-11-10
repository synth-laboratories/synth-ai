"""Prepare, stream and recover uploads without retaining whole files in memory."""

from __future__ import annotations

from contextlib import aclosing
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from ._file_source import FileSource, open_source
from ._generated.models import File, Upload
from .errors import DataIntegrityError, SynthError, UploadTransferError, check_upload_format

if TYPE_CHECKING:
    from ._transport import Transport


class ArtifactUploads:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def upload_file(
        self, filename: str, *, media_type: str | None, idempotency_key: str | None
    ) -> File:
        path = Path(filename)
        formats = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".markdown": "text/markdown",
            ".json": "application/json",
        }
        media = check_upload_format(media_type or formats.get(path.suffix.lower()))
        key = idempotency_key or str(uuid4())
        async with open_source(path, self._transport.file_io) as source:
            upload = await self._transport.request(
                "createUpload",
                Upload,
                body={
                    "name": path.name,
                    "media_type": media,
                    "size_bytes": source.size,
                    "digest_sha256": source.digest_sha256,
                },
                idempotency_key=key,
            )
            return await self._finish(upload, source, key)

    async def resume(self, upload_id: str, filename: str, *, idempotency_key: str) -> File:
        upload = await self._transport.request("getUpload", Upload, path={"upload_id": upload_id})
        if upload.state == "committed":
            return upload.file
        async with open_source(Path(filename), self._transport.file_io) as source:
            if (
                source.size != upload.file.size_bytes
                or source.digest_sha256 != upload.file.digest_sha256
            ):
                raise DataIntegrityError("resume content differs from the retained upload")
            return await self._finish(upload, source, idempotency_key)

    async def _finish(self, upload: Upload, source: FileSource, key: str) -> File:
        try:
            if (
                upload.file.size_bytes != source.size
                or upload.file.digest_sha256 != source.digest_sha256
            ):
                raise DataIntegrityError("upload manifest differs from the prepared source")
            if upload.state == "committed":
                return upload.file
            if upload.put is None:
                raise UploadTransferError(upload.upload_id, key)
            async with aclosing(source.chunks(self._transport.file_io)) as chunks:
                await self._transport.put_upload(upload.put, chunks, idempotency_key=key)
            await self._transport.file_io.run(source.verify)
            finalized = await self._transport.request(
                "finalizeUpload", Upload, path={"upload_id": upload.upload_id}, idempotency_key=key
            )
            if finalized.state != "committed":
                raise UploadTransferError(upload.upload_id, key)
            if (
                finalized.file.size_bytes != source.size
                or finalized.file.digest_sha256 != source.digest_sha256
            ):
                raise DataIntegrityError("committed upload differs from the prepared source")
            return finalized.file
        except SynthError as cause:
            raise UploadTransferError(upload.upload_id, key) from cause
