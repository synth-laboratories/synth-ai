"""Bounded artifact HTTP transfers and atomic verified downloads."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from typing import TYPE_CHECKING
from urllib.parse import quote

import httpx

from ._generated.operations import OPERATIONS
from ._http_errors import raise_stream_error
from ._validation import require_non_empty, validate_idempotency_key, verify_sha256
from .errors import (
    DataIntegrityError,
    SynthError,
    TransportFailureError,
    local_problem,
    output_content_too_large,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Callable
    from pathlib import Path

    from ._file_source import FileIO
    from ._generated.models import UploadGrant
    from ._retry import RetryPolicy

_ALLOWED_UPLOAD_HEADERS = frozenset(
    {"content-type", "x-amz-content-sha256", "if-none-match", "content-length"}
)
_CONTROL_CHARS = ("\r", "\n", "\0")


def _output_route(run_id: str, output_id: str) -> str:
    validated_run = require_non_empty("run_id", run_id)
    validated_output = require_non_empty("output_id", output_id)
    return (
        OPERATIONS["getRunOutputContent"]["path"]
        .replace("{run_id}", quote(validated_run, safe=""))
        .replace("{output_id}", quote(validated_output, safe=""))
    )


def _validate_pin(revision_id: str, digest_sha256: str) -> None:
    require_non_empty("revision_id", revision_id)
    require_non_empty("digest_sha256", digest_sha256)


class ArtifactTransfer:
    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        timeout: float,
        retry_policy: RetryPolicy,
        headers: Callable[[], dict[str, str]],
        file_io: FileIO,
    ) -> None:
        self.client = client
        self.base_url = base_url
        self.timeout = timeout
        self.retry_policy = retry_policy
        self.headers = headers
        self.file_io = file_io

    async def put_upload(
        self, grant: UploadGrant, content: bytes | AsyncIterable[bytes], *, idempotency_key: str
    ) -> None:
        validate_idempotency_key(idempotency_key, max_bytes=255)
        target = httpx.URL(grant.url)
        if (
            target.username
            or target.password
            or target.fragment
            or not target.host
            or target.scheme not in {"https", "http"}
            or (target.scheme == "http" and target.host not in {"localhost", "127.0.0.1", "::1"})
        ):
            raise SynthError("Upload grant has an invalid URL")
        headers = grant.headers
        if any(
            name.lower() not in _ALLOWED_UPLOAD_HEADERS
            or any(c in name or c in value for c in _CONTROL_CHARS)
            for name, value in headers.items()
        ):
            raise SynthError("Upload grant has unsupported headers")
        # Construct the request directly: never merge API/client headers, cookies
        # or default authentication into a storage-origin request.
        request = httpx.Request("PUT", target, content=content, headers=headers)
        try:
            response = await self.client.send(request, auth=None, follow_redirects=False)
        except httpx.TransportError as error:
            raise TransportFailureError(
                local_problem("upload_transfer_failed", "uploadFile", unknown=True),
                idempotency_key=idempotency_key,
            ) from error
        if response.status_code not in {200, 201, 204, 412}:
            raise SynthError("Upload transfer refused: HTTP " + str(response.status_code))

    async def read_output(
        self,
        run_id: str,
        output_id: str,
        *,
        revision_id: str,
        digest_sha256: str,
        max_bytes: int,
    ) -> tuple[bytes, str]:
        """Read a pinned output revision into memory, refusing past max_bytes.

        Returns the bytes and the served Content-Type. Stops reading at the bound.
        """
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("invalid read size bound")
        _validate_pin(revision_id, digest_sha256)
        route = _output_route(run_id, output_id)
        try:
            async with asyncio.timeout(self.retry_policy.deadline_seconds):
                async with self.client.stream(
                    "GET",
                    self.base_url + route,
                    params={"revision_id": revision_id, "digest_sha256": digest_sha256},
                    headers=self.headers(),
                    follow_redirects=False,
                    timeout=self.timeout,
                ) as response:
                    if not response.is_success:
                        await raise_stream_error(response)
                    if response.headers.get("X-Synth-Output-Revision", revision_id) != revision_id:
                        raise DataIntegrityError(
                            "output revision does not match the pinned reference"
                        )
                    declared = response.headers.get("Content-Length", "")
                    if declared.isdigit() and int(declared) > max_bytes:
                        raise output_content_too_large(int(declared), max_bytes, lower_bound=False)
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        content += chunk
                        if len(content) > max_bytes:
                            raise output_content_too_large(
                                len(content), max_bytes, lower_bound=True
                            )
                    media_type = str(
                        response.headers.get("Content-Type", "application/octet-stream")
                    )
        except (httpx.TransportError, TimeoutError) as cause:
            raise TransportFailureError(
                local_problem("output_transfer_failed", "getRunOutputContent")
            ) from cause
        data = bytes(content)
        verify_sha256(data, digest_sha256, label="output")
        return data, media_type

    async def download_output(
        self,
        run_id: str,
        output_id: str,
        destination: Path,
        *,
        revision_id: str,
        digest_sha256: str,
        max_bytes: int,
        expected_bytes: int | None = None,
    ) -> None:
        if (
            type(max_bytes) is not int
            or max_bytes <= 0
            or (
                expected_bytes is not None
                and (type(expected_bytes) is not int or not 0 <= expected_bytes <= max_bytes)
            )
        ):
            raise ValueError("invalid download size bound")
        _validate_pin(revision_id, digest_sha256)
        route = _output_route(run_id, output_id)
        temporary: str | None = None
        try:
            async with asyncio.timeout(self.retry_policy.deadline_seconds):
                async with self.client.stream(
                    "GET",
                    self.base_url + route,
                    params={"revision_id": revision_id, "digest_sha256": digest_sha256},
                    headers=self.headers(),
                    follow_redirects=False,
                    timeout=self.timeout,
                ) as response:
                    if not response.is_success:
                        await raise_stream_error(response)
                    if response.headers.get("X-Synth-Output-Revision", revision_id) != revision_id:
                        raise DataIntegrityError(
                            "output revision does not match the pinned reference"
                        )
                    if (
                        response.headers.get("X-Synth-Content-SHA256", digest_sha256)
                        != digest_sha256
                    ):
                        raise DataIntegrityError(
                            "output digest header does not match the pinned reference"
                        )
                    digest, size = hashlib.sha256(), 0
                    with tempfile.NamedTemporaryFile(
                        dir=destination.parent, prefix="." + destination.name + ".", delete=False
                    ) as target:
                        temporary = target.name
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > max_bytes:
                                raise DataIntegrityError("output exceeds the download size bound")
                            digest.update(chunk)
                            await self.file_io.run(lambda chunk=chunk: target.write(chunk))

                        def flush() -> None:
                            target.flush()
                            os.fsync(target.fileno())

                        await self.file_io.run(flush)
                    if digest.hexdigest() != digest_sha256 or (
                        expected_bytes is not None and size != expected_bytes
                    ):
                        raise DataIntegrityError("download does not match the pinned output")
                    source_path = temporary
                    await self.file_io.run(lambda: os.replace(source_path, destination))
                    temporary = None
        except (httpx.TransportError, TimeoutError) as cause:
            raise TransportFailureError(
                local_problem("output_transfer_failed", "getRunOutputContent")
            ) from cause
        finally:
            if temporary is not None:
                source_path = temporary
                await self.file_io.run(lambda: os.unlink(source_path))
