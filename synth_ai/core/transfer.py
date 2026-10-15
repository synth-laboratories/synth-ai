"""Credential-isolated bounded signed uploads. See SDK index_contract.md."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

import httpx

from synth_ai.core.local_files import open_regular_file
from synth_ai.errors import ContractError, OutcomeUncertainError, RequestTimeoutError
from synth_ai.options import TransferOptions
from synth_ai.types.uploads import TransferReceipt, validate_upload_headers


class TransferManifest(Protocol):
    @property
    def byte_size(self) -> int: ...
    @property
    def digest_sha256(self) -> str: ...
    @property
    def media_type(self) -> str: ...


class TransferSource(Protocol):
    @property
    def path(self) -> Path: ...
    @property
    def manifest(self) -> TransferManifest: ...


class TransferStatus(Protocol):
    @property
    def upload_id(self) -> str: ...
    @property
    def state(self) -> str: ...
    @property
    def manifest(self) -> TransferManifest: ...
    @property
    def transfer_url(self) -> str | None: ...
    @property
    def expires_at(self) -> datetime | None: ...
    @property
    def transfer_headers(self) -> dict[str, str]: ...


def validate_transfer(
    status: TransferStatus, source: TransferSource, options: TransferOptions, hosts: tuple[str, ...]
) -> str:
    if status.state != "prepared" or status.transfer_url is None or status.expires_at is None:
        raise ValueError("upload is not prepared for transfer")
    if status.expires_at <= datetime.now(UTC):
        raise ValueError("upload grant expired")
    if status.manifest != source.manifest or source.manifest.byte_size > options.maximum_bytes:
        raise ValueError("source manifest or selected bound does not match upload")
    validate_upload_headers(
        status.transfer_headers,
        size=source.manifest.byte_size,
        digest=source.manifest.digest_sha256,
        media_type=source.manifest.media_type,
    )
    url = urlsplit(status.transfer_url)
    if (
        not (
            url.scheme == "https"
            or (url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"})
        )
        or not url.hostname
        or url.hostname.lower() not in hosts
        or url.username
        or url.password
        or url.fragment
    ):
        raise ValueError("upload destination is outside the configured HTTPS host allowlist")
    if not source.path.is_file() or source.path.is_symlink():
        raise ValueError("source is no longer a regular file")
    return status.transfer_url


def chunks(source: TransferSource, options: TransferOptions, deadline: float) -> Iterator[bytes]:
    """Check bytes actually sent, rather than trusting an earlier file hash."""
    digest = sha256()
    size = 0
    with open_regular_file(source.path) as stream:
        for _ in range(options.maximum_bytes // options.chunk_bytes + 2):
            if time.monotonic() >= deadline:
                raise RequestTimeoutError("Upload deadline expired")
            chunk = stream.read(options.chunk_bytes)
            if not chunk:
                break
            size += len(chunk)
            if size > min(options.maximum_bytes, source.manifest.byte_size):
                raise ContractError("Upload source grew beyond its declared manifest")
            digest.update(chunk)
            yield chunk
        else:
            raise ContractError("Upload exceeded bounded read count")
    if size != source.manifest.byte_size or digest.hexdigest() != source.manifest.digest_sha256:
        raise ContractError("Upload source changed after selection")


async def async_chunks(
    source: TransferSource, options: TransferOptions, deadline: float
) -> AsyncIterator[bytes]:
    # Local file reads are bounded per chunk; HTTP upload remains native async.
    for chunk in chunks(source, options, deadline):
        yield chunk


def transfer(
    status: TransferStatus, source: TransferSource, options: TransferOptions, hosts: tuple[str, ...]
) -> TransferReceipt:
    # Use the same cancellable network owner as synchronous JSON/output calls.
    from synth_ai.core.http.blocking import BlockingLoop

    loop = BlockingLoop()
    try:
        return loop.call(async_transfer(status, source, options, hosts), options.timeout_seconds)
    except TimeoutError as cause:
        raise OutcomeUncertainError(
            "Upload transfer is uncertain; read upload status before repeating"
        ) from cause
    finally:
        loop.close()


async def async_transfer(
    status: TransferStatus, source: TransferSource, options: TransferOptions, hosts: tuple[str, ...]
) -> TransferReceipt:
    url = validate_transfer(status, source, options, hosts)
    deadline = time.monotonic() + options.timeout_seconds
    try:
        async with (
            asyncio.timeout(options.timeout_seconds),
            httpx.AsyncClient(trust_env=False) as client,
        ):
            request = client.build_request(
                "PUT",
                url,
                content=async_chunks(source, options, deadline),
                headers={
                    "Content-Type": source.manifest.media_type,
                    "Content-Length": str(source.manifest.byte_size),
                    **status.transfer_headers,
                },
                timeout=options.timeout_seconds,
            )
            response = await client.send(request, auth=None, follow_redirects=False, stream=True)
            try:
                if response.status_code not in {200, 204}:
                    raise ContractError(
                        "Upload destination refused transfer; no redirect is followed"
                    )
                return TransferReceipt(
                    upload_id=status.upload_id,
                    byte_size=source.manifest.byte_size,
                    digest_sha256=source.manifest.digest_sha256,
                    etag=response.headers.get("ETag"),
                )
            finally:
                await response.aclose()
    except (httpx.TransportError, ContractError, RequestTimeoutError, TimeoutError) as cause:
        raise OutcomeUncertainError(
            "Upload transfer is uncertain; read upload status before repeating"
        ) from cause
