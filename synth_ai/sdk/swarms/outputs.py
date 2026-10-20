"""Run-scoped output access. See SDK_0230_IMPLEMENTATION.md.

No signed download URLs or server filenames become local paths. Downloads are
published without overwriting only after full identity and digest verification.
"""

from __future__ import annotations

from .deprecated import artifact_alias

import os
import tempfile
from pathlib import Path

from synth_ai.contracts import swarms as operations
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.parameters import identifier, page_parameters, verified
from synth_ai.errors import OutputIntegrityError
from synth_ai.options import ReadOptions, TransferOptions
from synth_ai.types.base import Page, PageCursor

from .output_stream import AsyncOutputStream, OutputStream
from .types.outputs import DownloadReceipt, Output, TextContent


def text_transfer(
    maximum_bytes: int, request: ReadOptions | None, default_timeout: float
) -> TransferOptions:
    if type(maximum_bytes) is not int or not 1 <= maximum_bytes <= 65536:
        raise ValueError("text byte ceiling must be between 1 and 65536")
    return TransferOptions(
        timeout_seconds=(
            request.timeout_seconds
            if request and request.timeout_seconds is not None
            else default_timeout
        ),
        maximum_bytes=maximum_bytes,
        chunk_bytes=16384,
        retry_count_max=0,
    )


def decode_text(content: bytes, media_type: str) -> str:
    media_type = media_type.lower()
    if not (
        media_type.startswith("text/") or media_type in {"application/json", "application/xml"}
    ):
        raise OutputIntegrityError("Output is not a supported UTF-8 text media type")
    try:
        return content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as cause:
        raise OutputIntegrityError("Text output is not valid UTF-8") from cause


def temporary_destination(destination: Path) -> tuple[int, Path]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Download destination already exists")
    descriptor, name = tempfile.mkstemp(prefix=".synth-download-", dir=destination.parent)
    return descriptor, Path(name)


class Outputs:
    def __init__(self, transport: HttpTransport, run_id: str) -> None:
        self._transport = transport
        self.run_id = identifier(run_id)

    @artifact_alias
    def list(
        self, *, limit: int, cursor: PageCursor | None = None, request: ReadOptions | None = None
    ) -> Page[Output]:
        """Read one bounded page of current authorized descriptors."""
        return self._transport.request(
            operations.OUTPUTS,
            Page[Output],
            path=f"/swarms/runs/{self.run_id}/outputs",
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda result: verified(
                result,
                len(result.items) <= limit
                and all(item.run_id == self.run_id for item in result.items),
                "Output page identity or bound mismatch",
            ),
        )

    @artifact_alias
    def get(self, output_id: str, *, request: ReadOptions | None = None) -> Output:
        """Read metadata; a descriptor does not grant future content access."""
        output_id = identifier(output_id)
        return self._transport.request(
            operations.OUTPUT,
            Output,
            path=f"/swarms/runs/{self.run_id}/outputs/{output_id}",
            options=request,
            validate=lambda result: verified(
                result,
                result.run_id == self.run_id and result.output_id == output_id,
                "Output identity mismatch",
            ),
        )

    @artifact_alias
    def stream(self, output_id: str, *, transfer: TransferOptions) -> OutputStream:
        """Construct a lazy single-pass stream; chunks are provisional until EOF."""
        return OutputStream(self._transport, self.run_id, identifier(output_id), transfer)

    @artifact_alias
    def read_text(
        self, output_id: str, *, maximum_bytes: int, request: ReadOptions | None = None
    ) -> TextContent:
        """Read a complete verified UTF-8 output; never return an unverified prefix."""
        options = text_transfer(maximum_bytes, request, self._transport.request_timeout_seconds)
        with self.stream(output_id, transfer=options) as stream:
            content = b"".join(stream)
            return TextContent(
                metadata=stream.descriptor.metadata,
                reference=stream.reference,
                text=decode_text(content, stream.reference.media_type),
                transfer_timing=stream.timing,
            )

    @artifact_alias
    def download(
        self, output_id: str, destination: Path, *, transfer: TransferOptions
    ) -> DownloadReceipt:
        """Atomically publish verified bytes without replacing an existing destination."""
        output_id = identifier(output_id)
        descriptor, temporary = temporary_destination(destination)
        try:
            with (
                os.fdopen(descriptor, "wb") as local,
                self.stream(output_id, transfer=transfer) as stream,
            ):
                for chunk in stream:
                    local.write(chunk)
                local.flush()
                os.fsync(local.fileno())
                receipt = DownloadReceipt(
                    reference=stream.reference,
                    destination=destination,
                    byte_size=stream.bytes_read,
                    digest_sha256=stream.reference.digest_sha256,
                    transfer_timing=stream.timing,
                )
            os.link(temporary, destination)
            return receipt
        finally:
            temporary.unlink(missing_ok=True)


class AsyncOutputs:
    def __init__(self, transport: AsyncHttpTransport, run_id: str) -> None:
        self._transport = transport
        self.run_id = identifier(run_id)

    @artifact_alias
    async def list(
        self, *, limit: int, cursor: PageCursor | None = None, request: ReadOptions | None = None
    ) -> Page[Output]:
        """Read one bounded page of current authorized descriptors."""
        return await self._transport.request(
            operations.OUTPUTS,
            Page[Output],
            path=f"/swarms/runs/{self.run_id}/outputs",
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda result: verified(
                result,
                len(result.items) <= limit
                and all(item.run_id == self.run_id for item in result.items),
                "Output page identity or bound mismatch",
            ),
        )

    @artifact_alias
    async def get(self, output_id: str, *, request: ReadOptions | None = None) -> Output:
        """Read metadata; a descriptor does not grant future content access."""
        output_id = identifier(output_id)
        return await self._transport.request(
            operations.OUTPUT,
            Output,
            path=f"/swarms/runs/{self.run_id}/outputs/{output_id}",
            options=request,
            validate=lambda result: verified(
                result,
                result.run_id == self.run_id and result.output_id == output_id,
                "Output identity mismatch",
            ),
        )

    @artifact_alias
    def stream(self, output_id: str, *, transfer: TransferOptions) -> AsyncOutputStream:
        """Construct a lazy single-pass stream; chunks are provisional until EOF."""
        return AsyncOutputStream(self._transport, self.run_id, identifier(output_id), transfer)

    @artifact_alias
    async def read_text(
        self, output_id: str, *, maximum_bytes: int, request: ReadOptions | None = None
    ) -> TextContent:
        """Read a complete verified UTF-8 output; never return an unverified prefix."""
        options = text_transfer(maximum_bytes, request, self._transport.request_timeout_seconds)
        async with self.stream(output_id, transfer=options) as stream:
            content = b"".join([chunk async for chunk in stream])
            return TextContent(
                metadata=stream.descriptor.metadata,
                reference=stream.reference,
                text=decode_text(content, stream.reference.media_type),
                transfer_timing=stream.timing,
            )

    @artifact_alias
    async def download(
        self, output_id: str, destination: Path, *, transfer: TransferOptions
    ) -> DownloadReceipt:
        """Atomically publish verified bytes without replacing an existing destination."""
        output_id = identifier(output_id)
        descriptor, temporary = temporary_destination(destination)
        try:
            with os.fdopen(descriptor, "wb") as local:
                async with self.stream(output_id, transfer=transfer) as stream:
                    async for chunk in stream:
                        local.write(chunk)
                local.flush()
                os.fsync(local.fileno())
                receipt = DownloadReceipt(
                    reference=stream.reference,
                    destination=destination,
                    byte_size=stream.bytes_read,
                    digest_sha256=stream.reference.digest_sha256,
                    transfer_timing=stream.timing,
                )
            os.link(temporary, destination)
            return receipt
        finally:
            temporary.unlink(missing_ok=True)
