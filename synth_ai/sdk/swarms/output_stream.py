"""Bounded verified output streams. See SDK_0230_IMPLEMENTATION.md.

Chunks are provisional until EOF. Only verified complete bytes may be published
as a downloaded file. Closing observation never changes a remote run.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Generator, Iterator
from contextlib import contextmanager
from hashlib import sha256
from types import TracebackType

import httpx

from synth_ai.contracts import swarms as operations
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.blocking import CLEANUP_SECONDS, CancellationScope
from synth_ai.core.http.cleanup import handle_cleanup_failure
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.http.lifetime import await_cleanup
from synth_ai.core.http.response import decode_response
from synth_ai.errors import (
    NotFoundError,
    OutputIntegrityError,
    PermissionDeniedError,
    RequestTimeoutError,
    ResultNotReadyError,
    TransportError,
)
from synth_ai.options import ReadOptions, TransferOptions
from synth_ai.types.base import ClientTiming, ResponseModel

from .types.outputs import Output, OutputRef, TransferTiming


def available(output: Output, run_id: str, output_id: str, maximum_bytes: int) -> OutputRef:
    if output.run_id != run_id or output.output_id != output_id:
        raise OutputIntegrityError("Output descriptor identity mismatch")
    if output.state == "pending":
        raise ResultNotReadyError("Output is not available yet")
    if output.state == "withheld":
        raise PermissionDeniedError("Output delivery is not authorized")
    if output.state in {"expired", "deleted"}:
        raise NotFoundError("Output is no longer available")
    if output.reference is None:
        raise OutputIntegrityError("Available output lacks an immutable reference")
    if output.reference.byte_size > maximum_bytes:
        raise OutputIntegrityError("Output exceeds the selected byte ceiling")
    return output.reference


def validate_headers(response: httpx.Response, reference: OutputRef) -> None:
    expected = {
        "X-Synth-Contract": "swarms.v1",
        "X-Synth-Run-Id": reference.run_id,
        "X-Synth-Output-Id": reference.output_id,
        "X-Synth-Output-Revision": reference.revision_id,
        "X-Synth-Content-SHA256": reference.digest_sha256,
        "ETag": '"' + reference.digest_sha256 + '"',
    }
    if any(response.headers.get(key) != value for key, value in expected.items()):
        raise OutputIntegrityError("Output response identity mismatch")
    media_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if media_type != reference.media_type.lower():
        raise OutputIntegrityError("Output media type mismatch")
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise OutputIntegrityError("Compressed output is outside the byte contract")
    content_type = response.headers.get("Content-Type", "")
    for parameter in content_type.split(";")[1:]:
        key, separator, value = parameter.strip().partition("=")
        if (
            separator
            and key.lower() == "charset"
            and value.strip('"').lower() not in {"utf-8", "utf8"}
        ):
            raise OutputIntegrityError("Output character encoding is not UTF-8")
    length = response.headers.get("Content-Length")
    if length is not None and (
        len(length) > 20
        or not length.isascii()
        or not length.isdigit()
        or int(length) != reference.byte_size
    ):
        raise OutputIntegrityError("Output content length mismatch")


class Verification:
    def __init__(self, options: TransferOptions) -> None:
        self.options = options
        self.started = time.monotonic()
        self.deadline = self.started + options.timeout_seconds
        self.digest = sha256()
        self.byte_size = 0
        self.first_byte: float | None = None
        self.reference: OutputRef | None = None
        self.timing: TransferTiming | None = None

    def remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise RequestTimeoutError("Output transfer deadline expired")
        return remaining

    def consume(self, chunk: bytes) -> None:
        self.remaining()
        if self.reference is None:
            raise RuntimeError("Output descriptor has not been read")
        self.byte_size += len(chunk)
        if self.byte_size > min(self.options.maximum_bytes, self.reference.byte_size):
            raise OutputIntegrityError("Output exceeded its declared size")
        if chunk and self.first_byte is None:
            self.first_byte = time.monotonic()
        self.digest.update(chunk)

    def finish(self) -> None:
        self.remaining()
        if (
            self.reference is None
            or self.byte_size != self.reference.byte_size
            or self.digest.hexdigest() != self.reference.digest_sha256
        ):
            raise OutputIntegrityError("Output size or digest verification failed")
        self.timing = TransferTiming(
            total_ms=(time.monotonic() - self.started) * 1000,
            first_byte_ms=None
            if self.first_byte is None
            else (self.first_byte - self.started) * 1000,
        )


class OutputStream:
    """Single-consumer lazy stream. Check verified after EOF; early close is unverified."""

    def __init__(
        self, transport: HttpTransport, run_id: str, output_id: str, options: TransferOptions
    ) -> None:
        self._transport = transport
        self._run_id = run_id
        self._output_id = output_id
        self._options = options
        self._state: Verification | None = None
        self._descriptor: Output | None = None
        self._response: httpx.Response | None = None
        self._iterator: Iterator[bytes] | None = None
        self._closed = False
        self._reading = False
        self._read_lock = threading.Lock()
        self._reader_thread: int | None = None
        self._reader_finished = threading.Event()
        self._reader_finished.set()
        self._close_finished = threading.Event()
        self._close_error: BaseException | None = None
        self._scope = CancellationScope()

    @contextmanager
    def _read(self) -> Generator[None, None, None]:
        with self._read_lock:
            if self._reading:
                raise RuntimeError("Output stream permits one active reader")
            self._reading = True
            self._reader_thread = threading.get_ident()
            self._reader_finished.clear()
        try:
            with self._scope.activate():
                yield
        finally:
            with self._read_lock:
                self._reading = False
                self._reader_thread = None
                self._reader_finished.set()

    @property
    def verified(self) -> bool:
        return self._state is not None and self._state.timing is not None

    @property
    def bytes_read(self) -> int:
        return self._state.byte_size if self._state else 0

    @property
    def descriptor(self) -> Output:
        if self._descriptor is None:
            raise RuntimeError("Output stream has not opened")
        return self._descriptor

    @property
    def reference(self) -> OutputRef:
        if self._state is None or self._state.reference is None:
            raise RuntimeError("Output stream has not opened")
        return self._state.reference

    @property
    def timing(self) -> TransferTiming:
        if self._state is None or self._state.timing is None:
            raise RuntimeError("Output transfer has not been verified")
        return self._state.timing

    def _open(self) -> None:
        if self._closed:
            raise RuntimeError("Output stream is closed")
        if self._state is not None:
            return
        self._state = Verification(self._options)
        state = self._state
        try:
            self._transport._register_stream(self)
            path = f"/swarms/runs/{self._run_id}/outputs/{self._output_id}"
            self._descriptor = self._transport.request(
                operations.OUTPUT,
                Output,
                path=path,
                options=ReadOptions(timeout_seconds=state.remaining(), read_retry_count=0),
            )
            if self._closed:
                raise RuntimeError("Output stream closed while reading its descriptor")
            state.reference = available(
                self._descriptor, self._run_id, self._output_id, self._options.maximum_bytes
            )
            reference = state.reference
            response = self._transport._open_stream(
                operations.OUTPUT_CONTENT,
                path=path + "/content",
                query={
                    "revision_id": reference.revision_id,
                    "digest_sha256": reference.digest_sha256,
                },
                remaining_seconds=state.remaining(),
                digest=reference.digest_sha256,
            )
            self._response = response
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise OutputIntegrityError("Compressed output is outside the bounded contract")
            if response.status_code != 200:
                body = bytearray()
                for chunk in response.iter_bytes(chunk_size=self._options.chunk_bytes):
                    state.remaining()
                    if len(body) + len(chunk) > min(65536, self._transport.maximum_response_bytes):
                        raise OutputIntegrityError("Output error exceeded the response ceiling")
                    body.extend(chunk)
                decode_response(
                    response,
                    bytes(body),
                    ResponseModel,
                    ClientTiming(total_ms=0, first_byte_ms=0, attempts=1),
                    None,
                    False,
                    "swarms.v1",
                )
                raise OutputIntegrityError("Unexpected output response")
            validate_headers(response, reference)
            self._iterator = response.iter_bytes(chunk_size=self._options.chunk_bytes)
        except httpx.TimeoutException as cause:
            primary = RequestTimeoutError("Output transfer timed out")
            self._close_after_failure(primary)
            raise primary from cause
        except httpx.TransportError as cause:
            primary = TransportError("Output transfer could not open")
            self._close_after_failure(primary)
            raise primary from cause
        except BaseException as primary:
            self._close_after_failure(primary)
            raise

    def __iter__(self) -> OutputStream:
        return self

    def __next__(self) -> bytes:
        with self._read():
            return self._next()

    def _next(self) -> bytes:
        if self._closed:
            raise StopIteration
        try:
            self._open()
            if self._iterator is None or self._state is None:
                raise RuntimeError("Output stream did not initialize")
            try:
                chunk = next(self._iterator)
            except StopIteration:
                self._state.finish()
                self.close()
                raise
            if self._closed:
                raise StopIteration
            self._state.consume(chunk)
            return chunk
        except httpx.TimeoutException as cause:
            primary = RequestTimeoutError("Output transfer timed out")
            self._close_after_failure(primary)
            raise primary from cause
        except httpx.TransportError as cause:
            primary = TransportError("Output transfer interrupted; no automatic retry")
            self._close_after_failure(primary)
            raise primary from cause
        except BaseException as primary:
            self._close_after_failure(primary)
            raise

    def _close_after_failure(self, primary: BaseException) -> None:
        try:
            self.close()
        except Exception as cause:
            handle_cleanup_failure(cause, primary, mutation=False, identity=None)

    def close(self) -> None:
        with self._read_lock:
            own_reader = self._reader_thread == threading.get_ident()
            closing = not self._closed
            self._closed = True
        if not closing:
            if own_reader:
                return
            if not self._close_finished.wait(4 * CLEANUP_SECONDS):
                raise RuntimeError("Output stream cleanup did not complete")
            if self._close_error is not None:
                raise self._close_error
            return
        try:
            with self._scope.cleanup():
                self._scope.cancel()
                if not own_reader and not self._reader_finished.wait(CLEANUP_SECONDS):
                    raise RuntimeError("Output stream reader did not close")
                if self._response is not None:
                    self._transport._release_response(self._response)
        except BaseException as error:
            self._close_error = error
            raise
        finally:
            self._transport._unregister_stream(self)
            self._close_finished.set()

    def __enter__(self) -> OutputStream:
        with self._read():
            self._open()
            return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exception is not None:
            self._close_after_failure(exception)
        else:
            self.close()


class AsyncOutputStream:
    """Single-consumer lazy stream. Check verified after EOF; early close is unverified."""

    def __init__(
        self, transport: AsyncHttpTransport, run_id: str, output_id: str, options: TransferOptions
    ) -> None:
        self._transport = transport
        self._run_id = run_id
        self._output_id = output_id
        self._options = options
        self._state: Verification | None = None
        self._descriptor: Output | None = None
        self._response: httpx.Response | None = None
        self._iterator: AsyncIterator[bytes] | None = None
        self._closed = False
        self._reading = False
        self._closing_task: asyncio.Task[None] | None = None
        self._reading_task: asyncio.Task[object] | None = None

    @property
    def verified(self) -> bool:
        return self._state is not None and self._state.timing is not None

    @property
    def bytes_read(self) -> int:
        return self._state.byte_size if self._state else 0

    @property
    def descriptor(self) -> Output:
        if self._descriptor is None:
            raise RuntimeError("Output stream has not opened")
        return self._descriptor

    @property
    def reference(self) -> OutputRef:
        if self._state is None or self._state.reference is None:
            raise RuntimeError("Output stream has not opened")
        return self._state.reference

    @property
    def timing(self) -> TransferTiming:
        if self._state is None or self._state.timing is None:
            raise RuntimeError("Output transfer has not been verified")
        return self._state.timing

    async def _open(self) -> None:
        try:
            timeout = (
                self._state.remaining()
                if self._state is not None
                else self._options.timeout_seconds
            )
            async with asyncio.timeout(timeout):
                await self._open_response()
        except TimeoutError as cause:
            primary = RequestTimeoutError("Output opening deadline expired")
            await self._close_after_failure(primary)
            raise primary from cause

    async def _open_response(self) -> None:
        if self._closed:
            raise RuntimeError("Output stream is closed")
        if self._state is not None:
            return
        self._state = Verification(self._options)
        state = self._state
        try:
            self._transport._register_stream(self)
            path = f"/swarms/runs/{self._run_id}/outputs/{self._output_id}"
            self._descriptor = await self._transport.request(
                operations.OUTPUT,
                Output,
                path=path,
                options=ReadOptions(timeout_seconds=state.remaining(), read_retry_count=0),
            )
            if self._closed:
                raise RuntimeError("Output stream closed while reading its descriptor")
            state.reference = available(
                self._descriptor, self._run_id, self._output_id, self._options.maximum_bytes
            )
            reference = state.reference
            response = await self._transport._open_stream(
                operations.OUTPUT_CONTENT,
                path=path + "/content",
                query={
                    "revision_id": reference.revision_id,
                    "digest_sha256": reference.digest_sha256,
                },
                remaining_seconds=state.remaining(),
                digest=reference.digest_sha256,
            )
            self._response = response
            if self._closed:
                await self._transport._release_response(response)
                raise RuntimeError("Output stream closed during opening")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise OutputIntegrityError("Compressed output is outside the bounded contract")
            if response.status_code != 200:
                body = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=self._options.chunk_bytes):
                    state.remaining()
                    if len(body) + len(chunk) > min(65536, self._transport.maximum_response_bytes):
                        raise OutputIntegrityError("Output error exceeded the response ceiling")
                    body.extend(chunk)
                decode_response(
                    response,
                    bytes(body),
                    ResponseModel,
                    ClientTiming(total_ms=0, first_byte_ms=0, attempts=1),
                    None,
                    False,
                    "swarms.v1",
                )
                raise OutputIntegrityError("Unexpected output response")
            validate_headers(response, reference)
            self._iterator = response.aiter_bytes(chunk_size=self._options.chunk_bytes)
        except httpx.TimeoutException as cause:
            primary = RequestTimeoutError("Output transfer timed out")
            await self._close_after_failure(primary)
            raise primary from cause
        except httpx.TransportError as cause:
            primary = TransportError("Output transfer could not open")
            await self._close_after_failure(primary)
            raise primary from cause
        except BaseException as primary:
            await self._close_after_failure(primary)
            raise

    def __aiter__(self) -> AsyncOutputStream:
        return self

    async def __anext__(self) -> bytes:
        if self._reading:
            raise RuntimeError("Output stream permits one active reader")
        self._reading = True
        task = asyncio.create_task(self._next())
        self._reading_task = task
        try:
            return await task
        finally:
            self._reading_task = None
            self._reading = False

    async def _next(self) -> bytes:
        if self._closed:
            raise StopAsyncIteration
        try:
            await self._open()
            if self._iterator is None or self._state is None:
                raise RuntimeError("Output stream did not initialize")
            try:
                async with asyncio.timeout(self._state.remaining()):
                    chunk = await anext(self._iterator)
            except StopAsyncIteration:
                self._state.finish()
                await self.aclose()
                raise
            if self._closed:
                raise StopAsyncIteration
            self._state.consume(chunk)
            return chunk
        except (TimeoutError, httpx.TimeoutException) as cause:
            primary = RequestTimeoutError("Output transfer deadline expired")
            await self._close_after_failure(primary)
            raise primary from cause
        except httpx.TransportError as cause:
            primary = TransportError("Output transfer interrupted; no automatic retry")
            await self._close_after_failure(primary)
            raise primary from cause
        except BaseException as primary:
            await self._close_after_failure(primary)
            raise

    async def _close_after_failure(self, primary: BaseException) -> None:
        try:
            await self.aclose()
        except (Exception, asyncio.CancelledError) as cause:
            handle_cleanup_failure(cause, primary, mutation=False, identity=None)

    async def aclose(self) -> None:
        # An external close owns cancellation; its reader must unwind without
        # waiting on the task that is draining that same reader.
        if self._closing_task is not None and asyncio.current_task() is self._reading_task:
            return
        if self._closing_task is None:
            self._closed = True
            reader = self._reading_task
            if reader is asyncio.current_task():
                reader = None
            self._closing_task = asyncio.create_task(self._close_response(reader))
        await await_cleanup(self._closing_task)

    async def _close_response(self, reader: asyncio.Task[object] | None) -> None:
        try:
            async with asyncio.timeout(self._transport.request_timeout_seconds):
                if reader is not None:
                    reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)
                if self._response is not None:
                    await self._transport._release_response(self._response)
        finally:
            self._transport._unregister_stream(self)

    async def __aenter__(self) -> AsyncOutputStream:
        if self._reading:
            raise RuntimeError("Output stream permits one active reader")
        self._reading = True
        task = asyncio.create_task(self._open())
        self._reading_task = task
        try:
            await task
            return self
        finally:
            self._reading_task = None
            self._reading = False

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exception is not None:
            await self._close_after_failure(exception)
        else:
            await self.aclose()
