"""Owned synchronous facade over cancellable async network IO.

See SDK_0230_IMPLEMENTATION.md. No user callback runs on this private loop.
Each network phase shares the request's monotonic deadline. A timeout waits for
cancellation acknowledgement before returning, so no request continues in a
background thread after the caller receives its outcome.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Coroutine, Generator, Iterator
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypeVar

import httpx

T = TypeVar("T")
CLEANUP_SECONDS = 1.0
_OBSERVATION_SCOPE: ContextVar[CancellationScope | None] = ContextVar(
    "synth_observation", default=None
)


class ScopedCall:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.finished = threading.Event()
        self._lock = threading.Lock()
        self._task: asyncio.Task[object] | None = None
        self._cancelled = False

    def start(self, task: asyncio.Task[object] | None) -> bool:
        with self._lock:
            self._task = task
            return not self._cancelled

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            if self._task is not None:
                self.loop.call_soon_threadsafe(self._task.cancel)


class CancellationScope:
    """Cancel only network phases owned by one synchronous observation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: set[ScopedCall] = set()
        self._closed = False

    @contextmanager
    def activate(self) -> Generator[None, None, None]:
        token = _OBSERVATION_SCOPE.set(self)
        try:
            yield
        finally:
            _OBSERVATION_SCOPE.reset(token)

    @contextmanager
    def cleanup(self) -> Generator[None, None, None]:
        token = _OBSERVATION_SCOPE.set(None)
        try:
            yield
        finally:
            _OBSERVATION_SCOPE.reset(token)

    def register(self, call: ScopedCall) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("SDK observation is closed")
            self._calls.add(call)

    def release(self, call: ScopedCall) -> None:
        with self._lock:
            self._calls.discard(call)

    def cancel(self) -> None:
        with self._lock:
            self._closed = True
            calls = tuple(self._calls)
        for call in calls:
            call.cancel()
        deadline = time.monotonic() + CLEANUP_SECONDS
        for call in calls:
            if not call.finished.wait(max(0, deadline - time.monotonic())):
                raise RuntimeError("SDK observation cancellation did not complete")


class BlockingLoop:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._serve, name="synth-http", daemon=True)
        self._ready = threading.Event()
        self._closed = False
        self._thread.start()
        if not self._ready.wait(CLEANUP_SECONDS):
            raise RuntimeError("SDK network loop did not start")

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    def call(self, operation: Coroutine[None, None, T], timeout: float) -> T:
        if self._closed:
            operation.close()
            raise RuntimeError("SDK network loop is closed")
        if threading.current_thread() is self._thread:
            operation.close()
            raise RuntimeError("Synchronous SDK calls cannot reenter their network loop")
        finished = threading.Event()
        scope = _OBSERVATION_SCOPE.get()
        owned = ScopedCall(self._loop) if scope is not None else None
        if scope is not None and owned is not None:
            try:
                scope.register(owned)
            except BaseException:
                operation.close()
                raise

        async def bounded() -> T:
            try:
                if owned is not None and not owned.start(asyncio.current_task()):
                    operation.close()
                    raise asyncio.CancelledError
                async with asyncio.timeout(max(0, timeout)):
                    return await operation
            finally:
                finished.set()
                if owned is not None:
                    owned.finished.set()

        exchange = bounded()
        try:
            future = asyncio.run_coroutine_threadsafe(exchange, self._loop)
        except BaseException:
            exchange.close()
            operation.close()
            if scope is not None and owned is not None:
                scope.release(owned)
                owned.finished.set()
            raise
        try:
            return future.result(timeout=max(0, timeout) + CLEANUP_SECONDS)
        except FutureTimeout:
            if future.done():
                return future.result()
            future.cancel()
            if not finished.wait(CLEANUP_SECONDS):
                raise RuntimeError("SDK network cancellation did not complete") from None
            raise TimeoutError("SDK network deadline expired") from None
        except BaseException:
            if not future.done():
                future.cancel()
                if not finished.wait(CLEANUP_SECONDS):
                    raise RuntimeError("SDK network cancellation did not complete") from None
            raise
        finally:
            if scope is not None and owned is not None:
                scope.release(owned)

    def close(self) -> None:
        if self._closed:
            return

        async def shutdown() -> None:
            current = asyncio.current_task()
            tasks = [task for task in asyncio.all_tasks() if task is not current]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await self._loop.shutdown_asyncgens()

        try:
            self.call(shutdown(), CLEANUP_SECONDS)
        finally:
            self._closed = True
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(CLEANUP_SECONDS)
            if self._thread.is_alive():
                raise RuntimeError("SDK network loop did not close")


class ResponseStream(httpx.SyncByteStream):
    def __init__(self, response: httpx.Response, loop: BlockingLoop, deadline: float) -> None:
        self._response = response
        self._loop = loop
        self._deadline = deadline
        self._closed = False

    def __iter__(self) -> Iterator[bytes]:
        iterator = self._response.aiter_raw()

        async def next_chunk() -> bytes | None:
            return await anext(iterator, None)

        try:
            while True:
                chunk = self._loop.call(next_chunk(), self._deadline - time.monotonic())
                if chunk is None:
                    return
                yield chunk
        except TimeoutError as error:
            raise httpx.ReadTimeout("Response total deadline expired") from error

    def close(self) -> None:
        if self._closed:
            return
        token = _OBSERVATION_SCOPE.set(None)
        try:
            self._loop.call(self._response.aclose(), CLEANUP_SECONDS)
        finally:
            _OBSERVATION_SCOPE.reset(token)
        self._closed = True


class DeadlineTransport(httpx.BaseTransport):
    """The SDK owns both the connection pool and the loop that cancels its IO."""

    def __init__(self, maximum_connections: int) -> None:
        self._maximum_connections = maximum_connections
        self._loop: BlockingLoop | None = None
        self._client = self._create_client()
        self._lock = threading.Lock()
        self._closed = False
        self._close_finished = threading.Event()
        self._close_error: BaseException | None = None

    def _get_client(self) -> tuple[BlockingLoop, httpx.AsyncClient]:
        with self._lock:
            if self._closed:
                raise RuntimeError("SDK transport is closed")
            if self._loop is None:
                self._loop = BlockingLoop()

            return self._loop, self._client

    def _create_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            trust_env=True,
            auth=None,
            limits=httpx.Limits(max_connections=self._maximum_connections),
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        started = time.monotonic()
        loop, client = self._get_client()
        # Request construction is SDK-owned and always sets all timeout phases.
        timeout = request.extensions.get("timeout")
        if not isinstance(timeout, dict):
            raise ValueError("SDK request requires an explicit deadline")
        seconds = timeout.get("read")
        if not isinstance(seconds, (float, int)) or seconds <= 0:
            raise httpx.ReadTimeout("Request total deadline expired", request=request)
        deadline = started + seconds
        seconds = max(0, deadline - time.monotonic())
        try:
            response = loop.call(
                client.send(request, stream=True, auth=None, follow_redirects=False), seconds
            )
        except TimeoutError as error:
            raise httpx.ReadTimeout("Request total deadline expired", request=request) from error
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            stream=ResponseStream(response, loop, deadline),
            extensions=response.extensions,
            request=request,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                closing = False
            else:
                self._closed = True
                closing = True
        if not closing:
            if not self._close_finished.wait(4 * CLEANUP_SECONDS):
                raise RuntimeError("SDK transport cleanup did not complete")
            if self._close_error is not None:
                raise self._close_error
            return
        try:
            loop, client = self._loop or BlockingLoop(), self._client
            try:
                loop.call(client.aclose(), CLEANUP_SECONDS)
            finally:
                loop.close()
        except BaseException as error:
            self._close_error = error
            raise
        finally:
            self._close_finished.set()
