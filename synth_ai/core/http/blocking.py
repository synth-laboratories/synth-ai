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
from collections.abc import Coroutine, Iterator
from concurrent.futures import TimeoutError as FutureTimeout
from typing import TypeVar

import httpx

T = TypeVar("T")
CLEANUP_SECONDS = 1.0


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
        finished = threading.Event()

        async def bounded() -> T:
            try:
                async with asyncio.timeout(max(0, timeout)):
                    return await operation
            finally:
                finished.set()

        if self._closed:
            operation.close()
            raise RuntimeError("SDK network loop is closed")
        if threading.current_thread() is self._thread:
            operation.close()
            raise RuntimeError("Synchronous SDK calls cannot reenter their network loop")
        future = asyncio.run_coroutine_threadsafe(bounded(), self._loop)
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
        self._loop.call(self._response.aclose(), CLEANUP_SECONDS)
        self._closed = True


class DeadlineTransport(httpx.BaseTransport):
    """The SDK owns both the connection pool and the loop that cancels its IO."""

    def __init__(self, maximum_connections: int) -> None:
        self._maximum_connections = maximum_connections
        self._loop: BlockingLoop | None = None
        self._client = self._create_client()
        self._lock = threading.Lock()
        self._closed = False

    def _get_client(self) -> tuple[BlockingLoop, httpx.AsyncClient]:
        with self._lock:
            if self._closed:
                raise RuntimeError("SDK transport is closed")
            if self._loop is None:
                self._loop = BlockingLoop()

            return self._loop, self._client

    def _create_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            trust_env=False,
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
                return
            self._closed = True
            loop, client = self._loop or BlockingLoop(), self._client
        if loop is not None:
            try:
                if client is not None:
                    loop.call(client.aclose(), CLEANUP_SECONDS)
            finally:
                loop.close()
