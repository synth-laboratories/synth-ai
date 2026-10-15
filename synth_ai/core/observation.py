"""Product-neutral bounded event observation. See backend SDK surface.md."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Generator, Iterator
from contextlib import contextmanager
from hashlib import sha256
from types import TracebackType
from typing import Protocol, TypeVar

from synth_ai.core.http.blocking import CLEANUP_SECONDS, CancellationScope
from synth_ai.core.http.cleanup import handle_cleanup_failure
from synth_ai.core.http.lifetime import AsyncStreamOwner, StreamOwner, await_cleanup
from synth_ai.errors import ContractError, RequestTimeoutError
from synth_ai.options import ObservationOptions, ReadOptions
from synth_ai.types.base import EventCursor, Page


class EventValue(Protocol):
    @property
    def event_id(self) -> str: ...
    @property
    def sequence(self) -> int: ...
    def model_dump_json(self) -> str: ...


E = TypeVar("E", bound=EventValue)


class PageReader(Protocol[E]):
    def __call__(
        self, *, limit: int, cursor: EventCursor | None, request: ReadOptions | None = None
    ) -> Page[E, EventCursor]: ...


class AsyncPageReader(Protocol[E]):
    async def __call__(
        self, *, limit: int, cursor: EventCursor | None, request: ReadOptions | None = None
    ) -> Page[E, EventCursor]: ...


class EventStream(Iterator[E]):
    """Lazy managed stream; close stops observation only. All storage is bounded."""

    def __init__(
        self,
        fetch_page: PageReader[E],
        cursor: EventCursor | None,
        observation: ObservationOptions,
        *,
        owner: StreamOwner | None = None,
    ) -> None:
        self._fetch_page = fetch_page
        self._cursor = cursor
        self._options = observation
        self._deadline: float | None = None
        self._last_activity: float | None = None
        self._buffer: deque[E] = deque()
        self._seen: dict[str, str] = {}
        self._sequence = 0
        self._complete = False
        self._closed = False
        self._owner = owner
        self._registered = False
        self._reading = False
        self._read_lock = threading.Lock()
        self._reader_thread: int | None = None
        self._reader_finished = threading.Event()
        self._reader_finished.set()
        self._close_finished = threading.Event()
        self._close_error: BaseException | None = None
        self._scope = CancellationScope()
        self._wake = threading.Event()

    @contextmanager
    def _read(self) -> Generator[None, None, None]:
        with self._read_lock:
            if self._reading:
                raise RuntimeError("Event stream permits one active reader")
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

    def __iter__(self) -> EventStream[E]:
        return self

    def __next__(self) -> E:
        with self._read():
            return self._next()

    def _next(self) -> E:
        if self._closed:
            raise StopIteration
        now = time.monotonic()
        if self._deadline is None:
            self._deadline = now + self._options.timeout_seconds
            self._last_activity = now
        try:
            if self._owner is not None and not self._registered:
                self._owner._register_stream(self)
                self._registered = True
            if time.monotonic() >= self._deadline:
                raise RequestTimeoutError("Event observation deadline expired")
            while not self._buffer:
                if self._closed:
                    raise StopIteration
                if self._complete:
                    self.close()
                    raise StopIteration
                remaining = self._deadline - time.monotonic()
                idle = self._options.idle_timeout_seconds - (
                    time.monotonic()
                    - (self._last_activity if self._last_activity is not None else now)
                )
                if min(remaining, idle) <= 0:
                    raise RequestTimeoutError("Event observation deadline expired")
                page = self._fetch_page(
                    limit=min(200, self._options.maximum_events),
                    cursor=self._cursor,
                    request=ReadOptions(
                        timeout_seconds=min(
                            remaining, idle, self._options.reconnect_deadline_seconds
                        ),
                        read_retry_count=self._options.retry_count_max,
                    ),
                )
                if self._closed:
                    raise StopIteration
                if time.monotonic() >= self._deadline:
                    raise RequestTimeoutError("Event observation deadline expired")
                for event in page.items:
                    fingerprint = sha256(event.model_dump_json().encode("utf-8")).hexdigest()
                    prior = self._seen.get(event.event_id)
                    if prior is not None:
                        if prior != fingerprint:
                            raise ContractError("Replayed event changed its payload")
                        continue
                    if event.sequence <= self._sequence:
                        raise ContractError("Event order regressed")
                    if len(self._seen) >= self._options.maximum_events:
                        raise ContractError("Event observation exceeded its event bound")
                    self._seen[event.event_id] = fingerprint
                    self._sequence = event.sequence
                    self._buffer.append(event)
                    self._last_activity = time.monotonic()
                self._cursor = page.next_cursor
                self._complete = page.complete
                if not page.complete and page.next_cursor is None:
                    raise ContractError("Incomplete event page requires a resume cursor")
                if not self._buffer and (not page.complete):
                    self._wake.wait(
                        min(
                            self._options.poll_interval_seconds,
                            max(0, self._deadline - time.monotonic()),
                            max(
                                0,
                                self._options.idle_timeout_seconds
                                - (
                                    time.monotonic()
                                    - (
                                        self._last_activity
                                        if self._last_activity is not None
                                        else now
                                    )
                                ),
                            ),
                        )
                    )
            return self._buffer.popleft()
        except BaseException as primary:
            try:
                self.close()
            except Exception as cause:
                handle_cleanup_failure(cause, primary, mutation=False, identity=None)
            raise

    def close(self) -> None:
        """End local observation without cancelling durable remote work.

        See: backend/notes/specifications/sdk/index_contract.md."""
        with self._read_lock:
            own_reader = self._reader_thread == threading.get_ident()
            closing = not self._closed
            self._closed = True
            self._wake.set()
        if not closing:
            if own_reader:
                return
            if not self._close_finished.wait(2 * CLEANUP_SECONDS):
                raise RuntimeError("Event stream cleanup did not complete")
            if self._close_error is not None:
                raise self._close_error
            return
        completed = False
        try:
            self._scope.cancel()
            if not own_reader and not self._reader_finished.wait(CLEANUP_SECONDS):
                raise RuntimeError("Event stream reader did not close")
            completed = True
        except BaseException as error:
            self._close_error = error
            raise
        finally:
            if completed and self._owner is not None and self._registered:
                self._owner._unregister_stream(self)
                self._registered = False
            if completed:
                self._buffer.clear()
                self._seen.clear()
            self._close_finished.set()

    def __enter__(self) -> EventStream[E]:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exception is not None:
            try:
                self.close()
            except Exception as cause:
                handle_cleanup_failure(cause, exception, mutation=False, identity=None)
        else:
            self.close()


class AsyncEventStream(AsyncIterator[E]):
    """Native async observer with explicit lifetime and no background task."""

    def __init__(
        self,
        fetch_page: AsyncPageReader[E],
        cursor: EventCursor | None,
        observation: ObservationOptions,
        *,
        owner: AsyncStreamOwner | None = None,
    ) -> None:
        self._fetch_page = fetch_page
        self._cursor = cursor
        self._options = observation
        self._deadline: float | None = None
        self._last_activity: float | None = None
        self._buffer: deque[E] = deque()
        self._seen: dict[str, str] = {}
        self._sequence = 0
        self._complete = False
        self._closed = False
        self._owner = owner
        self._registered = False
        self._reading = False
        self._reading_task: asyncio.Task[object] | None = None
        self._closing_task: asyncio.Task[None] | None = None

    def __aiter__(self) -> AsyncEventStream[E]:
        return self

    async def __anext__(self) -> E:
        if self._reading:
            raise RuntimeError("Event stream permits one active reader")
        self._reading = True
        task = asyncio.create_task(self._next())
        self._reading_task = task
        try:
            return await task
        finally:
            self._reading_task = None
            self._reading = False

    async def _next(self) -> E:
        if self._closed:
            raise StopAsyncIteration
        now = time.monotonic()
        if self._deadline is None:
            self._deadline = now + self._options.timeout_seconds
            self._last_activity = now
        try:
            if self._owner is not None and not self._registered:
                self._owner._register_stream(self)
                self._registered = True
            if time.monotonic() >= self._deadline:
                raise RequestTimeoutError("Event observation deadline expired")
            while not self._buffer:
                if self._closed:
                    raise StopAsyncIteration
                if self._complete:
                    await self.aclose()
                    raise StopAsyncIteration
                remaining = self._deadline - time.monotonic()
                idle = self._options.idle_timeout_seconds - (
                    time.monotonic()
                    - (self._last_activity if self._last_activity is not None else now)
                )
                if min(remaining, idle) <= 0:
                    raise RequestTimeoutError("Event observation deadline expired")
                async with asyncio.timeout(
                    min(remaining, idle, self._options.reconnect_deadline_seconds)
                ):
                    page = await self._fetch_page(
                        limit=min(200, self._options.maximum_events),
                        cursor=self._cursor,
                        request=ReadOptions(
                            timeout_seconds=min(
                                remaining, idle, self._options.reconnect_deadline_seconds
                            ),
                            read_retry_count=self._options.retry_count_max,
                        ),
                    )
                if self._closed:
                    raise StopAsyncIteration
                if time.monotonic() >= self._deadline:
                    raise RequestTimeoutError("Event observation deadline expired")
                for event in page.items:
                    fingerprint = sha256(event.model_dump_json().encode("utf-8")).hexdigest()
                    prior = self._seen.get(event.event_id)
                    if prior is not None:
                        if prior != fingerprint:
                            raise ContractError("Replayed event changed its payload")
                        continue
                    if event.sequence <= self._sequence:
                        raise ContractError("Event order regressed")
                    if len(self._seen) >= self._options.maximum_events:
                        raise ContractError("Event observation exceeded its event bound")
                    self._seen[event.event_id] = fingerprint
                    self._sequence = event.sequence
                    self._buffer.append(event)
                    self._last_activity = time.monotonic()
                self._cursor = page.next_cursor
                self._complete = page.complete
                if not page.complete and page.next_cursor is None:
                    raise ContractError("Incomplete event page requires a resume cursor")
                if not self._buffer and (not page.complete):
                    await asyncio.sleep(
                        min(
                            self._options.poll_interval_seconds,
                            max(0, self._deadline - time.monotonic()),
                            max(
                                0,
                                self._options.idle_timeout_seconds
                                - (
                                    time.monotonic()
                                    - (
                                        self._last_activity
                                        if self._last_activity is not None
                                        else now
                                    )
                                ),
                            ),
                        )
                    )
            return self._buffer.popleft()
        except TimeoutError as cause:
            primary = RequestTimeoutError("Event observation deadline expired")
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
        """End async local ownership without cancelling durable work or closing borrowed clients.

        See: backend/notes/specifications/sdk/index_contract.md."""
        if self._closing_task is not None and asyncio.current_task() is self._reading_task:
            return
        if self._closing_task is None:
            self._closed = True
            reader = self._reading_task
            if reader is asyncio.current_task():
                reader = None
            self._closing_task = asyncio.create_task(self._close_owned(reader))
        await await_cleanup(self._closing_task)

    async def _close_owned(self, reader: asyncio.Task[object] | None) -> None:
        completed = False
        try:
            if reader is not None:
                reader.cancel()
                done, _ = await asyncio.wait({reader}, timeout=self._options.idle_timeout_seconds)
                if not done:
                    raise RuntimeError("Event stream reader cancellation did not complete")
                if not reader.cancelled():
                    reader.exception()
            completed = True
        finally:
            if completed and self._owner is not None and self._registered:
                self._owner._unregister_stream(self)
                self._registered = False
            if completed:
                self._buffer.clear()
                self._seen.clear()

    async def __aenter__(self) -> AsyncEventStream[E]:
        return self

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
