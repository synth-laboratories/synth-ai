"""Product-neutral bounded event observation. See backend SDK surface.md."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Iterator
from hashlib import sha256
from types import TracebackType
from typing import Protocol, TypeVar

from synth_ai.core.http.lifetime import AsyncStreamOwner, StreamOwner
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

    def __iter__(self) -> EventStream[E]:
        return self

    def __next__(self) -> E:
        if self._reading:
            raise RuntimeError("Event stream permits one active reader")
        self._reading = True
        try:
            return self._next()
        finally:
            self._reading = False

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
                    time.sleep(
                        min(
                            self._options.poll_interval_seconds,
                            max(0, self._deadline - time.monotonic()),
                        )
                    )
            return self._buffer.popleft()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """End local observation without cancelling durable remote work.

        See: backend/notes/specifications/sdk/index_contract.md."""
        self._closed = True
        if self._owner is not None and self._registered:
            self._owner._unregister_stream(self)
            self._registered = False
        self._buffer.clear()
        self._seen.clear()

    def __enter__(self) -> EventStream[E]:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
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

    def __aiter__(self) -> AsyncEventStream[E]:
        return self

    async def __anext__(self) -> E:
        if self._reading:
            raise RuntimeError("Event stream permits one active reader")
        self._reading = True
        try:
            return await self._next()
        finally:
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
                        )
                    )
            return self._buffer.popleft()
        except TimeoutError as cause:
            await self.aclose()
            raise RequestTimeoutError("Event observation deadline expired") from cause
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        """End async local ownership without cancelling durable work or closing borrowed clients.

        See: backend/notes/specifications/sdk/index_contract.md."""
        self._closed = True
        if self._owner is not None and self._registered:
            self._owner._unregister_stream(self)
            self._registered = False
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
        await self.aclose()
