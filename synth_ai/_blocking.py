"""Dedicated loop preserves async connection ownership, including in notebook callers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Coroutine, Iterator
from concurrent.futures import Future
from threading import Event, Thread
from typing import TypeVar

from .errors import ClientClosedError

T = TypeVar("T")


class Runner:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        ready = Event()

        def run() -> None:
            asyncio.set_event_loop(self.loop)
            ready.set()
            self.loop.run_forever()

        self.thread = Thread(target=run, name="synth-client", daemon=True)
        self.thread.start()
        ready.wait()
        self.closed = False

    def call(self, operation: Coroutine[object, object, T]) -> T:
        if self.closed:
            operation.close()
            raise ClientClosedError("client is closed")
        future: Future[T] = asyncio.run_coroutine_threadsafe(operation, self.loop)
        try:
            return future.result()
        except BaseException:
            future.cancel()
            raise

    def iterate(self, source: AsyncGenerator[T, None]) -> Iterator[T]:
        async def advance() -> T:
            return await anext(source)

        try:
            while True:
                try:
                    yield self.call(advance())
                except StopAsyncIteration:
                    return
        finally:
            self.call(source.aclose())

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()
