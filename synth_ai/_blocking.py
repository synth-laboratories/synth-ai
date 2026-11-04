"""Dedicated loop preserves async connection ownership, including in notebook callers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine, Iterator
from concurrent.futures import Future
from threading import Event, Thread
from typing import Any, TypeVar

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

    def call(self, operation: Coroutine[Any, Any, T]) -> T:
        if self.closed:
            operation.close()
            raise RuntimeError("client is closed")
        future: Future[T] = asyncio.run_coroutine_threadsafe(operation, self.loop)
        return future.result()

    def iterate(self, source: AsyncIterator[T]) -> Iterator[T]:
        async def advance() -> T:
            return await anext(source)

        try:
            while True:
                try:
                    yield self.call(advance())
                except StopAsyncIteration:
                    return
        finally:
            close = getattr(source, "aclose", None)
            if close is not None:
                self.call(close())

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()
