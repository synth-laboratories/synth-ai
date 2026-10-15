"""Owned response lifetimes for otherwise borrowed HTTP clients."""

import asyncio
from typing import Protocol


async def await_cleanup(task: asyncio.Task[None]) -> None:
    """Retain cleanup ownership and surface cancellation only after its bounded task ends."""
    cancelled: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(task)
            break
        except asyncio.CancelledError as error:
            if task.cancelled():
                raise
            cancelled = error
        except Exception as error:
            if cancelled is not None:
                cancelled.add_note(f"SDK cleanup failed: {type(error).__name__}")
                raise cancelled from error
            raise
    if cancelled is not None:
        raise cancelled


class Closeable(Protocol):
    def close(self) -> None: ...


class AsyncCloseable(Protocol):
    async def aclose(self) -> None: ...


class StreamOwner(Protocol):
    def _register_stream(self, stream: Closeable) -> None: ...
    def _unregister_stream(self, stream: Closeable) -> None: ...


class AsyncStreamOwner(Protocol):
    def _register_stream(self, stream: AsyncCloseable) -> None: ...
    def _unregister_stream(self, stream: AsyncCloseable) -> None: ...
