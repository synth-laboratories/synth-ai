from __future__ import annotations

import asyncio
import contextlib
from typing import Any, AsyncIterator

from synth_ai.core.errors import HTTPError

try:
    import synth_ai_py as _synth_ai_py
except Exception:  # pragma: no cover - optional rust bindings
    _synth_ai_py = None


async def stream_sse_events(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    json_payload: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> AsyncIterator[dict[str, Any]]:
    if _synth_ai_py is None:
        raise HTTPError(
            status=500,
            url=url,
            message="sse_stream_failed",
            body_snippet="synth_ai_py is not available",
        )

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    stream_done = False

    def _on_event(event: dict[str, Any]) -> bool:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception:
            return False
        return True

    def _run_stream() -> None:
        nonlocal stream_done
        try:
            _synth_ai_py.stream_sse(
                url,
                headers or {},
                method,
                json_payload,
                timeout,
                _on_event,
            )
        finally:
            stream_done = True
            with contextlib.suppress(Exception):
                loop.call_soon_threadsafe(queue.put_nowait, None)

    stream_future = loop.run_in_executor(None, _run_stream)

    # NOTE: No per-event timeout - SSE streams can have long gaps between events.
    # The overall stream is managed by the Rust core timeout, and the application
    # should implement its own high-level timeout logic if needed.
    while True:
        item = await queue.get()
        if item is None:
            await stream_future
            return
        yield item
