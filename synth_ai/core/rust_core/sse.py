from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for rust_core.sse.") from exc


def _require_rust() -> Any:
    if synth_ai_py is None:
        raise RuntimeError("synth_ai_py is required for SSE streaming.")
    return synth_ai_py


async def stream_sse_events(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    json_payload: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> AsyncIterator[dict[str, Any]]:
    rust = _require_rust()
    iterator = rust.stream_sse_events(
        url,
        headers=headers,
        method=method,
        json_payload=json_payload,
        timeout=timeout,
    )
    while True:
        try:
            item = await asyncio.to_thread(next, iterator)
        except StopIteration:
            return
        yield item
