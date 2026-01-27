from __future__ import annotations

import json
from typing import Any, AsyncIterator

from synth_ai.core.errors import HTTPError
from synth_ai.core.rust_core.http import get_shared_http_client


async def stream_sse_events(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    json_payload: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> AsyncIterator[dict[str, Any]]:
    client = get_shared_http_client()
    async with client.stream(
        method,
        url,
        headers=headers,
        json=json_payload,
        timeout=timeout,
    ) as resp:
        if resp.status_code >= 400:
            text = await resp.aread()
            body = text.decode(errors="ignore") if text else ""
            raise HTTPError(
                status=resp.status_code,
                url=str(resp.url),
                message="sse_stream_failed",
                body_snippet=body[:200] if body else None,
                detail=None,
            )

        async for line in resp.aiter_lines():
            decoded = line.strip()
            if not decoded or decoded.startswith(":"):
                continue
            if not decoded.startswith("data:"):
                continue
            data = decoded[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                return
            try:
                event = json.loads(data)
            except Exception:
                continue
            if isinstance(event, dict):
                yield event
