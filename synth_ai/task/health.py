"""Helpers for probing Task App health endpoints."""

from __future__ import annotations

from typing import Any

import aiohttp


async def task_app_health(task_app_url: str) -> dict[str, Any]:
    """Probe a Task App base URL for basic reachability.

    Behavior:
    - Try HEAD first (follows redirects)
    - Fallback to GET if HEAD is unsupported
    - Returns {ok: bool, status?: int, error?: str}
    """

    async def _try_request(session: aiohttp.ClientSession, method: str) -> dict[str, Any] | None:
        request = getattr(session, method)
        async with request(task_app_url, allow_redirects=True) as response:
            if 200 <= response.status < 400:
                return {"ok": True, "status": response.status}
        return None

    try:
        async with aiohttp.ClientSession() as session:
            for method in ("head", "get"):
                result = await _try_request(session, method)
                if result is not None:
                    return result
        return {"ok": False, "status": None}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
