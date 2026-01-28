"""Helpers for probing Task App health endpoints."""

from __future__ import annotations

import asyncio
import urllib.error
import urllib.request
from typing import Any


async def task_app_health(task_app_url: str) -> dict[str, Any]:
    """Probe a Task App base URL for basic reachability.

    Behavior:
    - Try HEAD first (follows redirects)
    - Fallback to GET if HEAD is unsupported
    - Returns {ok: bool, status?: int, error?: str}
    """

    def _sync_request(method: str) -> dict[str, Any] | None:
        req = urllib.request.Request(task_app_url, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                status = response.getcode()
                if 200 <= status < 400:
                    return {"ok": True, "status": status}
        except Exception:
            return None
        return None

    try:
        for method in ("head", "get"):
            result = await asyncio.to_thread(_sync_request, method)
            if result is not None:
                return result
        return {"ok": False, "status": None}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
