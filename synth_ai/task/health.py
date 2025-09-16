from __future__ import annotations

from typing import Any, Dict
import aiohttp


async def task_app_health(task_app_url: str) -> Dict[str, Any]:
    """Probe a Task App base URL for basic reachability.

    Behavior:
    - Try HEAD first (follows redirects)
    - Fallback to GET if HEAD is unsupported
    - Returns {ok: bool, status?: int, error?: str}
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(task_app_url, allow_redirects=True) as r:
                if 200 <= r.status < 400:
                    return {"ok": True, "status": r.status}
        async with aiohttp.ClientSession() as session:
            async with session.get(task_app_url, allow_redirects=True) as r2:
                if 200 <= r2.status < 400:
                    return {"ok": True, "status": r2.status}
        return {"ok": False, "status": None}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


