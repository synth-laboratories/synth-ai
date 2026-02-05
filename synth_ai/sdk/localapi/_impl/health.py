"""Helpers for probing Task App health endpoints."""

from __future__ import annotations

from typing import Any

try:
    import synth_ai_py
except Exception:  # pragma: no cover
    synth_ai_py = None


async def task_app_health(task_app_url: str) -> dict[str, Any]:
    """Probe a Task App base URL for basic reachability."""

    if synth_ai_py is not None and hasattr(synth_ai_py, "localapi_task_app_health"):
        return await __import__("asyncio").to_thread(
            synth_ai_py.localapi_task_app_health, task_app_url
        )

    try:
        import httpx

        url = task_app_url.rstrip("/")
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(f"{url}/health")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return {"healthy": False, "error": str(exc)}
