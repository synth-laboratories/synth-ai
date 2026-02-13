"""Helpers for probing Container health endpoints."""

from __future__ import annotations

from typing import Any

try:
    import synth_ai_py
except Exception:  # pragma: no cover
    synth_ai_py = None


async def container_health(container_url: str) -> dict[str, Any]:
    """Probe a Container base URL for basic reachability."""

    if synth_ai_py is not None and hasattr(synth_ai_py, "container_health_check"):
        return await __import__("asyncio").to_thread(
            synth_ai_py.container_health_check, container_url
        )

    try:
        import httpx

        url = container_url.rstrip("/")
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(f"{url}/health")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return {"healthy": False, "error": str(exc)}
