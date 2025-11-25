from __future__ import annotations

from typing import Any

from synth_ai.core._utils.http import AsyncHttpClient


def _api_base(b: str) -> str:
    b = (b or "").rstrip("/")
    return b if b.endswith("/api") else f"{b}/api"


async def backend_health(base_url: str, api_key: str) -> dict[str, Any]:
    async with AsyncHttpClient(base_url, api_key, timeout=15.0) as http:
        js = await http.get(f"{_api_base(base_url)}/health")
    return {"ok": True, "raw": js}


async def task_app_health(task_app_url: str) -> dict[str, Any]:
    # Delegate to central task module for consistency
    from synth_ai.sdk.task.health import task_app_health as _th

    return await _th(task_app_url)


async def pricing_preflight(
    base_url: str,
    api_key: str,
    *,
    job_type: str,
    gpu_type: str,
    estimated_seconds: float,
    container_count: int,
) -> dict[str, Any]:
    body = {
        "job_type": job_type,
        "gpu_type": gpu_type,
        "estimated_seconds": float(estimated_seconds or 0.0),
        "container_count": int(container_count or 1),
    }
    async with AsyncHttpClient(base_url, api_key, timeout=30.0) as http:
        js = await http.post_json(f"{_api_base(base_url)}/v1/pricing/preflight", json=body)
    return js if isinstance(js, dict) else {"raw": js}


async def balance_autumn_normalized(base_url: str, api_key: str) -> dict[str, Any]:
    async with AsyncHttpClient(base_url, api_key, timeout=30.0) as http:
        js = await http.get(f"{_api_base(base_url)}/v1/balance/autumn-normalized")
    return js if isinstance(js, dict) else {"raw": js}
