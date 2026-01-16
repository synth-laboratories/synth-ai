from typing import Any

from synth_ai.core.http import AsyncHttpClient
from synth_ai.core.urls import (
    synth_api_v1_base,
    synth_balance_autumn_normalized_url,
    synth_health_url,
    synth_pricing_preflight_url,
)


async def backend_health(synth_user_key: str, synth_base_url: str | None = None) -> dict[str, Any]:
    async with AsyncHttpClient(
        synth_api_v1_base(synth_base_url), synth_user_key, timeout=15.0
    ) as http:
        js = await http.get(synth_health_url(synth_base_url))
    return {"ok": True, "raw": js}


async def task_app_health(localapi_url: str) -> dict[str, Any]:
    # Delegate to central task module for consistency
    from synth_ai.sdk.task.health import task_app_health as _th

    return await _th(localapi_url)


async def pricing_preflight(
    synth_user_key: str,
    *,
    job_type: str,
    gpu_type: str,
    estimated_seconds: float,
    container_count: int,
    synth_base_url: str | None = None,
) -> dict[str, Any]:
    body = {
        "job_type": job_type,
        "gpu_type": gpu_type,
        "estimated_seconds": float(estimated_seconds or 0.0),
        "container_count": int(container_count or 1),
    }
    async with AsyncHttpClient(
        synth_api_v1_base(synth_base_url), synth_user_key, timeout=30.0
    ) as http:
        js = await http.post_json(synth_pricing_preflight_url(synth_base_url), json=body)
    return js if isinstance(js, dict) else {"raw": js}


async def balance_autumn_normalized(
    synth_user_key: str, synth_base_url: str | None = None
) -> dict[str, Any]:
    async with AsyncHttpClient(
        synth_api_v1_base(synth_base_url), synth_user_key, timeout=30.0
    ) as http:
        js = await http.get(synth_balance_autumn_normalized_url(synth_base_url))
    return js if isinstance(js, dict) else {"raw": js}
