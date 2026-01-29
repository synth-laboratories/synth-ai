"""Helpers for probing Task App health endpoints."""

from __future__ import annotations

from typing import Any

import synth_ai_py


async def task_app_health(task_app_url: str) -> dict[str, Any]:
    """Probe a Task App base URL for basic reachability."""

    return await __import__("asyncio").to_thread(synth_ai_py.localapi_task_app_health, task_app_url)
