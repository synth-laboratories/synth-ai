from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import suppress

import aiohttp


def _api_base(b: str) -> str:
    b = (b or "").rstrip("/")
    return b if b.endswith("/api") else f"{b}/api"


async def stream_events(
    base_url: str,
    api_key: str,
    job_id: str,
    *,
    seconds: int = 60,
    on_event: Callable[[dict], None] | None = None,
) -> None:
    if seconds <= 0:
        return
    headers = {"Accept": "text/event-stream", "Authorization": f"Bearer {api_key}"}
    candidates = [
        f"{_api_base(base_url)}/rl/jobs/{job_id}/events?since_seq=0",
        f"{_api_base(base_url)}/learning/jobs/{job_id}/events?since_seq=0",
    ]
    for url in candidates:
        try:
            async with (
                aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session,
                session.get(url, headers=headers) as resp,
            ):
                if resp.status != 200:
                    continue
                start_t = time.time()
                async for raw in resp.content:
                    line = raw.decode(errors="ignore").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    if on_event:
                        with suppress(Exception):
                            on_event(obj)
                    if (time.time() - start_t) >= seconds:
                        return
        except Exception:
            continue
