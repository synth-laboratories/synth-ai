from __future__ import annotations

from collections.abc import Callable
import time

from synth_ai.core.rust_core.sse import stream_sse_events
from synth_ai.core.rust_core.urls import ensure_api_base


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
    api_base = ensure_api_base(base_url)
    candidates = [
        f"{api_base}/rl/jobs/{job_id}/events?since_seq=0",
        f"{api_base}/learning/jobs/{job_id}/events?since_seq=0",
    ]
    for url in candidates:
        try:
            start_t = time.time()
            async for obj in stream_sse_events(url, headers=headers, timeout=None):
                if on_event:
                    try:
                        on_event(obj)
                    except Exception:
                        pass
                if (time.time() - start_t) >= seconds:
                    return
        except Exception:
            continue
