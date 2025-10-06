from __future__ import annotations

"""
examples/rl/check.py

Quick diagnostics for RL setup:
- Backend health
- Task App health (HEAD/GET)
"""

import argparse
import asyncio
import os
from typing import Optional

from synth_ai.config.base_url import get_backend_from_env
from synth_ai.learning import RlClient, backend_health
import httpx

try:  # Allow running as script without package install
    from examples.common.backend import resolve_backend_url  # type: ignore
except Exception:  # pragma: no cover - fallback for direct execution

    def resolve_backend_url() -> str:
        base, _ = get_backend_from_env()
        base = base.rstrip("/")
        return base if base.endswith("/api") else f"{base}/api"


def _api_base(b: str) -> str:
    b = (b or "").rstrip("/")
    return b if b.endswith("/api") else f"{b}/api"


async def _backend_health(base_url: str, api_key: str) -> bool:
    try:
        res = await backend_health(base_url, api_key)
        ok = bool(res.get("ok", True))
        print("backend.health:", "OK" if ok else "WARN")
        return ok
    except Exception as e:
        print(f"backend.health: WARN ({type(e).__name__}: {e})")
        return False


async def _task_app_health(task_app_url: str) -> bool:
    """Probe the task app /health endpoint directly.

    Considers both {"ok": true} and {"healthy": true} as success, and treats
    HTTP 200 as OK even if the body schema differs.
    Optionally sends X-API-Key if ENVIRONMENT_API_KEY is set.
    """
    try:
        base = (task_app_url or "").rstrip("/")
        url = f"{base}/health"
        headers = {}
        env_key = os.getenv("ENVIRONMENT_API_KEY", "").strip()
        if env_key:
            headers["X-API-Key"] = env_key
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, headers=headers)
        ok = (r.status_code == 200)
        status = None
        try:
            data = r.json()
            status = data.get("status") or ("healthy" if data.get("healthy") else None)
            if not ok:
                ok = bool(data.get("ok") or data.get("healthy"))
        except Exception:
            pass
        print("task_app.health:", "OK" if ok else "WARN", status)
        return ok
    except Exception as e:
        print(f"task_app.health: WARN ({type(e).__name__}: {e})")
        return False


# Trainer check removed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    base_default = resolve_backend_url()
    _, key_default = get_backend_from_env()
    p.add_argument("--backend-url", type=str, default=base_default)
    p.add_argument(
        "--api-key",
        type=str,
        default=(os.getenv("SYNTH_API_KEY", "").strip() or key_default),
    )
    p.add_argument("--task-app-url", type=str, default=os.getenv("TASK_APP_BASE_URL", "").strip())
    return p.parse_args()


async def _main() -> int:
    args = _parse_args()
    ok = True
    ok &= await _backend_health(args.backend_url, args.api_key)
    if args.task_app_url:
        ok &= await _task_app_health(args.task_app_url)
    print("result:", "OK" if ok else "WARN")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
