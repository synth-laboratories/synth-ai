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

from synth_ai.learning import RlClient, backend_health, task_app_health
def _resolve_backend_url() -> str:
    override = os.getenv("BACKEND_OVERRIDE", "").strip()
    if override:
        base = override
    else:
        raw = os.getenv("PROD_BACKEND_URL", "").strip()
        base = raw or "https://agent-learning.onrender.com/api"
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
    try:
        res = await task_app_health(task_app_url)
        ok = bool(res.get("ok", False))
        print("task_app.health:", "OK" if ok else "WARN", res.get("status"))
        return ok
    except Exception as e:
        print(f"task_app.health: WARN ({type(e).__name__}: {e})")
        return False


# Trainer check removed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backend-url", type=str, default=_resolve_backend_url())
    p.add_argument("--api-key", type=str, default=os.getenv("SYNTH_API_KEY", "").strip())
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


