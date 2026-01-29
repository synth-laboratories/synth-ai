#!/usr/bin/env python3
"""Minimal offline MIPRO demo (Banking77).

Usage:
    uv run python demos/mipro/offline_demo.py
    uv run python demos/mipro/offline_demo.py --rollouts 5 --local-port 8016
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from demos.mipro.utils import (
    build_mipro_config,
    create_job,
    create_task_app_url,
    poll_job,
    resolve_backend_url,
    should_include_task_app_key,
    start_local_api,
)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIPRO offline demo")
    parser.add_argument(
        "--backend-url",
        default=None,
        help="Backend base URL (defaults to SYNTH_BACKEND_URL or SDK default)",
    )
    parser.add_argument("--local-host", default="localhost")
    parser.add_argument("--local-port", type=int, default=8016)
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    backend_url = (args.backend_url or resolve_backend_url()).rstrip("/")
    api_key = os.environ.get("SYNTH_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SYNTH_API_KEY is required")

    local_task_url, env_key, local_port = start_local_api(
        local_host=args.local_host,
        local_port=args.local_port,
        backend_url=backend_url,
    )
    task_app_url, tunnel = await create_task_app_url(
        backend_url=backend_url,
        local_host=args.local_host,
        local_port=local_port,
        env_key=env_key,
    )
    if tunnel:
        print("Waiting for tunnel propagation...")
        await asyncio.sleep(10.0)
    include_task_key = should_include_task_app_key(backend_url)

    seeds = list(range(args.rollouts))
    config_body = build_mipro_config(
        task_app_url=task_app_url,
        task_app_api_key=env_key if include_task_key else None,
        mode="offline",
        seeds=seeds,
    )

    try:
        job_id = create_job(backend_url, api_key, config_body)
        print(f"Offline job: {job_id}")
        detail = poll_job(backend_url, api_key, job_id, timeout=args.timeout)
        print(
            f"Offline status: {detail.get('status')} best_score={detail.get('best_score')}"
        )
    finally:
        if tunnel:
            print("\nClosing tunnel...")
            tunnel.close()


if __name__ == "__main__":
    asyncio.run(main())
