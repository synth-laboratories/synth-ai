#!/usr/bin/env python3
"""Minimal online MIPRO demo (Banking77).

Usage:
    uv run python demos/mipro/online_demo.py
    uv run python demos/mipro/online_demo.py --rollouts 3
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
    extract_candidate_text,
    get_job_detail,
    get_system_state,
    new_rollout_id,
    push_status,
    resolve_backend_url,
    run_rollout,
    start_local_api,
)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIPRO online demo")
    parser.add_argument(
        "--backend-url",
        default=None,
        help="Backend base URL (defaults to SYNTH_BACKEND_URL or SDK default)",
    )
    parser.add_argument("--local-host", default="localhost")
    parser.add_argument("--local-port", type=int, default=8016)
    parser.add_argument("--rollouts", type=int, default=20)
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
    # Online MIPRO doesn't need tunneling - user drives rollouts locally
    task_app_url, tunnel = await create_task_app_url(
        backend_url=backend_url,
        local_host=args.local_host,
        local_port=local_port,
        env_key=env_key,
        mode="online",  # No tunneling needed for online mode
    )
    if tunnel:
        print("Waiting for tunnel propagation...")
        await asyncio.sleep(10.0)

    seeds = list(range(args.rollouts))
    # Online MIPRO doesn't need task_app_api_key - backend never calls task app
    config_body = build_mipro_config(
        task_app_url=task_app_url,
        task_app_api_key=None,  # Not needed for online mode
        mode="online",
        seeds=seeds,
    )

    job_id = create_job(backend_url, api_key, config_body)
    print(f"Online job: {job_id}")

    detail = get_job_detail(backend_url, api_key, job_id, include_metadata=True)
    metadata = detail.get("metadata", {})
    system_id = metadata.get("mipro_system_id")
    proxy_url = metadata.get("mipro_proxy_url")
    if not system_id or not proxy_url:
        raise RuntimeError(f"Missing mipro_system_id/proxy_url in metadata: {metadata}")
    print(f"System ID: {system_id}")
    print(f"Proxy URL: {proxy_url}")

    model = os.environ.get("BANKING77_MODEL", "gpt-4.1-nano")
    try:
        for seed in seeds:
            rollout_id = new_rollout_id(seed)
            inference_url = f"{proxy_url}/{rollout_id}/chat/completions"
            reward, used_candidate_id = run_rollout(
                task_app_url=task_app_url,
                env_key=env_key,
                seed=seed,
                inference_url=inference_url,
                model=model,
                rollout_id=rollout_id,
            )
            candidate_label = used_candidate_id or "n/a"
            print(
                f"Rollout {seed}: reward={reward:.3f} id={rollout_id} candidate={candidate_label}"
            )
            push_status(
                backend_url=backend_url,
                api_key=api_key,
                system_id=system_id,
                rollout_id=rollout_id,
                reward=reward,
                candidate_id=used_candidate_id,
            )

        state = get_system_state(backend_url, api_key, system_id)
        best_candidate_id = state.get("best_candidate_id")
        candidate_text = extract_candidate_text(state, best_candidate_id)
        print(
            "Online state: "
            f"best_candidate_id={best_candidate_id} "
            f"version={state.get('version')} "
            f"candidates={len(state.get('candidates', {}))}"
        )
        if candidate_text:
            preview = candidate_text[:800] + ("..." if len(candidate_text) > 800 else "")
            print("\nBest candidate text:\n" + preview)
        else:
            print("\nBest candidate text: <not available>")
    finally:
        if tunnel:
            print("\nClosing tunnel...")
            tunnel.close()


if __name__ == "__main__":
    asyncio.run(main())
