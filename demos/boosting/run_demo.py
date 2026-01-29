#!/usr/bin/env python3
"""Run GEPA prompt optimization for LLM-based tabular featurization + XGBoost scoring.

Usage:
  uv run python demos/boosting/run_demo.py
  uv run python demos/boosting/run_demo.py --local
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from copy import deepcopy
from typing import Any

import httpx

from synth_ai.core.utils.env import mint_demo_api_key
from synth_ai.core.tunnels import (
    PortConflictBehavior,
    TunnelBackend,
    TunneledLocalAPI,
    acquire_port,
    cleanup_all,
)
from synth_ai.sdk.localapi._impl.server import run_server_background
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.optimization.internal.prompt_learning import PromptLearningJob
from synth_ai.sdk.optimization.internal.learning.prompt_learning_client import PromptLearningClient

from localapi_boosting import DEFAULT_SYSTEM_PROMPT, create_boosting_local_api, score_prompt


parser = argparse.ArgumentParser(description="Run GEPA boosting demo")
parser.add_argument(
    "--local",
    action="store_true",
    help="Run in local mode (localhost backend, no tunnels)",
)
parser.add_argument(
    "--local-host",
    type=str,
    default="localhost",
    help="Hostname for local API URLs",
)
args = parser.parse_args()

LOCAL_MODE = args.local
LOCAL_HOST = args.local_host


def wait_for_health_check_sync(host: str, port: int, api_key: str, timeout: float = 30.0) -> None:
    health_url = f"http://{host}:{port}/health"
    headers = {"X-API-Key": api_key} if api_key else {}
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get(health_url, headers=headers, timeout=5.0)
            if response.status_code in (200, 400):
                return
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Health check failed: {health_url} not ready after {timeout}s")


if LOCAL_MODE:
    SYNTH_API_BASE = "http://localhost:8000"
    TUNNEL_BACKEND = TunnelBackend.Localhost
    LOCAL_API_PORT = 8121
else:
    SYNTH_API_BASE = os.environ.get("SYNTH_BACKEND_URL", "https://api-dev.usesynth.ai")
    TUNNEL_BACKEND = TunnelBackend.CloudflareManagedLease
    LOCAL_API_PORT = 8017

print(f"Backend: {SYNTH_API_BASE}")
print(f"Tunnel backend: {TUNNEL_BACKEND.value}")

r = httpx.get(f"{SYNTH_API_BASE}/health", timeout=30)
if r.status_code != 200:
    raise RuntimeError(f"Backend not healthy: status {r.status_code}")

API_KEY = os.environ.get("SYNTH_API_KEY", "")
if not API_KEY:
    print("No SYNTH_API_KEY found, minting demo key...")
    API_KEY = mint_demo_api_key(backend_url=SYNTH_API_BASE)
else:
    print(f"Using SYNTH_API_KEY: {API_KEY[:20]}...")

os.environ["SYNTH_API_KEY"] = API_KEY

ENVIRONMENT_API_KEY = ensure_localapi_auth(
    backend_base=SYNTH_API_BASE,
    synth_api_key=API_KEY,
)
print(f"Env key ready: {ENVIRONMENT_API_KEY[:12]}...{ENVIRONMENT_API_KEY[-4:]}")


async def main() -> None:
    print("Starting local task app...", flush=True)
    app = create_boosting_local_api(system_prompt=DEFAULT_SYSTEM_PROMPT)
    port = acquire_port(LOCAL_API_PORT, on_conflict=PortConflictBehavior.FIND_NEW)
    print(f"Launching local server on port {port}...", flush=True)
    server = run_server_background(app, host="0.0.0.0", port=port)
    print("Waiting for local health check...", flush=True)
    wait_for_health_check_sync("localhost", port, ENVIRONMENT_API_KEY)
    print("Local task app healthy.", flush=True)

    if LOCAL_MODE:
        local_api_url = f"http://{LOCAL_HOST}:{port}"
        tunnel = None
    else:
        print("Creating tunnel...", flush=True)
        tunnel = await TunneledLocalAPI.create(
            local_port=port,
            backend=TUNNEL_BACKEND,
            backend_url=SYNTH_API_BASE,
            api_key=API_KEY,
            env_api_key=ENVIRONMENT_API_KEY,
        )
        local_api_url = tunnel.url
        print("Tunnel created.", flush=True)

    print(f"Local API URL: {local_api_url}")

    config_body = {
        "prompt_learning": {
            "task_app_url": local_api_url,
            "algorithm": "gepa",
            "initial_prompt": {
                "id": "boosting_featurizer",
                "name": "Boosting Featurizer",
                "messages": [
                    {"role": "system", "order": 0, "pattern": DEFAULT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "order": 1,
                        "pattern": "Row:\n{row_text}\n\nReturn a tool call with engineered features.",
                    },
                ],
                "wildcards": {"row_text": "REQUIRED"},
            },
            "policy": {
                "model": "gpt-4.1-nano",
                "provider": "openai",
                "inference_mode": "synth_hosted",
                "temperature": 0.0,
                "max_completion_tokens": 256,
            },
            "env_config": {"split": "train"},
            "gepa": {
                "env_name": "boosting",
                "evaluation": {"seeds": list(range(15)), "validation_seeds": list(range(20, 35))},
                "rollout": {"budget": 60, "max_concurrent": 4, "minibatch_size": 2},
                "mutation": {"rate": 0.4},
                "population": {"initial_size": 3, "num_generations": 4, "children_per_generation": 2},
                "archive": {"pareto_set_size": 10},
                "token": {"counting_model": "gpt-4"},
            },
        }
    }

    print("Submitting GEPA job...", flush=True)
    pl_job = PromptLearningJob.from_dict(
        config_dict=deepcopy(config_body),
        backend_url=SYNTH_API_BASE,
    )

    job_id = pl_job.submit()
    print(f"GEPA job ID: {job_id}")

    def on_status_update(status: dict[str, Any]) -> None:
        status_name = status.get("status", "unknown")
        best = status.get("best_score") or status.get("best_reward")
        if isinstance(best, (int, float)):
            print(f"[status] {status_name} | best={best:.4f}")
        else:
            print(f"[status] {status_name}")

    gepa_result = pl_job.poll_until_complete(timeout=3600.0, interval=15.0, on_status=on_status_update)

    if gepa_result.succeeded:
        print("GEPA completed successfully.")
        if isinstance(gepa_result.best_score, (float, int)):
            print(f"Best score: {gepa_result.best_score:.4f}")
    else:
        print(f"GEPA failed: {gepa_result.error}")

    optimized_prompt = DEFAULT_SYSTEM_PROMPT
    try:
        pl_client = PromptLearningClient(SYNTH_API_BASE, API_KEY)
        prompt_results = await pl_client.get_prompts(gepa_result.job_id)
        if prompt_results.top_prompts:
            top = prompt_results.top_prompts[0]
            if isinstance(top, dict):
                for key in ("full_text", "prompt", "template", "content", "system_prompt"):
                    if key in top and isinstance(top[key], str) and top[key].strip():
                        optimized_prompt = top[key].strip()
                        break
        if prompt_results.best_prompt and isinstance(prompt_results.best_prompt, dict):
            for msg in prompt_results.best_prompt.get("messages", []):
                if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                    optimized_prompt = msg["content"].strip()
                    break
    except Exception as exc:
        print(f"Warning: could not fetch optimized prompt ({exc}). Using baseline prompt.")

    print("Optimized system prompt preview:")
    print(optimized_prompt[:400])

    inference_url = os.getenv("BOOSTING_INFERENCE_URL")
    if inference_url:
        print("Running quick baseline vs optimized comparison...")
        baseline = await score_prompt(
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            inference_url=inference_url,
            model="gpt-4.1-nano",
            api_key=API_KEY,
            policy_config={},
        )
        optimized = await score_prompt(
            system_prompt=optimized_prompt,
            inference_url=inference_url,
            model="gpt-4.1-nano",
            api_key=API_KEY,
            policy_config={},
        )
        print(f"Baseline reward: {baseline['reward']:.4f} (AUC={baseline['auc']:.4f})")
        print(f"Optimized reward: {optimized['reward']:.4f} (AUC={optimized['auc']:.4f})")
    else:
        print("Set BOOSTING_INFERENCE_URL to run baseline vs optimized scoring.")

    if tunnel:
        await tunnel.close()
    server.should_exit = True
    cleanup_all()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        cleanup_all()
