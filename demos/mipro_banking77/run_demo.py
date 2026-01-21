#!/usr/bin/env python3
"""Run the Banking77 MIPRO demo end-to-end (offline + online).

Usage:
    uv run python demos/mipro_banking77/run_demo.py --mode both
    uv run python demos/mipro_banking77/run_demo.py --mode offline
    uv run python demos/mipro_banking77/run_demo.py --mode online
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from demos.gepa_banking77 import localapi_banking77
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.localapi._impl import run_server_background
from synth_ai.core.tunnels import PortConflictBehavior, acquire_port


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

    raise RuntimeError(
        f"Health check failed: {health_url} not ready after {timeout}s. "
        "Make sure your task app has a /health endpoint."
    )


def build_initial_prompt() -> Dict[str, Any]:
    user_prompt = (
        "Customer Query: {query}\n\n"
        "Available Intents:\n{available_intents}\n\n"
        "Classify this query into one of the above banking intents using the tool call."
    )
    return {
        "id": "banking77_pattern",
        "name": "Banking77 Classification",
        "messages": [
            {
                "role": "system",
                "order": 0,
                "pattern": localapi_banking77.SYSTEM_PROMPT,
            },
            {"role": "user", "order": 1, "pattern": user_prompt},
        ],
        "wildcards": {"query": "REQUIRED", "available_intents": "OPTIONAL"},
    }


def build_mipro_config(
    *,
    task_app_url: str,
    task_app_api_key: str,
    mode: str,
    seeds: Iterable[int],
) -> Dict[str, Any]:
    seed_list = list(seeds)
    policy_model = os.environ.get("BANKING77_POLICY_MODEL", "gpt-4.1-nano")
    policy_provider = os.environ.get("BANKING77_POLICY_PROVIDER", "openai")
    proposer_model = os.environ.get("BANKING77_PROPOSER_MODEL", policy_model)
    proposer_provider = os.environ.get("BANKING77_PROPOSER_PROVIDER", policy_provider)
    proposer_url = os.environ.get(
        "BANKING77_PROPOSER_URL", "https://api.openai.com/v1/responses"
    )
    return {
        "prompt_learning": {
            "algorithm": "mipro",
            "task_app_id": "banking77",
            "task_app_url": task_app_url,
            "task_app_api_key": task_app_api_key,
            "initial_prompt": build_initial_prompt(),
            "policy": {
                "model": policy_model,
                "provider": policy_provider,
                "inference_mode": "synth_hosted",
                "temperature": 0.0,
                "max_completion_tokens": 256,
            },
            "mipro": {
                "mode": mode,
                "bootstrap_train_seeds": seed_list,
                "online_pool": seed_list,
                "online_proposer_mode": "inline",
                "online_proposer_min_rollouts": 20,
                "proposer": {
                    "mode": "instruction_only",
                    "model": proposer_model,
                    "provider": proposer_provider,
                    "inference_url": proposer_url,
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "generate_at_iterations": [0],
                    "instructions_per_batch": 1,
                },
            },
        },
    }


def create_job(backend_url: str, api_key: str, config_body: Dict[str, Any]) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = httpx.post(
        f"{backend_url}/api/prompt-learning/online/jobs",
        json={"algorithm": "mipro", "config_body": config_body},
        headers=headers,
        timeout=60.0,
    )
    if response.status_code != 200:
        print(f"Error response: {response.status_code}")
        print(f"Response body: {response.text}")
    response.raise_for_status()
    payload = response.json()
    job_id = payload.get("job_id")
    if not job_id:
        raise RuntimeError(f"Missing job_id in response: {payload}")
    return str(job_id)


def get_job_detail(
    backend_url: str, api_key: str, job_id: str, *, include_metadata: bool = True
) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = httpx.get(
        f"{backend_url}/api/prompt-learning/online/jobs/{job_id}",
        params={
            "include_events": False,
            "include_snapshot": False,
            "include_metadata": include_metadata,
        },
        headers=headers,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def poll_job(
    backend_url: str,
    api_key: str,
    job_id: str,
    *,
    timeout: float = 600.0,
    interval: float = 2.0,
) -> Dict[str, Any]:
    start = time.time()
    while time.time() - start < timeout:
        detail = get_job_detail(backend_url, api_key, job_id, include_metadata=True)
        status = detail.get("status", "")
        if status in {"succeeded", "failed"}:
            return detail
        time.sleep(interval)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def build_stage_payloads(stage_id: str, instruction_text: str) -> Dict[str, Any]:
    return {
        stage_id: {
            "stage_id": stage_id,
            "module_id": stage_id,
            "instruction_text": instruction_text,
            "instruction_indices": [1],
            "instruction_lines": instruction_text.splitlines(),
            "demo_indices": [],
            "baseline_messages": [],
        }
    }


def explain_proposer_only() -> None:
    print("Using proposer pipeline only; skipping manual candidate registration.")


def run_rollout(
    task_app_url: str,
    env_key: str,
    *,
    seed: int,
    inference_url: str,
    model: str,
    rollout_id: str,
) -> tuple[float, str]:
    import time
    rollout_start = time.perf_counter()
    payload = {
        "trace_correlation_id": rollout_id,
        "env": {"seed": seed, "config": {"seed": seed, "split": "train"}},
        "policy": {"config": {"model": model, "inference_url": inference_url}},
    }
    headers = {"X-API-Key": env_key}
    request_start = time.perf_counter()
    response = httpx.post(
        f"{task_app_url}/rollout",
        json=payload,
        headers=headers,
        timeout=120.0,
    )
    request_duration = (time.perf_counter() - request_start) * 1000.0
    response.raise_for_status()
    body = response.json()
    if os.getenv("BANKING77_DEBUG_RESPONSE") == "1":
        print(f"[DEBUG] rollout_response_body: {body}")
    reward_info = body.get("reward_info", {}) if isinstance(body, dict) else {}
    reward = reward_info.get("outcome_reward")
    if reward is None and isinstance(body, dict):
        metrics = body.get("metrics", {}) or {}
        reward = metrics.get("outcome_reward")
        if reward is None:
            reward = (metrics.get("outcome_objectives") or {}).get("reward", 0.0)
    if reward is None:
        reward = (reward_info.get("outcome_objectives") or {}).get("reward", 0.0)
    
    # Extract candidate_id from response metadata (set by task app from proxy headers)
    metadata = body.get("metadata", {}) if isinstance(body, dict) else {}
    candidate_id = metadata.get("mipro_candidate_id")
    if not candidate_id:
        # Fallback to header if metadata not available
        candidate_id = response.headers.get("x-mipro-candidate-id", "")
    if not candidate_id:
        raise RuntimeError(f"Missing mipro_candidate_id in rollout response")
    candidate_id = str(candidate_id)
    
    rollout_duration = (time.perf_counter() - rollout_start) * 1000.0
    print(f"[TIMING] Rollout {rollout_id}: total={rollout_duration:.2}ms (task_app_request={request_duration:.2}ms)")
    
    return float(reward or 0.0), candidate_id


def push_status(
    backend_url: str,
    api_key: str,
    system_id: str,
    rollout_id: str,
    reward: float,
    candidate_id: str,
) -> None:
    import time
    headers = {"Authorization": f"Bearer {api_key}"}
    status_start = time.perf_counter()
    response = httpx.post(
        f"{backend_url}/api/prompt-learning/online/mipro/systems/{system_id}/status",
        json={
            "rollout_id": rollout_id,
            "status": "reward",
            "reward": reward,
            "candidate_id": candidate_id,
        },
        headers=headers,
        timeout=30.0,
    )
    status_duration = (time.perf_counter() - status_start) * 1000.0
    print(f"[TIMING] Status update (reward): {status_duration:.2}ms")
    if response.status_code != 200:
        print(f"Status push error: {response.status_code}")
        print(f"Response body: {response.text}")
    response.raise_for_status()

    done_start = time.perf_counter()
    response = httpx.post(
        f"{backend_url}/api/prompt-learning/online/mipro/systems/{system_id}/status",
        json={"rollout_id": rollout_id, "status": "done", "candidate_id": candidate_id},
        headers=headers,
        timeout=30.0,
    )
    done_duration = (time.perf_counter() - done_start) * 1000.0
    print(f"[TIMING] Status update (done): {done_duration:.2}ms")
    response.raise_for_status()


def get_system_state(backend_url: str, api_key: str, system_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = httpx.get(
        f"{backend_url}/api/prompt-learning/online/mipro/systems/{system_id}/state",
        headers=headers,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Banking77 MIPRO demo (offline/online)")
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("RUST_BACKEND_URL", "http://localhost:8090"),
        help="Rust backend base URL",
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "online", "both"],
        default="both",
        help="Which demo mode to run",
    )
    parser.add_argument("--local-host", default="localhost")
    parser.add_argument("--local-port", type=int, default=8016)
    parser.add_argument("--rollouts", type=int, default=3)
    args = parser.parse_args()

    backend_url = args.backend_url.rstrip("/")
    api_key = os.environ.get("SYNTH_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SYNTH_API_KEY is required")

    env_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
    if not env_key:
        env_key = ensure_localapi_auth(backend_base=None, upload=False, persist=False)
    os.environ["ENVIRONMENT_API_KEY"] = env_key

    port = acquire_port(args.local_port, on_conflict=PortConflictBehavior.FIND_NEW)
    if port != args.local_port:
        print(f"Port {args.local_port} in use, using port {port} instead")

    run_server_background(localapi_banking77.app, port)
    print(f"Waiting for local API on port {port}...")
    wait_for_health_check_sync("localhost", port, env_key, timeout=30.0)
    task_app_url = f"http://{args.local_host}:{port}"
    print(f"Local API URL: {task_app_url}")

    seeds = list(range(args.rollouts))

    if args.mode in {"offline", "both"}:
        print("\n=== Offline MIPRO ===")
        offline_config = build_mipro_config(
            task_app_url=task_app_url,
            task_app_api_key=env_key,
            mode="offline",
            seeds=seeds,
        )
        offline_job_id = create_job(backend_url, api_key, offline_config)
        print(f"Offline job: {offline_job_id}")
        offline_detail = poll_job(backend_url, api_key, offline_job_id, timeout=1800.0)
        print(
            f"Offline status: {offline_detail.get('status')} best_score={offline_detail.get('best_score')}"
        )

    if args.mode in {"online", "both"}:
        print("\n=== Online MIPRO ===")
        online_config = build_mipro_config(
            task_app_url=task_app_url,
            task_app_api_key=env_key,
            mode="online",
            seeds=seeds,
        )
        online_job_id = create_job(backend_url, api_key, online_config)
        print(f"Online job: {online_job_id}")
        online_detail = poll_job(backend_url, api_key, online_job_id, timeout=300.0)
        metadata = online_detail.get("metadata", {})
        system_id = metadata.get("mipro_system_id")
        proxy_url = metadata.get("mipro_proxy_url")
        if not system_id or not proxy_url:
            raise RuntimeError(f"Missing mipro_system_id/proxy_url in metadata: {metadata}")
        print(f"System ID: {system_id}")
        print(f"Proxy URL: {proxy_url}")

        explain_proposer_only()

        model = os.environ.get("BANKING77_MODEL", "gpt-4.1-nano")
        for seed in seeds:
            rollout_id = f"trace_rollout_{seed}_{uuid.uuid4().hex[:6]}"
            inference_url = f"{proxy_url}/{rollout_id}/chat/completions"
            reward, used_candidate_id = run_rollout(
                task_app_url,
                env_key,
                seed=seed,
                inference_url=inference_url,
                model=model,
                rollout_id=rollout_id,
            )
            print(f"Rollout {seed}: reward={reward:.3f} id={rollout_id} candidate={used_candidate_id}")
            push_status(backend_url, api_key, system_id, rollout_id, reward, used_candidate_id)

        state = get_system_state(backend_url, api_key, system_id)
        print(
            "Online state: "
            f"best_candidate_id={state.get('best_candidate_id')} "
            f"version={state.get('version')} "
            f"candidates={len(state.get('candidates', {}))}"
        )


if __name__ == "__main__":
    main()
