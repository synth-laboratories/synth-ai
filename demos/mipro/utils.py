#!/usr/bin/env python3
"""Shared helpers for MIPRO demos (online + offline)."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any, Dict, Iterable

import httpx

from demos.gepa_banking77 import localapi_banking77
from synth_ai.core.utils.urls import BACKEND_URL_BASE
from synth_ai.core.tunnels import PortConflictBehavior, acquire_port
from synth_ai.core.tunnels.tunneled_api import TunneledLocalAPI, TunnelBackend
from synth_ai.sdk.localapi._impl import run_server_background
from synth_ai.sdk.localapi.auth import ensure_localapi_auth


def wait_for_health_check_sync(
    host: str, port: int, api_key: str, timeout: float = 30.0
) -> None:
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


def resolve_backend_url() -> str:
    env_url = (os.environ.get("SYNTH_BACKEND_URL") or "").strip()
    if env_url:
        return env_url.rstrip("/")

    candidates = [
        "http://localhost:8000",
        "http://localhost:8001",
    ]
    for candidate in candidates:
        try:
            response = httpx.get(f"{candidate}/health", timeout=3.0)
            if response.status_code == 200:
                return candidate.rstrip("/")
        except (httpx.RequestError, httpx.TimeoutException):
            continue

    return BACKEND_URL_BASE.rstrip("/")


def is_local_backend(backend_url: str | None) -> bool:
    if not backend_url:
        return False
    lowered = backend_url.lower()
    return "localhost" in lowered or "127.0.0.1" in lowered


def should_tunnel_task_app(backend_url: str | None, *, mode: str = "offline") -> bool:
    """Determine if task app needs tunneling.

    Online MIPRO never needs tunneling - the backend doesn't call the task app.
    Offline MIPRO/GEPA need tunneling for remote backends.
    """
    # Online mode never needs tunneling - user drives rollouts locally
    if mode == "online":
        return False

    override = (os.environ.get("MIPRO_TUNNEL_TASK_APP") or "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    return not is_local_backend(backend_url)


async def create_task_app_url(
    *,
    backend_url: str | None,
    local_host: str,
    local_port: int,
    env_key: str,
    mode: str = "offline",
) -> tuple[str, TunneledLocalAPI | None]:
    """Create task app URL, tunneling if necessary.

    Online MIPRO never needs tunneling - the backend doesn't call the task app.
    Offline MIPRO/GEPA need tunneling for remote backends.
    """
    override = (os.environ.get("MIPRO_TASK_APP_URL") or "").strip()
    if override:
        return override.rstrip("/"), None

    if should_tunnel_task_app(backend_url, mode=mode):
        backend = (os.environ.get("MIPRO_TUNNEL_BACKEND") or "").strip().lower()
        if not backend:
            backend = "quick" if backend_url and "api-dev" in backend_url else "managed"
        if backend in {"managed", "cloudflare_managed"}:
            tunnel_backend = TunnelBackend.CloudflareManagedTunnel
        else:
            tunnel_backend = TunnelBackend.CloudflareQuickTunnel

        tunnel_backend_url = None
        if tunnel_backend == TunnelBackend.CloudflareManagedTunnel:
            tunnel_backend_url = (
                os.environ.get("MIPRO_TUNNEL_BACKEND_URL") or BACKEND_URL_BASE
            )

        tunnel = await TunneledLocalAPI.create(
            local_port=local_port,
            backend=tunnel_backend,
            backend_url=tunnel_backend_url,
            env_api_key=env_key,
            progress=True,
        )
        return tunnel.url.rstrip("/"), tunnel

    return f"http://{local_host}:{local_port}", None


def start_local_api(
    *,
    local_host: str,
    local_port: int,
    backend_url: str | None,
) -> tuple[str, str, int]:
    env_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
    if not env_key:
        upload = should_upload_env_key(backend_url)
        env_key = ensure_localapi_auth(
            backend_base=backend_url if upload else None,
            upload=upload,
            persist=False,
        )
    os.environ["ENVIRONMENT_API_KEY"] = env_key

    port = acquire_port(local_port, on_conflict=PortConflictBehavior.FIND_NEW)
    if port != local_port:
        print(f"Port {local_port} in use, using port {port} instead")

    run_server_background(localapi_banking77.app, port)
    print(f"Waiting for local API on port {port}...")
    wait_for_health_check_sync("localhost", port, env_key, timeout=30.0)
    task_app_url = f"http://{local_host}:{port}"
    print(f"Local API URL: {task_app_url}")
    return task_app_url, env_key, port


def should_include_task_app_key(backend_url: str | None) -> bool:
    override = os.environ.get("MIPRO_INCLUDE_TASK_APP_KEY")
    if override and override.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if not backend_url:
        return False
    lowered = backend_url.lower()
    return any(
        host in lowered
        for host in ("localhost:8090", "localhost:8097", "127.0.0.1:8090", "127.0.0.1:8097")
    )


def should_upload_env_key(backend_url: str | None) -> bool:
    if not backend_url:
        return False
    lowered = backend_url.lower()
    if any(
        host in lowered
        for host in ("localhost:8090", "localhost:8097", "127.0.0.1:8090", "127.0.0.1:8097")
    ):
        return False
    return True


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
    task_app_api_key: str | None,
    mode: str,
    seeds: Iterable[int],
) -> Dict[str, Any]:
    seed_list = list(seeds)
    default_val_seeds = [seed + 50 for seed in seed_list]
    policy_model = os.environ.get("BANKING77_POLICY_MODEL", "gpt-4.1-nano")
    policy_provider = os.environ.get("BANKING77_POLICY_PROVIDER", "openai")
    proposer_model = os.environ.get("BANKING77_PROPOSER_MODEL", policy_model)
    proposer_provider = os.environ.get("BANKING77_PROPOSER_PROVIDER", policy_provider)
    proposer_url = os.environ.get(
        "BANKING77_PROPOSER_URL", "https://api.openai.com/v1/responses"
    )
    prompt_learning: Dict[str, Any] = {
        "algorithm": "mipro",
        "task_app_id": "banking77",
        "task_app_url": task_app_url,
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
            "reference_pool": seed_list,
            "val_seeds": default_val_seeds,
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
    }

    if (os.environ.get("MIPRO_PROXY_MODELS") or "").strip().lower() in {"1", "true", "yes", "on"}:
        lo_model = os.environ.get("MIPRO_PROXY_LO_MODEL", policy_model)
        lo_provider = os.environ.get("MIPRO_PROXY_LO_PROVIDER", policy_provider)
        prompt_learning["mipro"]["proxy_models"] = {
            "hi_provider": policy_provider,
            "hi_model": policy_model,
            "lo_provider": lo_provider,
            "lo_model": lo_model,
        }

    if task_app_api_key:
        prompt_learning["task_app_api_key"] = task_app_api_key

    return {
        "prompt_learning": {
            **prompt_learning
        }
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


def get_system_state(backend_url: str, api_key: str, system_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = httpx.get(
        f"{backend_url}/api/prompt-learning/online/mipro/systems/{system_id}/state",
        headers=headers,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def extract_candidate_text(state: Dict[str, Any], candidate_id: str | None) -> str | None:
    if not candidate_id:
        return None
    candidates = state.get("candidates", {}) if isinstance(state, dict) else {}
    if not isinstance(candidates, dict):
        return None
    candidate = candidates.get(candidate_id)
    if not isinstance(candidate, dict):
        return None

    stage_payloads = candidate.get("stage_payloads", {})
    if isinstance(stage_payloads, dict) and stage_payloads:
        for payload in stage_payloads.values():
            if not isinstance(payload, dict):
                continue
            instruction_text = payload.get("instruction_text")
            if isinstance(instruction_text, str) and instruction_text.strip():
                return instruction_text.strip()
            instruction_lines = payload.get("instruction_lines")
            if isinstance(instruction_lines, list) and instruction_lines:
                joined = "\n".join(str(line) for line in instruction_lines)
                if joined.strip():
                    return joined.strip()

    deltas = candidate.get("deltas")
    if isinstance(deltas, dict):
        for key in ("instruction_text", "text", "content"):
            value = deltas.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    baseline_messages = candidate.get("baseline_messages")
    if isinstance(baseline_messages, list):
        for msg in baseline_messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "system":
                content = msg.get("content") or msg.get("pattern")
                if isinstance(content, str) and content.strip():
                    return content.strip()

    return None


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


def run_rollout(
    *,
    task_app_url: str,
    env_key: str,
    seed: int,
    inference_url: str,
    model: str,
    rollout_id: str,
) -> tuple[float, str | None]:
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
    reward_info = body.get("reward_info", {}) if isinstance(body, dict) else {}
    reward = reward_info.get("outcome_reward")
    if reward is None and isinstance(body, dict):
        metrics = body.get("metrics", {}) or {}
        reward = metrics.get("outcome_reward")
        if reward is None:
            reward = (metrics.get("outcome_objectives") or {}).get("reward", 0.0)
    if reward is None:
        reward = (reward_info.get("outcome_objectives") or {}).get("reward", 0.0)

    metadata = body.get("metadata", {}) if isinstance(body, dict) else {}
    candidate_id = metadata.get("mipro_candidate_id")
    if not candidate_id:
        candidate_id = response.headers.get("x-mipro-candidate-id", "")
    candidate_id = str(candidate_id) if candidate_id else None

    rollout_duration = (time.perf_counter() - rollout_start) * 1000.0
    print(
        f"[TIMING] Rollout {rollout_id}: total={rollout_duration:.2}ms "
        f"(task_app_request={request_duration:.2}ms)"
    )
    return float(reward or 0.0), candidate_id


def push_status(
    *,
    backend_url: str,
    api_key: str,
    system_id: str,
    rollout_id: str,
    reward: float,
    candidate_id: str | None,
) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    status_start = time.perf_counter()
    reward_payload = {
        "rollout_id": rollout_id,
        "status": "reward",
        "reward": reward,
    }
    if candidate_id:
        reward_payload["candidate_id"] = candidate_id
    response = httpx.post(
        f"{backend_url}/api/prompt-learning/online/mipro/systems/{system_id}/status",
        json=reward_payload,
        headers=headers,
        timeout=30.0,
    )
    status_duration = (time.perf_counter() - status_start) * 1000.0
    print(f"[TIMING] Status update (reward): {status_duration:.2}ms")
    response.raise_for_status()

    done_start = time.perf_counter()
    done_payload = {"rollout_id": rollout_id, "status": "done"}
    if candidate_id:
        done_payload["candidate_id"] = candidate_id
    response = httpx.post(
        f"{backend_url}/api/prompt-learning/online/mipro/systems/{system_id}/status",
        json=done_payload,
        headers=headers,
        timeout=30.0,
    )
    done_duration = (time.perf_counter() - done_start) * 1000.0
    print(f"[TIMING] Status update (done): {done_duration:.2}ms")
    response.raise_for_status()


def new_rollout_id(seed: int) -> str:
    return f"trace_rollout_{seed}_{uuid.uuid4().hex[:6]}"
