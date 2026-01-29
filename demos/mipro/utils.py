#!/usr/bin/env python3
"""Shared helpers for MIPRO demos (online + offline).

This module provides utility functions for running MIPRO (Multi-prompt Instruction
PRoposal Optimizer) in both online and offline modes.

MIPRO (https://arxiv.org/abs/2406.11695) is an algorithm for optimizing Language
Model (LM) programs by improving free-form instructions and few-shot demonstrations
without requiring module-level labels or gradients. It combines:
    - **Instruction Proposal**: Program- and data-aware techniques for proposing
      effective instructions
    - **Surrogate Modeling**: Stochastic mini-batch evaluation for learning a
      surrogate model of the objective
    - **Meta-Optimization**: Refining how LMs construct proposals over time

Key concepts:
    - **Online MIPRO**: User drives rollouts locally. No tunneling or ENVIRONMENT_API_KEY
      required since the backend never calls the task app.
    - **Offline MIPRO**: Backend orchestrates all rollouts. Requires tunneling to expose
      local task app and ENVIRONMENT_API_KEY for authentication.

Example usage (online mode)::

    from demos.mipro.utils import (
        build_mipro_config,
        create_job,
        run_rollout,
        push_status,
    )

    # Create job config (no task_app_api_key needed for online)
    config = build_mipro_config(
        task_app_url="http://localhost:8016",
        task_app_api_key=None,  # Not needed for online mode
        mode="online",
        seeds=range(20),
    )

    # Create job and run rollouts locally
    job_id = create_job(backend_url, api_key, config)
    # ... run rollouts through proxy URL ...
"""

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
    """Wait for task app health endpoint to become available.

    Polls the /health endpoint of the task app until it returns a successful
    response or the timeout is exceeded.

    Args:
        host: Hostname of the task app (e.g., "localhost").
        port: Port number the task app is running on.
        api_key: Environment API key for authentication.
        timeout: Maximum time in seconds to wait. Defaults to 30.0.

    Raises:
        RuntimeError: If health check fails after timeout.

    Example::

        wait_for_health_check_sync("localhost", 8016, env_key, timeout=30.0)
    """
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
    """Resolve the backend URL to use for MIPRO jobs.

    Resolution order:
        1. SYNTH_BACKEND_URL environment variable if set
        2. Localhost on ports 8000 or 8001 if a backend is running there
        3. Default Synth backend URL (BACKEND_URL_BASE)

    Returns:
        Backend URL without trailing slash.

    Example::

        backend_url = resolve_backend_url()
        # Returns "http://localhost:8000" if local backend is running,
        # otherwise returns the configured or default URL.
    """
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
    """Check if the backend URL points to a local server.

    Args:
        backend_url: The backend URL to check.

    Returns:
        True if the URL contains "localhost" or "127.0.0.1", False otherwise.
    """
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
    """Start the Banking77 task app as a local API server.

    This function:
        1. Ensures ENVIRONMENT_API_KEY is set (generates one if needed)
        2. Acquires an available port (finding a new one if the requested port is in use)
        3. Starts the Banking77 task app in the background
        4. Waits for the health check to pass

    Args:
        local_host: Hostname to bind the server to.
        local_port: Preferred port to run the server on.
        backend_url: Backend URL (used to determine if env key should be uploaded).

    Returns:
        Tuple of (task_app_url, env_key, actual_port):
            - task_app_url: Full URL of the running task app
            - env_key: The ENVIRONMENT_API_KEY being used
            - actual_port: The port the server is running on (may differ from requested)

    Example::

        task_app_url, env_key, port = start_local_api(
            local_host="localhost",
            local_port=8016,
            backend_url="https://api-dev.usesynth.ai",
        )
    """
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
    """Determine if task_app_api_key should be included in the config.

    For certain local backend configurations, the task_app_api_key must be
    included directly in the config rather than fetched from the database.

    Args:
        backend_url: The backend URL being used.

    Returns:
        True if task_app_api_key should be included in config.

    Note:
        Can be overridden with MIPRO_INCLUDE_TASK_APP_KEY environment variable.
    """
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
    """Determine if ENVIRONMENT_API_KEY should be uploaded to the backend.

    For remote backends, the env key must be registered in the database so the
    backend can authenticate requests to the task app. For certain local
    configurations, this is not needed.

    Args:
        backend_url: The backend URL being used.

    Returns:
        True if the env key should be uploaded to the backend.
    """
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
    """Build the initial prompt template for Banking77 classification.

    Creates a prompt pattern with system and user messages for classifying
    customer queries into banking intents.

    Returns:
        Dict containing the prompt pattern with:
            - id: Unique identifier for the pattern
            - name: Human-readable name
            - messages: List of message templates with placeholders
            - wildcards: Mapping of placeholder names to requirements
    """
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
    """Build a MIPRO job configuration.

    Creates a complete configuration for either online or offline MIPRO optimization.

    Args:
        task_app_url: URL of the task app (e.g., "http://localhost:8016").
            For online mode, this is used locally; backend never calls it.
        task_app_api_key: ENVIRONMENT_API_KEY for task app authentication.
            **Not needed for online mode** - pass None.
            Required for offline mode (backend needs it to call task app).
        mode: Either "online" or "offline".
            - "online": User drives rollouts locally through proxy URL.
            - "offline": Backend orchestrates all rollouts.
        seeds: Iterable of seed integers for training examples.

    Returns:
        Configuration dict ready to pass to create_job().

    Example (online mode)::

        config = build_mipro_config(
            task_app_url="http://localhost:8016",
            task_app_api_key=None,  # Not needed for online
            mode="online",
            seeds=range(20),
        )

    Example (offline mode)::

        config = build_mipro_config(
            task_app_url="https://xxx.trycloudflare.com",
            task_app_api_key="sk_env_xxx",  # Required for offline
            mode="offline",
            seeds=range(50),
        )

    Environment variables:
        BANKING77_POLICY_MODEL: Model for policy (default: gpt-4.1-nano)
        BANKING77_POLICY_PROVIDER: Provider for policy (default: openai)
        BANKING77_PROPOSER_MODEL: Model for proposer (default: same as policy)
        BANKING77_PROPOSER_PROVIDER: Provider for proposer (default: same as policy)
    """
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
    """Create a MIPRO job on the backend.

    Submits a new MIPRO optimization job and returns the job ID.

    Args:
        backend_url: Backend base URL (e.g., "https://api-dev.usesynth.ai").
        api_key: Synth API key (SYNTH_API_KEY) for authentication.
        config_body: Job configuration from build_mipro_config().

    Returns:
        Job ID string (e.g., "pl_xxxxxxxxxxxxxxxx").

    Raises:
        httpx.HTTPStatusError: If job creation fails.
        RuntimeError: If response is missing job_id.

    Example::

        job_id = create_job(backend_url, api_key, config)
        print(f"Created job: {job_id}")
    """
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
    """Get details for a MIPRO job.

    For online MIPRO, the metadata contains the mipro_system_id and mipro_proxy_url
    needed to drive rollouts.

    Args:
        backend_url: Backend base URL.
        api_key: Synth API key for authentication.
        job_id: Job ID returned from create_job().
        include_metadata: Whether to include metadata (contains proxy URL).

    Returns:
        Job detail dict including status, metadata, etc.

    Example::

        detail = get_job_detail(backend_url, api_key, job_id)
        system_id = detail["metadata"]["mipro_system_id"]
        proxy_url = detail["metadata"]["mipro_proxy_url"]
    """
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
    """Get the current state of a MIPRO system.

    The state includes all candidates, their scores, and the current best candidate.

    Args:
        backend_url: Backend base URL.
        api_key: Synth API key for authentication.
        system_id: MIPRO system ID from job metadata.

    Returns:
        System state dict containing:
            - best_candidate_id: ID of the best performing candidate
            - version: Current version number
            - candidates: Dict of all candidates and their data

    Example::

        state = get_system_state(backend_url, api_key, system_id)
        best_id = state["best_candidate_id"]
        print(f"Best candidate: {best_id}")
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    response = httpx.get(
        f"{backend_url}/api/prompt-learning/online/mipro/systems/{system_id}/state",
        headers=headers,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def extract_candidate_text(state: Dict[str, Any], candidate_id: str | None) -> str | None:
    """Extract the prompt text from a candidate in the system state.

    Searches through various locations in the candidate data to find the
    optimized prompt text (instruction text, deltas, or baseline messages).

    Args:
        state: System state from get_system_state().
        candidate_id: ID of the candidate to extract text from.

    Returns:
        The prompt text string, or None if not found.

    Example::

        state = get_system_state(backend_url, api_key, system_id)
        text = extract_candidate_text(state, state["best_candidate_id"])
        if text:
            print(f"Best prompt:\\n{text}")
    """
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
    """Poll a job until completion (for offline MIPRO).

    Continuously checks job status until it reaches "succeeded" or "failed".
    Primarily used for offline mode where the backend orchestrates rollouts.

    Args:
        backend_url: Backend base URL.
        api_key: Synth API key for authentication.
        job_id: Job ID to poll.
        timeout: Maximum time to wait in seconds. Defaults to 600.0 (10 minutes).
        interval: Time between polls in seconds. Defaults to 2.0.

    Returns:
        Final job detail dict when job completes.

    Raises:
        TimeoutError: If job doesn't complete within timeout.

    Example::

        detail = poll_job(backend_url, api_key, job_id, timeout=300.0)
        print(f"Job finished with status: {detail['status']}")
    """
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
    """Execute a single rollout for online MIPRO.

    Sends a rollout request to the local task app. The task app makes LLM calls
    through the proxy URL, which injects the selected candidate's prompt.

    Args:
        task_app_url: Local task app URL (e.g., "http://localhost:8016").
        env_key: ENVIRONMENT_API_KEY for task app authentication.
        seed: Seed value determining which example to use.
        inference_url: Proxy URL for LLM calls (includes rollout_id).
            Format: "{proxy_url}/{rollout_id}/chat/completions"
        model: Model name for inference (e.g., "gpt-4.1-nano").
        rollout_id: Unique identifier for this rollout.

    Returns:
        Tuple of (reward, candidate_id):
            - reward: Float reward value (0.0 to 1.0 for accuracy)
            - candidate_id: ID of the candidate that was used (or None)

    Example::

        inference_url = f"{proxy_url}/{rollout_id}/chat/completions"
        reward, candidate_id = run_rollout(
            task_app_url="http://localhost:8016",
            env_key=env_key,
            seed=0,
            inference_url=inference_url,
            model="gpt-4.1-nano",
            rollout_id="trace_rollout_0_abc123",
        )
        print(f"Reward: {reward}, Candidate: {candidate_id}")
    """
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
    """Report rollout results to the backend for online MIPRO.

    After completing a rollout locally, call this to report the reward back
    to the backend. The backend uses these rewards to update candidate scores
    and potentially generate new candidates.

    This sends two status updates:
        1. "reward" status with the reward value
        2. "done" status to mark the rollout complete

    Args:
        backend_url: Backend base URL.
        api_key: Synth API key for authentication.
        system_id: MIPRO system ID from job metadata.
        rollout_id: Rollout ID used in run_rollout().
        reward: Reward value from the rollout (0.0 to 1.0).
        candidate_id: ID of the candidate that was used (from run_rollout()).

    Example::

        push_status(
            backend_url=backend_url,
            api_key=api_key,
            system_id=system_id,
            rollout_id=rollout_id,
            reward=1.0,
            candidate_id="baseline",
        )
    """
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
    """Generate a unique rollout ID for online MIPRO.

    Creates an ID that includes the seed for debuggability and a random
    suffix for uniqueness.

    Args:
        seed: Seed value for this rollout.

    Returns:
        Unique rollout ID string (e.g., "trace_rollout_0_a1b2c3").

    Example::

        rollout_id = new_rollout_id(0)  # "trace_rollout_0_abc123"
    """
    return f"trace_rollout_{seed}_{uuid.uuid4().hex[:6]}"
