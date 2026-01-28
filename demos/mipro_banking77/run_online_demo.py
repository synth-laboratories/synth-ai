#!/usr/bin/env python3
"""
Run Banking77 MIPRO Online Optimization

This script runs MIPRO optimization in online mode on Banking77 using the synth-ai SDK.
Online MIPRO means you drive rollouts locally, and the backend provides a proxy URL
that selects prompt candidates for each LLM call.

Key differences from offline mode:
- No tunneling required (backend never calls your task app)
- You control the rollout loop
- Real-time prompt evolution as rewards are reported

Usage:
    uv run python demos/mipro_banking77/run_online_demo.py --rollouts 100
    uv run python demos/mipro_banking77/run_online_demo.py --rollouts 1000 --model gpt-4.1-nano
    
    # With production backend
    SYNTH_BACKEND_URL=https://api.usesynth.ai uv run python demos/mipro_banking77/run_online_demo.py --rollouts 100
"""

from __future__ import annotations

import argparse
import json
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
from synth_ai.core.utils.urls import BACKEND_URL_BASE


def wait_for_health_check_sync(host: str, port: int, api_key: str, timeout: float = 30.0) -> None:
    """Wait for task app health endpoint to become available."""
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
    """Resolve the backend URL to use for MIPRO jobs."""
    # Check environment variables first
    for env_var in ("SYNTH_URL", "SYNTH_BACKEND_URL", "RUST_BACKEND_URL"):
        env_url = (os.environ.get(env_var) or "").strip()
        if env_url:
            return env_url.rstrip("/")

    # Try local backends
    candidates = ["http://localhost:8000", "http://localhost:8001"]
    for candidate in candidates:
        try:
            response = httpx.get(f"{candidate}/health", timeout=3.0)
            if response.status_code == 200:
                return candidate.rstrip("/")
        except (httpx.RequestError, httpx.TimeoutException):
            continue

    return BACKEND_URL_BASE.rstrip("/")


def build_initial_prompt() -> Dict[str, Any]:
    """Build the initial prompt template for Banking77 classification."""
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


def build_mipro_online_config(
    *,
    task_app_url: str,
    train_seeds: list[int],
    val_seeds: list[int],
    min_rollouts_before_proposal: int = 20,
) -> Dict[str, Any]:
    """Build a MIPRO online job configuration.
    
    Note: Proposal generation is handled entirely by the backend service.
    The client just runs rollouts and reports rewards. The proposer API key
    is resolved from the backend's environment (OPENAI_API_KEY or PROD_OPENAI_API_KEY).
    """
    policy_model = os.environ.get("BANKING77_POLICY_MODEL", "gpt-4.1-nano")
    policy_provider = os.environ.get("BANKING77_POLICY_PROVIDER", "openai")
    proposer_model = os.environ.get("BANKING77_PROPOSER_MODEL", "gpt-4.1-mini")
    proposer_provider = os.environ.get("BANKING77_PROPOSER_PROVIDER", "openai")
    
    return {
        "prompt_learning": {
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
                "mode": "online",
                "bootstrap_train_seeds": train_seeds,
                "val_seeds": val_seeds,
                "online_pool": train_seeds,
                "online_proposer_mode": "inline",
                "online_proposer_min_rollouts": min_rollouts_before_proposal,
                # How many rollouts to assign to each candidate before picking a new one
                "online_rollouts_per_candidate": 10,
                # Proposer config - API key is resolved from backend environment
                "proposer": {
                    "mode": "instruction_only",
                    "model": proposer_model,
                    "provider": proposer_provider,
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "generate_at_iterations": [0, 1, 2],
                    "instructions_per_batch": 2,
                    # api_key NOT included - resolved from backend's OPENAI_API_KEY env var
                },
            },
        },
    }


def create_job(backend_url: str, api_key: str, config_body: Dict[str, Any]) -> str:
    """Create a MIPRO job on the backend."""
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
    """Get details for a MIPRO job."""
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
    """Get the current state of a MIPRO system."""
    headers = {"Authorization": f"Bearer {api_key}"}
    response = httpx.get(
        f"{backend_url}/api/prompt-learning/online/mipro/systems/{system_id}/state",
        headers=headers,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def extract_candidate_text(state: Dict[str, Any], candidate_id: str | None) -> str | None:
    """Extract the prompt text from a candidate in the system state."""
    if not candidate_id:
        return None
    candidates = state.get("candidates", {}) if isinstance(state, dict) else {}
    if not isinstance(candidates, dict):
        return None
    candidate = candidates.get(candidate_id)
    if not isinstance(candidate, dict):
        return None

    # Try stage_payloads first
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

    # Try deltas
    deltas = candidate.get("deltas")
    if isinstance(deltas, dict):
        for key in ("instruction_text", "text", "content"):
            value = deltas.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    # Try baseline_messages
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


def run_rollout(
    *,
    task_app_url: str,
    env_key: str,
    seed: int,
    inference_url: str,
    model: str,
    rollout_id: str,
) -> float:
    """Execute a single rollout for online MIPRO."""
    payload = {
        "trace_correlation_id": rollout_id,
        "env": {"seed": seed, "config": {"seed": seed, "split": "train"}},
        "policy": {"config": {"model": model, "inference_url": inference_url}},
    }
    headers = {"X-API-Key": env_key}
    response = httpx.post(
        f"{task_app_url}/rollout",
        json=payload,
        headers=headers,
        timeout=120.0,
    )
    response.raise_for_status()
    body = response.json()
    
    # Extract reward
    reward_info = body.get("reward_info", {}) if isinstance(body, dict) else {}
    reward = reward_info.get("outcome_reward")
    if reward is None and isinstance(body, dict):
        metrics = body.get("metrics", {}) or {}
        reward = metrics.get("outcome_reward")
        if reward is None:
            reward = (metrics.get("outcome_objectives") or {}).get("reward", 0.0)
    if reward is None:
        reward = (reward_info.get("outcome_objectives") or {}).get("reward", 0.0)

    return float(reward or 0.0)


def push_status(
    *,
    backend_url: str,
    api_key: str,
    system_id: str,
    rollout_id: str,
    reward: float,
) -> str:
    """Report rollout results to the backend for online MIPRO."""
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Send reward status
    reward_payload = {
        "rollout_id": rollout_id,
        "status": "reward",
        "reward": reward,
    }
    response = httpx.post(
        f"{backend_url}/api/prompt-learning/online/mipro/systems/{system_id}/status",
        json=reward_payload,
        headers=headers,
        timeout=30.0,
    )
    response.raise_for_status()
    reward_response = response.json() if response.content else {}
    candidate_id = reward_response.get("candidate_id")
    if not candidate_id:
        raise RuntimeError(
            "Missing candidate_id in backend status response. "
            "Backend must return candidate_id for MIPRO rollouts."
        )

    # Send done status
    done_payload = {"rollout_id": rollout_id, "status": "done"}
    response = httpx.post(
        f"{backend_url}/api/prompt-learning/online/mipro/systems/{system_id}/status",
        json=done_payload,
        headers=headers,
        timeout=30.0,
    )
    response.raise_for_status()
    return str(candidate_id)


def new_rollout_id(seed: int) -> str:
    """Generate a unique rollout ID for online MIPRO."""
    return f"trace_rollout_{seed}_{uuid.uuid4().hex[:6]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Banking77 MIPRO Online Optimization")
    parser.add_argument(
        "--backend-url",
        default=None,
        help="Backend base URL (defaults to SYNTH_BACKEND_URL, RUST_BACKEND_URL, or SDK default)",
    )
    parser.add_argument("--local-host", default="localhost", help="Local API hostname")
    parser.add_argument("--local-port", type=int, default=8016, help="Local API port")
    parser.add_argument("--rollouts", type=int, default=100, help="Number of rollouts to run")
    parser.add_argument("--train-size", type=int, default=30, help="Training seeds count")
    parser.add_argument("--val-size", type=int, default=20, help="Validation seeds count")
    parser.add_argument("--model", type=str, default="gpt-4.1-nano", help="Model to use for inference")
    parser.add_argument("--output", type=str, default=None, help="Output file for results (JSON)")
    parser.add_argument("--min-proposal-rollouts", type=int, default=20, 
                       help="Minimum rollouts before generating new proposals")
    args = parser.parse_args()

    # Resolve backend URL
    backend_url = (args.backend_url or resolve_backend_url()).rstrip("/")
    print(f"Backend URL: {backend_url}")
    
    
    # Get API key
    api_key = os.environ.get("SYNTH_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SYNTH_API_KEY is required. Set it with: export SYNTH_API_KEY=sk_live_...")
    print(f"API Key: {api_key[:20]}...")

    # Check backend health
    try:
        r = httpx.get(f"{backend_url}/health", timeout=30)
        if r.status_code != 200:
            print(f"WARNING: Backend health check returned status {r.status_code}")
        else:
            print(f"Backend health: OK")
    except Exception as e:
        print(f"WARNING: Backend health check failed: {e}")

    # Set up environment key (for local task app auth)
    env_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
    if not env_key:
        env_key = ensure_localapi_auth(backend_base=None, upload=False, persist=False)
    os.environ["ENVIRONMENT_API_KEY"] = env_key
    print(f"Environment key: {env_key[:12]}...{env_key[-4:]}")

    # Set model for the task app
    if args.model:
        os.environ["BANKING77_POLICY_MODEL"] = args.model
    print(f"Model: {os.environ.get('BANKING77_POLICY_MODEL', 'gpt-4.1-nano')}")

    # Start local task app
    port = acquire_port(args.local_port, on_conflict=PortConflictBehavior.FIND_NEW)
    if port != args.local_port:
        print(f"Port {args.local_port} in use, using port {port} instead")

    run_server_background(localapi_banking77.app, port)
    print(f"Waiting for local API on port {port}...")
    wait_for_health_check_sync("localhost", port, env_key, timeout=30.0)
    task_app_url = f"http://{args.local_host}:{port}"
    print(f"Local API URL: {task_app_url}")

    # Build seeds
    train_seeds = list(range(args.train_size))
    val_seeds = list(range(args.train_size, args.train_size + args.val_size))
    
    print("\n" + "="*60)
    print("MIPRO Online Configuration")
    print("="*60)
    print(f"  Rollouts: {args.rollouts}")
    print(f"  Train seeds: {args.train_size}")
    print(f"  Val seeds: {args.val_size}")
    print(f"  Min rollouts before proposal: {args.min_proposal_rollouts}")
    print("="*60)

    # Build config and create job
    config_body = build_mipro_online_config(
        task_app_url=task_app_url,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        min_rollouts_before_proposal=args.min_proposal_rollouts,
    )

    print("\nCreating MIPRO online job...")
    start_time = time.time()
    job_id = create_job(backend_url, api_key, config_body)
    print(f"Job ID: {job_id}")

    # Get job details to get system_id and proxy_url
    detail = get_job_detail(backend_url, api_key, job_id, include_metadata=True)
    metadata = detail.get("metadata", {})
    system_id = metadata.get("mipro_system_id")
    proxy_url = metadata.get("mipro_proxy_url")
    
    if not system_id or not proxy_url:
        raise RuntimeError(f"Missing mipro_system_id or mipro_proxy_url in metadata: {metadata}")
    
    print(f"System ID: {system_id}")
    print(f"Proxy URL: {proxy_url}")

    # Run rollouts
    print("\n" + "="*60)
    print("Running Online Rollouts")
    print("="*60)

    model = os.environ.get("BANKING77_MODEL", args.model)
    rollout_results = []
    correct_count = 0
    candidate_stats: Dict[str, Dict[str, Any]] = {}

    for i in range(args.rollouts):
        seed = i % len(train_seeds)  # Cycle through seeds
        rollout_id = new_rollout_id(i)
        inference_url = f"{proxy_url}/{rollout_id}/chat/completions"
        
        try:
            reward = run_rollout(
                task_app_url=task_app_url,
                env_key=env_key,
                seed=seed,
                inference_url=inference_url,
                model=model,
                rollout_id=rollout_id,
            )
            
            if reward > 0:
                correct_count += 1
            
            rollout_results.append({
                "rollout_id": rollout_id,
                "seed": seed,
                "reward": reward,
            })
            
            # Push status to backend
            candidate_id = push_status(
                backend_url=backend_url,
                api_key=api_key,
                system_id=system_id,
                rollout_id=rollout_id,
                reward=reward,
            )
            
            # Track candidate statistics from backend
            if candidate_id:
                if candidate_id not in candidate_stats:
                    candidate_stats[candidate_id] = {"count": 0, "total_reward": 0.0}
                candidate_stats[candidate_id]["count"] += 1
                candidate_stats[candidate_id]["total_reward"] += reward
            
            # Progress update every 10 rollouts
            if (i + 1) % 10 == 0:
                running_acc = correct_count / (i + 1)
                print(
                    f"  Progress: {i+1}/{args.rollouts} | "
                    f"Running accuracy: {running_acc:.1%} | "
                    f"Candidate: {candidate_id}"
                )
                
        except Exception as e:
            print(f"  Error on rollout {i}: {e}")
            rollout_results.append({
                "rollout_id": rollout_id,
                "seed": seed,
                "reward": 0.0,
                "error": str(e),
            })

    elapsed = time.time() - start_time
    
    # Get final system state
    print("\nFetching final system state...")
    state = get_system_state(backend_url, api_key, system_id)
    best_candidate_id = state.get("best_candidate_id")
    best_candidate_text = extract_candidate_text(state, best_candidate_id)
    num_candidates = len(state.get("candidates", {}))

    # Calculate final statistics
    final_accuracy = correct_count / len(rollout_results) if rollout_results else 0.0
    
    # Print summary
    print("\n" + "="*60)
    print("MIPRO Online Results")
    print("="*60)
    print(f"Total rollouts: {len(rollout_results)}")
    print(f"Overall accuracy: {final_accuracy:.1%} ({correct_count}/{len(rollout_results)})")
    print(f"Total candidates explored: {num_candidates}")
    print(f"Best candidate: {best_candidate_id}")
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"Throughput: {len(rollout_results)/elapsed:.1f} rollouts/sec")
    
    # Print candidate statistics
    print("\n" + "-"*60)
    print("Candidate Performance:")
    print("-"*60)
    for cid, stats in sorted(candidate_stats.items(), key=lambda x: x[1]["total_reward"]/max(x[1]["count"], 1), reverse=True):
        avg_reward = stats["total_reward"] / max(stats["count"], 1)
        print(f"  {cid}: {stats['count']} rollouts, avg reward: {avg_reward:.3f}")
    
    if best_candidate_text:
        print("\n" + "-"*60)
        print("Best Candidate Prompt:")
        print("-"*60)
        preview = best_candidate_text[:500] + ("..." if len(best_candidate_text) > 500 else "")
        print(preview)
    
    print("="*60)

    # Save results
    results = {
        "method": "synth_mipro_online",
        "job_id": job_id,
        "system_id": system_id,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "backend_url": backend_url,
            "model": model,
            "rollouts": args.rollouts,
            "train_size": args.train_size,
            "val_size": args.val_size,
        },
        "results": {
            "total_rollouts": len(rollout_results),
            "correct": correct_count,
            "accuracy": final_accuracy,
            "num_candidates": num_candidates,
            "best_candidate_id": best_candidate_id,
            "best_candidate_text": best_candidate_text,
            "candidate_stats": candidate_stats,
        },
        "rollouts": rollout_results,
    }

    # Save to file if specified
    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"banking77_mipro_online_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
