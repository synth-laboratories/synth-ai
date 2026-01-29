#!/usr/bin/env python3
"""
Run MIPRO Continual Learning on EngineBench with difficulty splits.

This script demonstrates continual learning on coding-agent tasks:
- Split 1: easy cards
- Split 2: hard cards

Uses Codex agent by default and routes LLM calls through the MIPRO proxy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Add synth-ai to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_splits import DIFFICULTY_SPLITS, EngineBenchDifficultyDataset
from demos.engine_bench.localapi_engine_bench import (
    DEFAULT_SYSTEM_PROMPT,
    INSTANCE_IDS,
    app,
)
from synth_ai.core.env import mint_demo_api_key
from synth_ai.core.urls import BACKEND_URL_BASE
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port
from synth_ai.sdk.task.override_helpers import get_agent_skills_path

try:
    from synth_ai.sdk.task.server import run_server_background
except ImportError:  # pragma: no cover
    from synth_ai.sdk.task import run_server_background


def resolve_backend_url() -> str:
    """Resolve the backend URL."""
    for env_var in ("SYNTH_URL", "SYNTH_BACKEND_URL", "RUST_BACKEND_URL"):
        env_url = (os.environ.get(env_var) or "").strip()
        if env_url:
            return env_url.rstrip("/")
    return BACKEND_URL_BASE.rstrip("/")


def wait_for_health_check_sync(host: str, port: int, api_key: str, timeout: float = 30.0) -> None:
    """Wait for task app health endpoint."""
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


def build_initial_prompt() -> Dict[str, Any]:
    """Build the initial prompt template for MIPRO."""
    return {
        "id": "engine_bench_continual",
        "name": "EngineBench Continual (Difficulty Splits)",
        "messages": [
            {"role": "system", "order": 0, "pattern": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "order": 1,
                "pattern": "Solve the current Pokemon TCG card implementation task.",
            },
        ],
        "wildcards": {},
    }


def build_mipro_continual_config(
    *,
    task_app_url: str,
    train_seeds: List[int],
    val_seeds: List[int],
    min_rollouts_before_proposal: int = 10,
    system_id: Optional[str] = None,
    system_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build MIPRO online config for continual learning."""
    policy_model = os.environ.get("ENGINE_BENCH_POLICY_MODEL", "gpt-4o-mini")
    policy_provider = os.environ.get("ENGINE_BENCH_POLICY_PROVIDER", "openai")
    proposer_model = os.environ.get("ENGINE_BENCH_PROPOSER_MODEL", "gpt-4.1-mini")
    proposer_provider = os.environ.get("ENGINE_BENCH_PROPOSER_PROVIDER", "openai")

    mipro_section = {
        "mode": "online",
        "bootstrap_train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "online_pool": train_seeds,
        "online_proposer_mode": "inline",
        "online_proposer_min_rollouts": min_rollouts_before_proposal,
        "online_proposer_max_candidates": 50,
        "online_rollouts_per_candidate": 10,
        "ontology": {
            "enabled": True,
            "reads": True,
            "writes": True,
            "batch_proposer": {
                "enabled": True,
                "min_rollouts": 10,
                "batch_size": 40,
                "model": proposer_model,
                "provider": proposer_provider,
                "temperature": 0.7,
                "max_tokens": 1024,
            },
        },
        "proposer": {
            "mode": "instruction_only",
            "model": proposer_model,
            "provider": proposer_provider,
            "temperature": 0.7,
            "max_tokens": 512,
        },
    }

    if system_id:
        mipro_section["system_id"] = system_id
    if system_name:
        mipro_section["system_name"] = system_name

    return {
        "prompt_learning": {
            "algorithm": "mipro",
            "task_app_id": "engine_bench_continual",
            "task_app_url": task_app_url,
            "initial_prompt": build_initial_prompt(),
            "policy": {
                "model": policy_model,
                "provider": policy_provider,
                "inference_mode": "synth_hosted",
                "temperature": 0.0,
                "max_completion_tokens": 256,
            },
            "mipro": mipro_section,
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


def run_rollout(
    *,
    task_app_url: str,
    env_key: str,
    seed: int,
    inference_url: str,
    model: str,
    rollout_id: str,
    split_name: str,
    agent: str,
    timeout_s: int,
    context_overrides: List[Dict[str, Any]] | None = None,
    override_bundle_id: str | None = None,
) -> float:
    """Execute a single rollout for online MIPRO."""
    payload = {
        "trace_correlation_id": rollout_id,
        "env": {
            "seed": seed,
            "config": {
                "difficulty_split": split_name,
            },
        },
        "policy": {
            "config": {
                "model": model,
                "agent": agent,
                "timeout": timeout_s,
                "inference_url": inference_url,
            },
        },
    }
    if context_overrides:
        payload["context_overrides"] = context_overrides
        payload["override_bundle_id"] = override_bundle_id
    headers = {"X-API-Key": env_key}

    response = httpx.post(
        f"{task_app_url}/rollout",
        json=payload,
        headers=headers,
        timeout=timeout_s + 120,
    )
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
    payload = {
        "rollout_id": rollout_id,
        "status": "done",
        "reward": reward,
    }
    response = httpx.post(
        f"{backend_url}/api/prompt-learning/online/mipro/systems/{system_id}/status",
        json=payload,
        headers=headers,
        timeout=30.0,
    )
    response.raise_for_status()
    response_data = response.json() if response.content else {}
    candidate_id = response_data.get("candidate_id", "unknown")
    return str(candidate_id)


def new_rollout_id(split_name: str, seed: int) -> str:
    return f"trace_{split_name}_rollout_{seed}_{uuid.uuid4().hex[:6]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIPRO Continual Learning on EngineBench")
    parser.add_argument("--backend-url", default=None, help="Backend URL")
    parser.add_argument("--local-host", default="localhost", help="Local API hostname")
    parser.add_argument("--local-port", type=int, default=8040, help="Local API port")
    parser.add_argument("--rollouts-per-split", type=int, default=20, help="Rollouts per split")
    parser.add_argument("--model", default="gpt-4o-mini", help="Policy model to use")
    parser.add_argument("--agent", default="codex", help="Agent type: codex|opencode|claude_code")
    parser.add_argument("--timeout", type=int, default=600, help="Agent timeout (seconds)")
    parser.add_argument("--agents-md", type=str, default=None, help="Path to AGENTS.md override")
    parser.add_argument(
        "--skills-yaml",
        type=str,
        default=None,
        help="Path to agent skills.yaml override (written to agent-specific path)",
    )
    parser.add_argument("--train-size", type=int, default=20, help="Training seeds count")
    parser.add_argument("--val-size", type=int, default=10, help="Validation seeds count")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--system-id", type=str, default=None, help="Reuse existing system_id")
    parser.add_argument("--system-name", type=str, default=None, help="Human-readable system name")
    args = parser.parse_args()

    backend_url = (args.backend_url or resolve_backend_url()).rstrip("/")
    print(f"Backend URL: {backend_url}")
    print(f"EngineBench instances: {len(INSTANCE_IDS)}")

    api_key = os.environ.get("SYNTH_API_KEY", "").strip()
    if not api_key:
        print("Minting demo API key...")
        api_key = mint_demo_api_key(backend_url=backend_url)
    print(f"API Key: {api_key[:20]}...")

    try:
        r = httpx.get(f"{backend_url}/health", timeout=30)
        print(f"Backend health: {r.status_code}")
    except Exception as exc:
        print(f"WARNING: Backend health check failed: {exc}")

    env_key = ensure_localapi_auth(backend_base=backend_url, synth_api_key=api_key)
    os.environ["ENVIRONMENT_API_KEY"] = env_key
    print(f"Environment key: {env_key[:12]}...{env_key[-4:]}")

    if args.model:
        os.environ["ENGINE_BENCH_POLICY_MODEL"] = args.model

    port = acquire_port(args.local_port, on_conflict=PortConflictBehavior.FIND_NEW)
    if port != args.local_port:
        print(f"Port {args.local_port} in use, using port {port} instead")
    run_server_background(app, port)
    print(f"Waiting for local API on port {port}...")
    wait_for_health_check_sync("localhost", port, env_key, timeout=60.0)
    task_app_url = f"http://{args.local_host}:{port}"
    print(f"Local API URL: {task_app_url}")

    dataset = EngineBenchDifficultyDataset()
    split_stats = {split: dataset.split_stats(split) for split in DIFFICULTY_SPLITS}

    context_overrides: List[Dict[str, Any]] | None = None
    override_bundle_id = None
    if args.agents_md or args.skills_yaml:
        file_artifacts: Dict[str, str] = {}
        if args.agents_md:
            agents_path = Path(args.agents_md)
            if not agents_path.exists():
                raise FileNotFoundError(f"AGENTS.md not found: {agents_path}")
            file_artifacts["AGENTS.md"] = agents_path.read_text()
        if args.skills_yaml:
            skills_path = Path(args.skills_yaml)
            if not skills_path.exists():
                raise FileNotFoundError(f"skills.yaml not found: {skills_path}")
            skills_rel = get_agent_skills_path(args.agent, global_=False)
            file_artifacts[skills_rel] = skills_path.read_text()
        if file_artifacts:
            context_overrides = [
                {
                    "file_artifacts": file_artifacts,
                    "env_vars": {},
                    "preflight_script": None,
                    "mutation_type": "manual",
                }
            ]
            override_bundle_id = "enginebench_codex_context"

    print("\n" + "=" * 60)
    print("EngineBench Difficulty Splits")
    for split in DIFFICULTY_SPLITS:
        stats = split_stats[split]
        print(
            f"  {split}: {stats['count']} instances | "
            f"avg score {stats['avg_score']:.2f} | threshold {stats['threshold']}"
        )
    print("=" * 60)

    train_seeds = list(range(args.train_size))
    val_seeds = list(range(args.train_size, args.train_size + args.val_size))

    config_body = build_mipro_continual_config(
        task_app_url=task_app_url,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        min_rollouts_before_proposal=10,
        system_id=args.system_id,
        system_name=args.system_name,
    )

    print("\nCreating MIPRO online job...")
    start_time = time.time()
    job_id = create_job(backend_url, api_key, config_body)
    print(f"Job ID: {job_id}")

    detail = get_job_detail(backend_url, api_key, job_id, include_metadata=True)
    metadata = detail.get("metadata", {})
    system_id = metadata.get("mipro_system_id")
    proxy_url = metadata.get("mipro_proxy_url")

    if not system_id or not proxy_url:
        raise RuntimeError(f"Missing mipro_system_id or mipro_proxy_url in metadata: {metadata}")

    print(f"System ID: {system_id}")
    print(f"Proxy URL: {proxy_url}")

    all_results = {
        "method": "mipro_continual_engine_bench",
        "job_id": job_id,
        "system_id": system_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "backend_url": backend_url,
            "model": args.model,
            "agent": args.agent,
            "timeout": args.timeout,
            "rollouts_per_split": args.rollouts_per_split,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "difficulty_splits": split_stats,
        },
        "split_results": {},
        "checkpoints": [],
    }

    total_rollouts = 0
    total_reward = 0.0

    status_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="status_upload")
    pending_status_futures: List[Future] = []

    def async_push_status(rollout_id: str, reward: float) -> Future:
        return status_executor.submit(
            push_status,
            backend_url=backend_url,
            api_key=api_key,
            system_id=system_id,
            rollout_id=rollout_id,
            reward=reward,
        )

    for split_name in DIFFICULTY_SPLITS:
        split_start = time.time()
        split_rollouts = 0
        split_reward = 0.0
        split_ids = dataset.split_ids(split_name)

        print(f"\n{'=' * 60}")
        print(f"SPLIT {split_name.upper()} ({len(split_ids)} instances)")
        print(f"{'=' * 60}")

        last_candidate_id = "unknown"
        candidate_stats: Dict[str, Dict[str, Any]] = {}

        for i in range(args.rollouts_per_split):
            seed = i % len(split_ids)
            rollout_id = new_rollout_id(split_name, i)
            inference_url = f"{proxy_url}/{rollout_id}/chat/completions"

            try:
                reward = run_rollout(
                    task_app_url=task_app_url,
                    env_key=env_key,
                    seed=seed,
                    inference_url=inference_url,
                    model=args.model,
                    rollout_id=rollout_id,
                    split_name=split_name,
                    agent=args.agent,
                    timeout_s=args.timeout,
                    context_overrides=context_overrides,
                    override_bundle_id=override_bundle_id,
                )
                split_reward += reward
                total_reward += reward
                split_rollouts += 1
                total_rollouts += 1

                future = async_push_status(rollout_id, reward)
                pending_status_futures.append(future)

                completed = [f for f in pending_status_futures if f.done()]
                for f in completed:
                    try:
                        candidate_id = f.result()
                        last_candidate_id = candidate_id
                        if candidate_id and candidate_id != "unknown":
                            candidate_stats.setdefault(candidate_id, {"count": 0})
                            candidate_stats[candidate_id]["count"] += 1
                    except Exception as exc:
                        print(f"  Status upload error: {exc}")
                    pending_status_futures.remove(f)

                if (i + 1) % 5 == 0:
                    running_score = split_reward / split_rollouts if split_rollouts else 0.0
                    print(
                        f"  Progress: {i+1}/{args.rollouts_per_split} | "
                        f"Avg reward: {running_score:.3f} | Candidate: {last_candidate_id}"
                    )
            except Exception as exc:
                print(f"  Error on rollout {i}: {exc}")

        for f in pending_status_futures:
            try:
                candidate_id = f.result(timeout=30.0)
                if candidate_id and candidate_id != "unknown":
                    candidate_stats.setdefault(candidate_id, {"count": 0})
                    candidate_stats[candidate_id]["count"] += 1
            except Exception as exc:
                print(f"  Status upload error: {exc}")
        pending_status_futures.clear()

        split_elapsed = time.time() - split_start
        split_avg_reward = split_reward / split_rollouts if split_rollouts else 0.0

        state = get_system_state(backend_url, api_key, system_id)
        best_candidate_id = state.get("best_candidate_id")
        best_candidate_text = extract_candidate_text(state, best_candidate_id)

        checkpoint = {
            "split": split_name,
            "split_avg_reward": split_avg_reward,
            "total_rollouts_so_far": total_rollouts,
            "elapsed_seconds": split_elapsed,
            "best_candidate_id": best_candidate_id,
            "best_candidate_text": best_candidate_text[:500] + "..."
            if best_candidate_text and len(best_candidate_text) > 500
            else best_candidate_text,
            "candidate_stats": candidate_stats,
        }
        all_results["checkpoints"].append(checkpoint)
        all_results["split_results"][split_name] = {
            "avg_reward": split_avg_reward,
            "total": split_rollouts,
            "elapsed_seconds": split_elapsed,
        }

        print(f"\n  Split {split_name} Summary:")
        print(f"    Avg reward: {split_avg_reward:.3f}")
        print(f"    Time: {split_elapsed:.1f}s")
        print(f"    Best candidate: {best_candidate_id}")
        if best_candidate_text:
            preview = (
                best_candidate_text[:200] + "..."
                if len(best_candidate_text) > 200
                else best_candidate_text
            )
            print(f"    Best prompt: {preview}")

        if split_name != DIFFICULTY_SPLITS[-1]:
            print("\nPausing 20s between splits...")
            time.sleep(20)

    total_elapsed = time.time() - start_time
    all_results["total_elapsed_seconds"] = total_elapsed
    all_results["final_avg_reward"] = total_reward / total_rollouts if total_rollouts else 0.0

    print("\n" + "=" * 70)
    print("MIPRO CONTINUAL LEARNING - FINAL RESULTS")
    print("=" * 70)
    for split_name in DIFFICULTY_SPLITS:
        sr = all_results["split_results"].get(split_name, {})
        avg_reward = sr.get("avg_reward", 0.0)
        elapsed = sr.get("elapsed_seconds", 0.0)
        print(f"{split_name:<6} | avg reward: {avg_reward:.3f} | time: {elapsed:.1f}s")
    print("-" * 70)
    print(f"Total rollouts: {total_rollouts}")
    print(f"Overall avg reward: {all_results['final_avg_reward']:.3f}")
    print(f"Total time: {total_elapsed:.1f}s")
    print("=" * 70)

    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"mipro_engine_bench_{timestamp}.json"

    status_executor.shutdown(wait=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
