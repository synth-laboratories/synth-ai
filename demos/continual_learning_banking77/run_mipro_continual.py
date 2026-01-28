#!/usr/bin/env python3
"""
Run MIPRO Continual Learning on Banking77 Progressive Splits

This script demonstrates continual learning:
- Initialize MIPRO system with Split 1 data
- Stream data from Split 1 → 2 → 3 → 4 sequentially
- Track prompt evolution and ontology growth at checkpoints
- No restarts - learning persists throughout

Usage:
    uv run python run_mipro_continual.py
    uv run python run_mipro_continual.py --rollouts-per-split 100 --model gpt-4.1-nano
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Add synth-ai to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_splits import (
    Banking77SplitDataset,
    format_available_intents,
    get_split_intents,
    get_split_size,
)
from synth_ai.core.utils.env import mint_demo_api_key
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.localapi._impl import run_server_background
from synth_ai.core.tunnels import PortConflictBehavior, acquire_port
from synth_ai.core.utils.urls import BACKEND_URL_BASE


# Tool schema (same as classic GEPA)
TOOL_NAME = "banking77_classify"
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Return the predicted banking77 intent label.",
        "parameters": {
            "type": "object",
            "properties": {"intent": {"type": "string"}},
            "required": ["intent"],
        },
    },
}

# Baseline system prompt
SYSTEM_PROMPT = (
    "You are an expert banking assistant that classifies customer queries into banking intents. "
    "Given a customer message, respond with exactly one intent label from the provided list "
    "using the `banking77_classify` tool."
)


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
    """Build the initial prompt template."""
    user_prompt = (
        "Customer Query: {query}\n\n"
        "Available Intents:\n{available_intents}\n\n"
        "Classify this query into one of the above banking intents using the tool call."
    )
    return {
        "id": "banking77_continual_pattern",
        "name": "Banking77 Continual Classification",
        "messages": [
            {"role": "system", "order": 0, "pattern": SYSTEM_PROMPT},
            {"role": "user", "order": 1, "pattern": user_prompt},
        ],
        "wildcards": {"query": "REQUIRED", "available_intents": "OPTIONAL"},
    }


def build_mipro_continual_config(
    *,
    task_app_url: str,
    train_seeds: List[int],
    val_seeds: List[int],
    min_rollouts_before_proposal: int = 20,
) -> Dict[str, Any]:
    """Build MIPRO online config for continual learning."""
    policy_model = os.environ.get("BANKING77_POLICY_MODEL", "gpt-4.1-nano")
    policy_provider = os.environ.get("BANKING77_POLICY_PROVIDER", "openai")
    proposer_model = os.environ.get("BANKING77_PROPOSER_MODEL", "gpt-4.1-mini")
    proposer_provider = os.environ.get("BANKING77_PROPOSER_PROVIDER", "openai")
    
    return {
        "prompt_learning": {
            "algorithm": "mipro",
            "task_app_id": "banking77_continual",
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
                "online_rollouts_per_candidate": 10,
                "proposer": {
                    "mode": "instruction_only",
                    "model": proposer_model,
                    "provider": proposer_provider,
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "generate_at_iterations": [0, 1, 2, 3],
                    "instructions_per_batch": 2,
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


def extract_ontology(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract ontology information from system state."""
    ontology = {}
    
    # Get all candidates and their properties
    candidates = state.get("candidates", {})
    if isinstance(candidates, dict):
        ontology["num_candidates"] = len(candidates)
        ontology["candidates"] = {}
        for cid, candidate in candidates.items():
            if isinstance(candidate, dict):
                ontology["candidates"][cid] = {
                    "rollout_count": candidate.get("rollout_count", 0),
                    "avg_reward": candidate.get("avg_reward"),
                    "parent_id": candidate.get("parent_id"),
                }
    
    # Get other state info
    ontology["version"] = state.get("version", 0)
    ontology["best_candidate_id"] = state.get("best_candidate_id")
    ontology["rollout_count"] = state.get("rollout_count", 0)
    ontology["reward_count"] = state.get("reward_count", 0)
    ontology["proposal_seq"] = state.get("proposal_seq", 0)
    
    return ontology


def run_rollout(
    *,
    task_app_url: str,
    env_key: str,
    seed: int,
    inference_url: str,
    model: str,
    rollout_id: str,
    intent_split: int,
) -> float:
    """Execute a single rollout for online MIPRO."""
    payload = {
        "trace_correlation_id": rollout_id,
        "env": {
            "seed": seed,
            "config": {
                "seed": seed,
                "split": "train",
                "intent_split": intent_split,
            },
        },
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
        candidate_id = "unknown"

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


def new_rollout_id(split_num: int, seed: int) -> str:
    """Generate a unique rollout ID."""
    return f"trace_split{split_num}_rollout_{seed}_{uuid.uuid4().hex[:6]}"


def create_continual_task_app(env_key: str, backend_url: str, api_key: str):
    """Create task app that can handle multiple intent splits."""
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import Optional, Any as AnyType
    
    from synth_ai.sdk.clients import AsyncOpenAI as SynthAsyncOpenAI
    
    app = FastAPI(title="Banking77 Continual Learning Task App")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    dataset = Banking77SplitDataset()
    dataset.ensure_ready(["train", "test"])
    
    class RolloutRequest(BaseModel):
        trace_correlation_id: str
        env: dict
        policy: dict
    
    async def classify_query(
        query: str,
        system_prompt: str,
        available_intents: str,
        model: str,
        policy_api_key: str,
        inference_url: str,
    ) -> str:
        """Classify a banking query."""
        user_msg = (
            f"Customer Query: {query}\n\n"
            f"Available Intents:\n{available_intents}\n\n"
            f"Classify this query into one of the above banking intents using the tool call."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        
        default_headers = {"X-API-Key": policy_api_key}
        client = SynthAsyncOpenAI(
            trial_id="banking77_continual",
            correlation_id=f"corr_{uuid.uuid4().hex[:12]}",
            base_url=inference_url,
            api_key="synth-interceptor",
            default_headers=default_headers,
        )
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
        )
        
        tool_call = response.choices[0].message.tool_calls[0]
        args_raw = tool_call.function.arguments
        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        return args["intent"]
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "task_app": "banking77_continual"}
    
    @app.post("/rollout")
    async def rollout(request: RolloutRequest, fastapi_request: Request):
        env_config = request.env.get("config", {})
        data_split = env_config.get("split", "train")
        seed = request.env.get("seed", 0)
        
        # Get intent split from config (default to 4 = all intents)
        intent_split = env_config.get("intent_split", 4)
        
        sample = dataset.sample(data_split=data_split, intent_split=intent_split, index=seed)
        
        policy_config = request.policy.get("config", {})
        inference_url = policy_config.get("inference_url") or f"{backend_url}/v1"
        policy_api_key = policy_config.get("api_key") or api_key
        
        prompt_override = (
            policy_config.get("system_prompt")
            or policy_config.get("instruction")
            or policy_config.get("prompt")
        )
        active_system_prompt = prompt_override or SYSTEM_PROMPT
        
        # Get available intents for this split
        split_labels = dataset.get_split_labels(intent_split)
        available_intents = format_available_intents(split_labels)
        
        start = time.perf_counter()
        predicted_intent = await classify_query(
            query=sample["text"],
            system_prompt=active_system_prompt,
            available_intents=available_intents,
            model=policy_config.get("model", "gpt-4.1-nano"),
            policy_api_key=policy_api_key,
            inference_url=inference_url,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        
        expected_intent = sample["label"]
        is_correct = (
            predicted_intent.lower().replace("_", " ").strip()
            == expected_intent.lower().replace("_", " ").strip()
        )
        reward = 1.0 if is_correct else 0.0
        
        return {
            "reward_info": {
                "outcome_reward": reward,
                "outcome_objectives": {"reward": reward, "latency_ms": latency_ms},
            },
            "trace_correlation_id": request.trace_correlation_id,
            "metadata": {
                "intent_split": intent_split,
                "expected_intent": expected_intent,
                "predicted_intent": predicted_intent,
            },
        }
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Run MIPRO Continual Learning on Banking77")
    parser.add_argument("--backend-url", default=None, help="Backend URL")
    parser.add_argument("--local-host", default="localhost", help="Local API hostname")
    parser.add_argument("--local-port", type=int, default=8030, help="Local API port")
    parser.add_argument("--rollouts-per-split", type=int, default=100, help="Rollouts per split")
    parser.add_argument("--model", default="gpt-4.1-nano", help="Model to use")
    parser.add_argument("--train-size", type=int, default=30, help="Training seeds count")
    parser.add_argument("--val-size", type=int, default=20, help="Validation seeds count")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()
    
    # Resolve backend
    backend_url = (args.backend_url or resolve_backend_url()).rstrip("/")
    print(f"Backend URL: {backend_url}")
    
    # Get API key
    api_key = os.environ.get("SYNTH_API_KEY", "").strip()
    if not api_key:
        print("Minting demo API key...")
        api_key = mint_demo_api_key(backend_url=backend_url)
    print(f"API Key: {api_key[:20]}...")
    
    # Check backend health
    try:
        r = httpx.get(f"{backend_url}/health", timeout=30)
        if r.status_code != 200:
            print(f"WARNING: Backend health check returned status {r.status_code}")
        else:
            print("Backend health: OK")
    except Exception as e:
        print(f"WARNING: Backend health check failed: {e}")
    
    # Set up environment key
    env_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
    if not env_key:
        env_key = ensure_localapi_auth(backend_base=None, upload=False, persist=False)
    os.environ["ENVIRONMENT_API_KEY"] = env_key
    print(f"Environment key: {env_key[:12]}...{env_key[-4:]}")
    
    # Set model
    if args.model:
        os.environ["BANKING77_POLICY_MODEL"] = args.model
    
    # Start local task app
    port = acquire_port(args.local_port, on_conflict=PortConflictBehavior.FIND_NEW)
    if port != args.local_port:
        print(f"Port {args.local_port} in use, using port {port} instead")
    
    app = create_continual_task_app(env_key, backend_url, api_key)
    run_server_background(app, port)
    print(f"Waiting for local API on port {port}...")
    wait_for_health_check_sync("localhost", port, env_key, timeout=30.0)
    task_app_url = f"http://{args.local_host}:{port}"
    print(f"Local API URL: {task_app_url}")
    
    # Build seeds
    train_seeds = list(range(args.train_size))
    val_seeds = list(range(args.train_size, args.train_size + args.val_size))
    
    print("\n" + "="*60)
    print("MIPRO Continual Learning Configuration")
    print("="*60)
    print(f"  Rollouts per split: {args.rollouts_per_split}")
    print(f"  Total rollouts: {args.rollouts_per_split * 4}")
    print(f"  Splits: 1 (2 intents) → 2 (7) → 3 (27) → 4 (77)")
    print("="*60)
    
    # Build config and create job
    config_body = build_mipro_continual_config(
        task_app_url=task_app_url,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        min_rollouts_before_proposal=20,
    )
    
    print("\nCreating MIPRO online job...")
    start_time = time.time()
    job_id = create_job(backend_url, api_key, config_body)
    print(f"Job ID: {job_id}")
    
    # Get job details
    detail = get_job_detail(backend_url, api_key, job_id, include_metadata=True)
    metadata = detail.get("metadata", {})
    system_id = metadata.get("mipro_system_id")
    proxy_url = metadata.get("mipro_proxy_url")
    
    if not system_id or not proxy_url:
        raise RuntimeError(f"Missing mipro_system_id or mipro_proxy_url in metadata: {metadata}")
    
    print(f"System ID: {system_id}")
    print(f"Proxy URL: {proxy_url}")
    
    # Results tracking
    all_results = {
        "method": "mipro_continual",
        "job_id": job_id,
        "system_id": system_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "backend_url": backend_url,
            "model": args.model,
            "rollouts_per_split": args.rollouts_per_split,
            "train_size": args.train_size,
            "val_size": args.val_size,
        },
        "checkpoints": [],
        "split_results": {},
    }
    
    model = os.environ.get("BANKING77_MODEL", args.model)
    total_rollouts = 0
    total_correct = 0
    
    # Run through all splits sequentially
    for split_num in [1, 2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"SPLIT {split_num}: {get_split_size(split_num)} intents")
        print(f"{'='*60}")
        
        split_start = time.time()
        split_correct = 0
        split_rollouts = 0
        candidate_stats: Dict[str, Dict[str, Any]] = {}
        
        # Get dataset size for this split
        dataset = Banking77SplitDataset()
        dataset.ensure_ready(["train"])
        split_train_size = dataset.size("train", split_num)
        
        for i in range(args.rollouts_per_split):
            # Cycle through seeds within the split's data
            seed = i % split_train_size
            rollout_id = new_rollout_id(split_num, i)
            inference_url = f"{proxy_url}/{rollout_id}/chat/completions"
            
            try:
                reward = run_rollout(
                    task_app_url=task_app_url,
                    env_key=env_key,
                    seed=seed,
                    inference_url=inference_url,
                    model=model,
                    rollout_id=rollout_id,
                    intent_split=split_num,
                )
                
                if reward > 0:
                    split_correct += 1
                    total_correct += 1
                split_rollouts += 1
                total_rollouts += 1
                
                # Push status to backend
                candidate_id = push_status(
                    backend_url=backend_url,
                    api_key=api_key,
                    system_id=system_id,
                    rollout_id=rollout_id,
                    reward=reward,
                )
                
                # Track candidate statistics
                if candidate_id:
                    if candidate_id not in candidate_stats:
                        candidate_stats[candidate_id] = {"count": 0, "total_reward": 0.0}
                    candidate_stats[candidate_id]["count"] += 1
                    candidate_stats[candidate_id]["total_reward"] += reward
                
                # Progress update every 20 rollouts
                if (i + 1) % 20 == 0:
                    running_acc = split_correct / split_rollouts if split_rollouts > 0 else 0
                    print(
                        f"  Progress: {i+1}/{args.rollouts_per_split} | "
                        f"Split accuracy: {running_acc:.1%} | "
                        f"Candidate: {candidate_id}"
                    )
                    
            except Exception as e:
                print(f"  Error on rollout {i}: {e}")
        
        split_elapsed = time.time() - split_start
        split_accuracy = split_correct / split_rollouts if split_rollouts > 0 else 0
        
        # Get checkpoint state
        state = get_system_state(backend_url, api_key, system_id)
        best_candidate_id = state.get("best_candidate_id")
        best_candidate_text = extract_candidate_text(state, best_candidate_id)
        ontology = extract_ontology(state)
        
        # Record checkpoint
        checkpoint = {
            "split": split_num,
            "num_intents": get_split_size(split_num),
            "total_rollouts_so_far": total_rollouts,
            "split_accuracy": split_accuracy,
            "cumulative_accuracy": total_correct / total_rollouts if total_rollouts > 0 else 0,
            "elapsed_seconds": split_elapsed,
            "ontology": ontology,
            "best_candidate_id": best_candidate_id,
            "best_candidate_text": best_candidate_text[:500] + "..." if best_candidate_text and len(best_candidate_text) > 500 else best_candidate_text,
            "candidate_stats": candidate_stats,
        }
        all_results["checkpoints"].append(checkpoint)
        all_results["split_results"][str(split_num)] = {
            "accuracy": split_accuracy,
            "correct": split_correct,
            "total": split_rollouts,
            "elapsed_seconds": split_elapsed,
        }
        
        # Print split summary
        print(f"\n  Split {split_num} Summary:")
        print(f"    Accuracy: {split_accuracy:.1%} ({split_correct}/{split_rollouts})")
        print(f"    Time: {split_elapsed:.1f}s")
        print(f"    Ontology: {ontology['num_candidates']} candidates, {ontology['proposal_seq']} proposals")
        print(f"    Best candidate: {best_candidate_id}")
        if best_candidate_text:
            preview = best_candidate_text[:200] + "..." if len(best_candidate_text) > 200 else best_candidate_text
            print(f"    Best prompt: {preview}")
    
    total_elapsed = time.time() - start_time
    all_results["total_elapsed_seconds"] = total_elapsed
    all_results["final_accuracy"] = total_correct / total_rollouts if total_rollouts > 0 else 0
    
    # Print final summary
    print("\n" + "="*70)
    print("MIPRO CONTINUAL LEARNING - FINAL RESULTS")
    print("="*70)
    print(f"{'Split':<15} {'Intents':<10} {'Accuracy':<15} {'Time (s)':<10}")
    print("-"*70)
    for split_num in [1, 2, 3, 4]:
        sr = all_results["split_results"].get(str(split_num), {})
        intents = get_split_size(split_num)
        accuracy = sr.get("accuracy", 0)
        elapsed = sr.get("elapsed_seconds", 0)
        print(f"Split {split_num:<9} {intents:<10} {accuracy:.1%}{'':<9} {elapsed:.1f}")
    print("-"*70)
    print(f"Total rollouts: {total_rollouts}")
    print(f"Overall accuracy: {all_results['final_accuracy']:.1%}")
    print(f"Total time: {total_elapsed:.1f}s")
    print("="*70)
    
    # Get final ontology state
    final_state = get_system_state(backend_url, api_key, system_id)
    final_ontology = extract_ontology(final_state)
    print("\nFinal Ontology:")
    print(f"  Total candidates: {final_ontology['num_candidates']}")
    print(f"  Total proposals: {final_ontology['proposal_seq']}")
    print(f"  Best candidate: {final_ontology['best_candidate_id']}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"mipro_continual_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
