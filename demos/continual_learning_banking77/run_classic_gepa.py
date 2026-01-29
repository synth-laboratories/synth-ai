#!/usr/bin/env python3
"""
Run Classic GEPA (Non-Continual) on Banking77 Progressive Splits

This script demonstrates the classic approach:
- Run GEPA on Split 1, save best prompt
- Run GEPA on Split 2 with warm start (prev best) AND cold start (baseline)
- Repeat for Splits 3 and 4

Usage:
    uv run python run_classic_gepa.py
    uv run python run_classic_gepa.py --rollouts 100 --model gpt-4.1-nano
    uv run python run_classic_gepa.py --split 2  # Run only split 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Add synth-ai to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset
from fastapi import Request

from data_splits import (
    Banking77SplitDataset,
    format_available_intents,
    get_split_intents,
    get_split_size,
)
from synth_ai.core.utils.env import mint_demo_api_key
from synth_ai.data.enums import SuccessStatus
from synth_ai.sdk.optimization.internal.prompt_learning import PromptLearningJob
from synth_ai.sdk.optimization.models import PromptLearningResult
from synth_ai.sdk.optimization.internal.learning.prompt_learning_client import PromptLearningClient
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.localapi._impl.server import run_server_background
from synth_ai.sdk.localapi._impl.contracts import RolloutMetrics, RolloutRequest, RolloutResponse, TaskInfo
from synth_ai.sdk.localapi._impl.trace_correlation_helpers import extract_trace_correlation_id
from synth_ai.core.tunnels import (
    PortConflictBehavior,
    TunnelBackend,
    TunneledLocalAPI,
    acquire_port,
    cleanup_all,
)
from synth_ai.sdk.clients import AsyncOpenAI as SynthAsyncOpenAI


# Tool schema
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


def resolve_backend_url() -> str:
    """Resolve the backend URL."""
    for env_var in ("SYNTH_URL", "SYNTH_BACKEND_URL", "RUST_BACKEND_URL"):
        env_url = (os.environ.get(env_var) or "").strip()
        if env_url:
            return env_url.rstrip("/")
    return "https://api.usesynth.ai"


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


async def classify_banking77_query(
    query: str,
    system_prompt: str,
    available_intents: str,
    model: str,
    api_key: str,
    inference_url: str,
) -> str:
    """Classify a banking query using the LLM."""
    user_msg = (
        f"Customer Query: {query}\n\n"
        f"Available Intents:\n{available_intents}\n\n"
        f"Classify this query into one of the above banking intents using the tool call."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    default_headers = {"X-API-Key": api_key}
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


def create_split_local_api(
    system_prompt: str,
    intent_split: int,
    api_key: str,
    backend_url: str,
):
    """Create a local API for a specific intent split."""
    dataset = Banking77SplitDataset()
    dataset.ensure_ready(["train", "test"])
    
    async def run_rollout(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
        data_split = request.env.config.get("split", "train")
        seed = request.env.seed
        
        sample = dataset.sample(data_split=data_split, intent_split=intent_split, index=seed)
        
        policy_config = request.policy.config or {}
        inference_url = policy_config.get("inference_url") or f"{backend_url}/v1"
        policy_api_key = policy_config.get("api_key") or api_key
        
        prompt_override = (
            policy_config.get("system_prompt")
            or policy_config.get("instruction")
            or policy_config.get("prompt")
        )
        active_system_prompt = prompt_override or system_prompt
        
        # Get available intents for this split
        split_labels = dataset.get_split_labels(intent_split)
        available_intents = format_available_intents(split_labels)
        
        start = time.perf_counter()
        predicted_intent = await classify_banking77_query(
            query=sample["text"],
            system_prompt=active_system_prompt,
            available_intents=available_intents,
            model=policy_config.get("model", "gpt-4.1-nano"),
            api_key=policy_api_key,
            inference_url=inference_url,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        
        expected_intent = sample["label"]
        is_correct = (
            predicted_intent.lower().replace("_", " ").strip()
            == expected_intent.lower().replace("_", " ").strip()
        )
        reward = 1.0 if is_correct else 0.0
        
        trace_correlation_id = request.trace_correlation_id
        
        return RolloutResponse(
            reward_info=RolloutMetrics(
                outcome_reward=reward,
                outcome_objectives={"reward": reward, "latency_ms": latency_ms},
                instance_objectives=[{"reward": reward, "latency_ms": latency_ms}],
                details={"latency_ms": latency_ms, "intent_split": intent_split},
            ),
            trace=None,
            trace_correlation_id=trace_correlation_id,
            inference_url=str(inference_url or ""),
            success_status=SuccessStatus.SUCCESS,
        )
    
    def provide_taskset_description():
        return {
            "splits": ["train", "test"],
            "sizes": {
                "train": dataset.size("train", intent_split),
                "test": dataset.size("test", intent_split),
            },
            "intent_split": intent_split,
            "num_intents": get_split_size(intent_split),
        }
    
    def provide_task_instances(seeds):
        for seed in seeds:
            sample = dataset.sample(data_split="train", intent_split=intent_split, index=seed)
            yield TaskInfo(
                task={"id": f"banking77_split{intent_split}", "name": f"Banking77 Split {intent_split}"},
                dataset={"id": f"banking77_split{intent_split}", "split": sample["split"], "index": sample["index"]},
                inference={"tool": TOOL_NAME},
                limits={"max_turns": 1},
                task_metadata={
                    "query": sample["text"],
                    "expected_intent": sample["label"],
                    "intent_split": intent_split,
                },
            )
    
    return create_local_api(
        LocalAPIConfig(
            app_id=f"banking77_split{intent_split}",
            name=f"Banking77 Split {intent_split} ({get_split_size(intent_split)} intents)",
            description=f"Banking77 with {get_split_size(intent_split)} intents for continual learning",
            provide_taskset_description=provide_taskset_description,
            provide_task_instances=provide_task_instances,
            rollout=run_rollout,
            cors_origins=["*"],
        )
    )


def extract_system_prompt_from_result(prompt_results, fallback: str) -> str:
    """Extract the system prompt from GEPA results."""
    if prompt_results.top_prompts:
        top = prompt_results.top_prompts[0]
        if isinstance(top, dict):
            if "full_text" in top and top["full_text"]:
                return top["full_text"]
            if "template" in top and top["template"]:
                template = top["template"]
                if "sections" in template:
                    for section in template["sections"]:
                        if section.get("role") == "system":
                            return section.get("content", fallback)
    
    if prompt_results.best_prompt:
        if isinstance(prompt_results.best_prompt, str):
            return prompt_results.best_prompt
        elif isinstance(prompt_results.best_prompt, dict):
            if "messages" in prompt_results.best_prompt:
                for msg in prompt_results.best_prompt["messages"]:
                    if msg.get("role") == "system":
                        return msg.get("content") or msg.get("pattern", fallback)
    
    return fallback


async def evaluate_prompt_accuracy(
    *,
    prompt: str,
    intent_split: int,
    api_key: str,
    backend_url: str,
    model: str,
    num_samples: int = 50,
) -> float:
    """Evaluate a prompt's accuracy on a validation set."""
    import httpx
    import random
    
    dataset = Banking77SplitDataset()
    dataset.ensure_ready(["train", "test"])
    
    # Get test samples for this split
    split_labels = dataset.get_split_labels(intent_split)
    available_intents = format_available_intents(split_labels)
    
    # Sample random test indices
    test_size = dataset.size("test", intent_split)
    sample_indices = random.sample(range(test_size), min(num_samples, test_size))
    
    correct = 0
    total = 0
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        for idx in sample_indices:
            sample = dataset.sample(data_split="test", intent_split=intent_split, index=idx)
            
            user_msg = (
                f"Customer Query: {sample['text']}\n\n"
                f"Available Intents:\n{available_intents}\n\n"
                f"Classify this query into one of the above banking intents using the tool call."
            )
            
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg},
            ]
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            
            payload = {
                "model": model,
                "messages": messages,
                "tools": [TOOL_SCHEMA],
                "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
            }
            
            try:
                response = await client.post(
                    f"{backend_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                
                if response.status_code != 200:
                    continue
                
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    continue
                    
                tool_calls = choices[0].get("message", {}).get("tool_calls", [])
                if not tool_calls:
                    continue
                    
                args_raw = tool_calls[0].get("function", {}).get("arguments")
                if not args_raw:
                    continue
                    
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                predicted_intent = args.get("intent", "")
                
                expected_intent = sample["label"]
                is_correct = (
                    predicted_intent.lower().replace("_", " ").strip()
                    == expected_intent.lower().replace("_", " ").strip()
                )
                
                if is_correct:
                    correct += 1
                total += 1
                
            except Exception as e:
                print(f"  Eval error: {e}")
                continue
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"  Evaluation: {correct}/{total} = {accuracy:.1%}")
    return accuracy


async def run_gepa_on_split(
    *,
    intent_split: int,
    initial_prompt: str,
    backend_url: str,
    api_key: str,
    env_key: str,
    model: str,
    rollouts: int,
    train_size: int,
    local_port: int,
    label: str,
) -> Dict[str, Any]:
    """Run GEPA optimization on a single intent split."""
    print(f"\n{'='*60}")
    print(f"Running GEPA on Split {intent_split} ({get_split_size(intent_split)} intents) - {label}")
    print(f"{'='*60}")
    
    # Create and start local API
    app = create_split_local_api(
        system_prompt=initial_prompt,
        intent_split=intent_split,
        api_key=api_key,
        backend_url=backend_url,
    )
    
    port = acquire_port(local_port, on_conflict=PortConflictBehavior.FIND_NEW)
    run_server_background(app, port)
    wait_for_health_check_sync("localhost", port, env_key, timeout=30.0)
    print(f"Local API ready on port {port}")
    
    # Create tunnel
    tunnel = await TunneledLocalAPI.create(
        local_port=port,
        backend=TunnelBackend.CloudflareManagedLease,
        api_key=api_key,
        env_api_key=env_key,
        backend_url=backend_url,
        progress=True,
    )
    local_api_url = tunnel.url
    print(f"Tunnel URL: {local_api_url}")
    
    # Build user prompt template
    user_prompt = (
        "Customer Query: {query}\n\n"
        "Available Intents:\n{available_intents}\n\n"
        "Classify this query into one of the above banking intents using the tool call."
    )
    
    # Configure GEPA job
    # Need at least pareto_set_size + 3 (for feedback) seeds
    effective_train_size = min(train_size, 50)
    train_seeds = list(range(effective_train_size))
    val_seeds = list(range(50, 70))
    
    # pareto_set_size must be <= train_size - 3 (to leave room for feedback seeds)
    max_pareto_size = max(1, effective_train_size - 3)
    pareto_size = min(15, max_pareto_size)
    
    config_body = {
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_id": f"banking77_split{intent_split}",
            "task_app_url": local_api_url,
            "initial_prompt": {
                "id": f"banking77_split{intent_split}_pattern",
                "name": f"Banking77 Split {intent_split} Classification",
                "messages": [
                    {"role": "system", "order": 0, "pattern": initial_prompt},
                    {"role": "user", "order": 1, "pattern": user_prompt},
                ],
                "wildcards": {"query": "REQUIRED", "available_intents": "OPTIONAL"},
            },
            "policy": {
                "model": model,
                "provider": "openai",
                "inference_mode": "synth_hosted",
                "temperature": 0.0,
                "max_completion_tokens": 256,
            },
            "env_config": {"split": "train"},
            "gepa": {
                "env_name": f"banking77_split{intent_split}",
                "evaluation": {
                    "seeds": train_seeds,
                    "validation_seeds": val_seeds,
                },
                "rollout": {
                    "budget": rollouts,
                    "max_concurrent": 50,
                    "minibatch_size": 5,
                },
                "mutation": {"rate": 0.3},
                "population": {
                    "initial_size": 3,
                    "num_generations": 3,
                    "children_per_generation": 2,
                },
                "archive": {"pareto_set_size": pareto_size},
                "token": {"counting_model": "gpt-4"},
            },
        },
    }
    
    # Submit and run GEPA job
    print(f"Creating GEPA job for {label}...")
    pl_job = PromptLearningJob.from_dict(
        config_dict=deepcopy(config_body),
        backend_url=backend_url,
        api_key=api_key,
    )
    
    start_time = time.time()
    job_id = pl_job.submit()
    print(f"Job ID: {job_id}")
    
    # Stream until complete
    try:
        gepa_result = await pl_job.stream_until_complete_async(
            timeout=1800.0,
            interval=10.0,
        )
    except ValueError as exc:
        if "stream ended without terminal event" not in str(exc):
            raise
        print("Stream ended early; falling back to polling job status...")
        status_payload = await pl_job.get_status_async()
        try:
            results_payload = await pl_job.get_results_async()
        except Exception:
            results_payload = {}
        if isinstance(results_payload, dict):
            status_payload.update(
                {
                    "best_prompt": results_payload.get("best_prompt"),
                    "best_score": results_payload.get("best_score"),
                }
            )
        gepa_result = PromptLearningResult.from_response(job_id, status_payload)
    
    elapsed = time.time() - start_time
    print(f"\nJob completed in {elapsed:.1f}s - Status: {gepa_result.status.value}")
    
    result = {
        "split": intent_split,
        "label": label,
        "num_intents": get_split_size(intent_split),
        "job_id": job_id,
        "status": gepa_result.status.value,
        "elapsed_seconds": elapsed,
        "initial_prompt": initial_prompt,
    }
    
    if gepa_result.succeeded:
        # Extract best prompt
        pl_client = PromptLearningClient(backend_url, api_key)
        prompt_results = await pl_client.get_prompts(job_id)
        
        best_prompt = extract_system_prompt_from_result(prompt_results, initial_prompt)
        
        # Run post-hoc evaluation to get accuracy
        print("Running post-hoc evaluation on validation set...")
        best_reward = await evaluate_prompt_accuracy(
            prompt=best_prompt,
            intent_split=intent_split,
            api_key=api_key,
            backend_url=backend_url,
            model=model,
            num_samples=50,  # Evaluate on 50 samples
        )
        
        result["best_prompt"] = best_prompt
        result["best_reward"] = best_reward
        result["succeeded"] = True
        
        print(f"Best reward: {best_reward:.1%}" if best_reward else "Best reward: N/A")
        print(f"Best prompt length: {len(best_prompt)} chars")
    else:
        result["succeeded"] = False
        result["error"] = str(gepa_result.error)
        print(f"Error: {gepa_result.error}")
    
    # Cleanup tunnel
    tunnel.close()
    
    return result


async def main():
    parser = argparse.ArgumentParser(description="Run Classic GEPA on Banking77 Splits")
    parser.add_argument("--backend-url", default=None, help="Backend URL")
    parser.add_argument("--model", default="gpt-4.1-nano", help="Model to use")
    parser.add_argument("--rollouts", type=int, default=100, help="Rollouts per split")
    parser.add_argument("--train-size", type=int, default=30, help="Training seeds per split")
    parser.add_argument("--split", type=int, default=None, help="Run only this split (1-4)")
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
    
    # Get environment key
    env_key = ensure_localapi_auth(
        backend_base=backend_url,
        synth_api_key=api_key,
    )
    print(f"Environment key: {env_key[:12]}...{env_key[-4:]}")
    
    # Baseline prompt
    BASELINE_PROMPT = (
        "You are an expert banking assistant that classifies customer queries into banking intents. "
        "Given a customer message, respond with exactly one intent label from the provided list "
        "using the `banking77_classify` tool."
    )
    
    # Determine which splits to run
    if args.split:
        splits_to_run = [args.split]
    else:
        splits_to_run = [1, 2, 3, 4]
    
    all_results = {
        "method": "classic_gepa",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "backend_url": backend_url,
            "model": args.model,
            "rollouts_per_split": args.rollouts,
            "train_size": args.train_size,
        },
        "splits": {},
    }
    
    # Track prompts for warm starts
    best_prompts = {0: BASELINE_PROMPT}  # Split 0 = baseline
    
    try:
        for split_num in splits_to_run:
            split_results = {"cold_start": None, "warm_start": None}
            
            # Cold start (baseline prompt)
            cold_result = await run_gepa_on_split(
                intent_split=split_num,
                initial_prompt=BASELINE_PROMPT,
                backend_url=backend_url,
                api_key=api_key,
                env_key=env_key,
                model=args.model,
                rollouts=args.rollouts,
                train_size=args.train_size,
                local_port=8020 + split_num * 2,
                label=f"Split {split_num} Cold Start",
            )
            split_results["cold_start"] = cold_result
            
            # Warm start (if we have a previous best prompt and this isn't split 1)
            if split_num > 1:
                prev_best = best_prompts.get(split_num - 1, BASELINE_PROMPT)
                warm_result = await run_gepa_on_split(
                    intent_split=split_num,
                    initial_prompt=prev_best,
                    backend_url=backend_url,
                    api_key=api_key,
                    env_key=env_key,
                    model=args.model,
                    rollouts=args.rollouts,
                    train_size=args.train_size,
                    local_port=8021 + split_num * 2,
                    label=f"Split {split_num} Warm Start",
                )
                split_results["warm_start"] = warm_result
            
            # Track best prompt for next split
            # Use cold start result if no warm start, otherwise compare
            if split_results["warm_start"] and split_results["warm_start"].get("succeeded"):
                warm_reward = split_results["warm_start"].get("best_reward", 0) or 0
                cold_reward = split_results["cold_start"].get("best_reward", 0) or 0
                if warm_reward >= cold_reward:
                    best_prompts[split_num] = split_results["warm_start"].get("best_prompt", BASELINE_PROMPT)
                else:
                    best_prompts[split_num] = split_results["cold_start"].get("best_prompt", BASELINE_PROMPT)
            elif split_results["cold_start"].get("succeeded"):
                best_prompts[split_num] = split_results["cold_start"].get("best_prompt", BASELINE_PROMPT)
            else:
                best_prompts[split_num] = best_prompts.get(split_num - 1, BASELINE_PROMPT)
            
            all_results["splits"][str(split_num)] = split_results
            
            # Print summary for this split
            print(f"\n{'='*60}")
            print(f"SPLIT {split_num} SUMMARY ({get_split_size(split_num)} intents)")
            print(f"{'='*60}")
            cold_reward = split_results["cold_start"].get("best_reward")
            print(f"  Cold Start: {cold_reward:.1%}" if cold_reward else "  Cold Start: N/A")
            if split_results["warm_start"]:
                warm_reward = split_results["warm_start"].get("best_reward")
                print(f"  Warm Start: {warm_reward:.1%}" if warm_reward else "  Warm Start: N/A")
                if cold_reward and warm_reward:
                    diff = warm_reward - cold_reward
                    print(f"  Difference: {diff:+.1%} ({'warm' if diff > 0 else 'cold'} is better)")
    
    finally:
        cleanup_all()
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"classic_gepa_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Print final summary table
    print("\n" + "="*70)
    print("FINAL RESULTS - Classic GEPA (Non-Continual)")
    print("="*70)
    print(f"{'Split':<15} {'Intents':<10} {'Cold Start':<15} {'Warm Start':<15}")
    print("-"*70)
    for split_num in splits_to_run:
        sr = all_results["splits"].get(str(split_num), {})
        intents = get_split_size(split_num)
        cold = sr.get("cold_start", {}).get("best_reward")
        warm = sr.get("warm_start", {}).get("best_reward") if sr.get("warm_start") else None
        cold_str = f"{cold:.1%}" if cold else "N/A"
        warm_str = f"{warm:.1%}" if warm else "-"
        print(f"Split {split_num:<9} {intents:<10} {cold_str:<15} {warm_str:<15}")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
