#!/usr/bin/env python3
"""Run HoVer GEPA benchmark following banking77 demo standards."""

import asyncio
import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from datasets import load_dataset
from openai import AsyncOpenAI

# Add synth-ai to path
sys.path.insert(0, "/Users/joshpurtell/Documents/GitHub/synth-ai")

from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient
from synth_ai.sdk.learning.rl import mint_environment_api_key, setup_environment_api_key
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.task import run_server_background
from synth_ai.sdk.task.contracts import RolloutMetrics, RolloutRequest, RolloutResponse, TaskInfo
from synth_ai.sdk.tunnels import TunnelBackend, TunneledLocalAPI, cleanup_all, kill_port, wait_for_health_check
from synth_ai.core.env import mint_demo_api_key

# Configuration
BACKEND_URL = os.environ.get("SYNTH_BACKEND_URL", "http://localhost:8000")
LOCAL_API_PORT = 8003

# HoVer config
APP_ID = "hover"
APP_NAME = "HoVer Claim Verification"

LABEL_MAP = {0: "SUPPORTED", 1: "REFUTED"}
REVERSE_LABEL_MAP = {"SUPPORTED": 0, "REFUTED": 1, "supported": 0, "refuted": 1}

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "verify_claim",
        "description": "Verify whether the claim is supported or refuted by the evidence",
        "parameters": {
            "type": "object",
            "properties": {"verdict": {"type": "string", "enum": ["SUPPORTED", "REFUTED"], "description": "The verdict"}},
            "required": ["verdict"],
        },
    },
}

BASELINE_SYSTEM_PROMPT = """You are a fact verification assistant. Your task is to determine whether a claim is SUPPORTED or REFUTED by the given evidence.

Analyze the evidence carefully and determine if it supports or refutes the claim.
- SUPPORTED: The evidence confirms the claim is true
- REFUTED: The evidence shows the claim is false

Give only the verdict, no explanation."""

USER_PROMPT = """Evidence:
{evidence}

Claim: {claim}

Based on the evidence above, is this claim SUPPORTED or REFUTED?"""

# Benchmark config
MODELS = ["gpt-4.1-nano", "gpt-5-nano", "gpt-4o-mini"]
RUNS_PER_MODEL = 3

# Use 150 for training, 500 for holdout validation
TRAINING_SEEDS = list(range(150))
VALIDATION_SEEDS = list(range(150, 650))
ROLLOUT_BUDGET = 500


class HoVerDataset:
    def __init__(self):
        self._cache = None

    def _load(self):
        if self._cache is None:
            self._cache = load_dataset("Dzeniks/hover", split="test")
        return self._cache

    def ensure_ready(self):
        self._load()

    def size(self) -> int:
        return len(self._load())

    def sample(self, index: int) -> dict:
        ds = self._load()
        idx = index % len(ds)
        row = ds[idx]
        label_idx = int(row.get("label", 0))
        return {
            "index": idx,
            "claim": row["claim"],
            "evidence": row["evidence"].strip(),
            "label": LABEL_MAP.get(label_idx, "SUPPORTED"),
            "label_idx": label_idx,
        }


def normalize_verdict(verdict: str) -> str:
    """Normalize verdict for comparison."""
    verdict = verdict.strip().upper()
    if "SUPPORT" in verdict:
        return "SUPPORTED"
    if "REFUT" in verdict:
        return "REFUTED"
    return verdict


def verdicts_match(predicted: str, expected: str) -> bool:
    """Check if predicted verdict matches expected."""
    return normalize_verdict(predicted) == normalize_verdict(expected)


async def verify_claim(
    claim: str,
    evidence: str,
    system_prompt: str,
    model: str,
    api_key: Optional[str] = None,
) -> str:
    """Verify a HoVer claim using OpenAI API."""
    client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()

    user_msg = USER_PROMPT.format(claim=claim, evidence=evidence)

    kwargs = {}
    if "gpt-5" in model:
        kwargs["temperature"] = 1.0
        kwargs["max_completion_tokens"] = 8192
    else:
        kwargs["temperature"] = 0.0
        kwargs["max_tokens"] = 256

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        tools=[TOOL_SCHEMA],
        tool_choice={"type": "function", "function": {"name": "verify_claim"}},
        **kwargs,
    )

    # Try to get tool call response first
    tool_calls = response.choices[0].message.tool_calls
    if tool_calls and len(tool_calls) > 0:
        args = json.loads(tool_calls[0].function.arguments)
        return args.get("verdict", "")

    # Fallback: use text content
    return response.choices[0].message.content or ""


def create_hover_local_api(system_prompt: str, env_api_key: str):
    """Create HoVer local API."""
    os.environ["ENVIRONMENT_API_KEY"] = env_api_key

    dataset = HoVerDataset()
    dataset.ensure_ready()

    async def run_rollout(request: RolloutRequest, fastapi_request) -> RolloutResponse:
        seed = request.env.seed
        sample = dataset.sample(seed)

        inference_url = request.policy.config.get("inference_url")
        os.environ["OPENAI_BASE_URL"] = inference_url
        api_key = request.policy.config.get("api_key")
        model = request.policy.config.get("model", "gpt-4o-mini")

        predicted = await verify_claim(
            claim=sample["claim"],
            evidence=sample["evidence"],
            system_prompt=system_prompt,
            model=model,
            api_key=api_key,
        )

        is_correct = verdicts_match(predicted, sample["label"])
        reward = 1.0 if is_correct else 0.0

        return RolloutResponse(
            run_id=request.run_id,
            metrics=RolloutMetrics(outcome_reward=reward),
            trace=None,
            trace_correlation_id=request.policy.config.get("trace_correlation_id"),
        )

    def provide_taskset_description():
        return {
            "splits": ["train"],
            "sizes": {"train": dataset.size()},
        }

    def provide_task_instances(seeds):
        for seed in seeds:
            sample = dataset.sample(seed)
            yield TaskInfo(
                task={"id": APP_ID, "name": APP_NAME},
                dataset={"id": APP_ID, "split": "train", "index": sample["index"]},
                inference={"tool": "verify_claim"},
                limits={"max_turns": 1},
                task_metadata={
                    "claim": sample["claim"],
                    "expected_label": sample["label"],
                },
            )

    return create_local_api(LocalAPIConfig(
        app_id=APP_ID,
        name=APP_NAME,
        description=f"{APP_NAME} local API for claim verification.",
        provide_taskset_description=provide_taskset_description,
        provide_task_instances=provide_task_instances,
        rollout=run_rollout,
        cors_origins=["*"],
    ))


async def run_single_experiment(
    model: str,
    run_number: int,
    api_key: str,
    env_api_key: str,
    results_dir: Path,
    dry_run: bool = False,
) -> dict:
    """Run a single GEPA experiment."""
    print(f"\n{'='*60}")
    print(f"Running: {model} run {run_number}")
    print(f"{'='*60}")

    if dry_run:
        print(f"[DRY RUN] Would run {model} run {run_number}")
        return {"model": model, "run": run_number, "status": "dry_run"}

    start_time = time.time()

    # Create local API
    app = create_hover_local_api(BASELINE_SYSTEM_PROMPT, env_api_key)

    kill_port(LOCAL_API_PORT)
    run_server_background(app, LOCAL_API_PORT)

    print(f"Waiting for local API on port {LOCAL_API_PORT}...")
    await wait_for_health_check("localhost", LOCAL_API_PORT, env_api_key, timeout=30.0)
    print("Local API ready!")

    # Create tunnel (or use localhost for local backend)
    if "localhost" in BACKEND_URL or "127.0.0.1" in BACKEND_URL:
        local_api_url = f"http://localhost:{LOCAL_API_PORT}"
        print(f"Using local URL directly: {local_api_url}")
    else:
        print("Provisioning Cloudflare tunnel...")
        tunnel = await TunneledLocalAPI.create(
            local_port=LOCAL_API_PORT,
            backend=TunnelBackend.CloudflareManagedTunnel,
            api_key=api_key,
            env_api_key=env_api_key,
            reason=f"hover_{model}_run{run_number}",
            backend_url=BACKEND_URL,
            progress=True,
        )
        local_api_url = tunnel.url
        print(f"Tunnel URL: {local_api_url}")

    # Build GEPA config
    policy_config = {
        "model": model,
        "provider": "openai",
        "inference_mode": "synth_hosted",
    }

    if "gpt-5" in model:
        policy_config["temperature"] = 1.0
        policy_config["max_completion_tokens"] = 8192
    else:
        policy_config["temperature"] = 0.0
        policy_config["max_completion_tokens"] = 256

    config_body = {
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_id": APP_ID,
            "task_app_url": local_api_url,
            "initial_prompt": {
                "id": "hover_pattern",
                "name": "HoVer Claim Verification",
                "messages": [
                    {"role": "system", "order": 0, "pattern": BASELINE_SYSTEM_PROMPT},
                    {"role": "user", "order": 1, "pattern": USER_PROMPT},
                ],
                "wildcards": {"claim": "REQUIRED", "evidence": "REQUIRED"},
            },
            "policy": policy_config,
            "env_config": {},
            "gepa": {
                "env_name": APP_ID,
                "evaluation": {
                    "seeds": TRAINING_SEEDS,
                    "validation_seeds": VALIDATION_SEEDS,
                },
                "rollout": {
                    "budget": ROLLOUT_BUDGET,
                    "max_concurrent": 10,
                    "minibatch_size": 10,
                },
                "mutation": {"rate": 0.3},
                "population": {
                    "initial_size": 10,
                    "num_generations": 5,
                    "children_per_generation": 4,
                },
                "archive": {"pareto_set_size": 20},
                "token": {"counting_model": "gpt-4"},
            },
        },
    }

    print(f"Creating GEPA job...")

    pl_job = PromptLearningJob.from_dict(
        config_dict=config_body,
        backend_url=BACKEND_URL,
        api_key=api_key,
        task_app_api_key=env_api_key,
        skip_health_check=True,
    )

    job_id = pl_job.submit()
    print(f"Job ID: {job_id}")

    result = pl_job.poll_until_complete(timeout=7200.0, interval=5.0, progress=True)

    elapsed = time.time() - start_time

    experiment_result = {
        "model": model,
        "run": run_number,
        "job_id": job_id,
        "status": result.status.value,
        "best_score": result.best_score if result.succeeded else None,
        "error": result.error if result.failed else None,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
        "backend_url": BACKEND_URL,
    }

    # Save optimized prompt if succeeded
    if result.succeeded:
        score_str = f"{result.best_score:.1%}" if result.best_score is not None else "N/A"
        print(f"\nGEPA succeeded! Best score: {score_str}")

        pl_client = PromptLearningClient(BACKEND_URL, api_key)
        prompt_results = await pl_client.get_prompts(job_id)

        if prompt_results.top_prompts:
            top_prompt = prompt_results.top_prompts[0]

            optimized_full_text = None
            try:
                optimized_full_text = await pl_client.get_prompt_text(job_id, rank=0)
                if not optimized_full_text:
                    optimized_full_text = await pl_client.get_prompt_text(job_id, rank=1)
            except Exception as e:
                print(f"Warning: Could not get prompt text: {e}")

            prompt_file = results_dir / f"hover_{model.replace('-', '_')}_run{run_number}_prompt.json"
            with open(prompt_file, "w") as f:
                json.dump({
                    "model": model,
                    "run": run_number,
                    "job_id": job_id,
                    "best_score": result.best_score,
                    "train_accuracy": top_prompt.get("train_accuracy") if isinstance(top_prompt, dict) else None,
                    "val_accuracy": top_prompt.get("val_accuracy") if isinstance(top_prompt, dict) else None,
                    "optimized_prompt_text": optimized_full_text,
                    "raw_prompt_data": top_prompt,
                }, f, indent=2)
            print(f"Saved optimized prompt to {prompt_file}")
            experiment_result["prompt_file"] = str(prompt_file)
    else:
        print(f"\nGEPA failed: {result.error}")

    # Save result
    result_file = results_dir / f"hover_{model.replace('-', '_')}_run{run_number}_result.json"
    with open(result_file, "w") as f:
        json.dump(experiment_result, f, indent=2)
    print(f"Saved result to {result_file}")

    # Cleanup
    cleanup_all()
    kill_port(LOCAL_API_PORT)

    return experiment_result


async def main():
    global BACKEND_URL

    parser = argparse.ArgumentParser(description="Run HoVer GEPA benchmark")
    parser.add_argument("--model", type=str, help="Run only this model")
    parser.add_argument("--run", type=int, help="Run only this run number")
    parser.add_argument("--dry-run", action="store_true", help="Print experiments without running")
    parser.add_argument("--backend-url", type=str, default=BACKEND_URL, help="Backend URL")
    args = parser.parse_args()

    BACKEND_URL = args.backend_url

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Get API key
    api_key = os.environ.get("SYNTH_API_KEY", "")
    if not api_key and not args.dry_run:
        print("Minting demo API key...")
        api_key = mint_demo_api_key(backend_url=BACKEND_URL)
        print(f"Demo API Key: {api_key[:25]}...")

    # Mint environment key
    if not args.dry_run:
        env_api_key = mint_environment_api_key()
        print(f"Minted env key: {env_api_key[:12]}...{env_api_key[-4:]}")

        result = setup_environment_api_key(BACKEND_URL, api_key, token=env_api_key)
        print(f"Uploaded env key: {result}")
    else:
        env_api_key = "dry_run_key"

    # Determine experiments to run
    models = [args.model] if args.model else MODELS
    runs = [args.run] if args.run else list(range(1, RUNS_PER_MODEL + 1))

    all_results = []

    for model in models:
        for run in runs:
            try:
                result = await run_single_experiment(
                    model=model,
                    run_number=run,
                    api_key=api_key,
                    env_api_key=env_api_key,
                    results_dir=results_dir,
                    dry_run=args.dry_run,
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error in {model} run {run}: {e}")
                all_results.append({
                    "model": model,
                    "run": run,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "backend_url": BACKEND_URL,
        "experiments": all_results,
    }

    summary_file = results_dir / "benchmark_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for r in all_results:
        score = f"{r['best_score']:.1%}" if r.get('best_score') is not None else "N/A"
        print(f"{r['model']} run {r['run']}: {r['status']} (score: {score})")


if __name__ == "__main__":
    asyncio.run(main())
