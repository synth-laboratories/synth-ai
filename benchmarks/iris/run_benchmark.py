#!/usr/bin/env python3
"""Run Iris GEPA benchmark following banking77 demo standards."""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from fastapi import Request
from openai import AsyncOpenAI
from synth_ai.core.env import mint_demo_api_key
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient
from synth_ai.sdk.learning.rl import mint_environment_api_key, setup_environment_api_key
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.task import run_server_background
from synth_ai.sdk.task.contracts import (
    DatasetInfo,
    InferenceInfo,
    LimitsInfo,
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskDescriptor,
    TaskInfo,
)
from synth_ai.sdk.tunnels import (
    TunnelBackend,
    TunneledLocalAPI,
    cleanup_all,
    kill_port,
    wait_for_health_check,
)

# Configuration
BACKEND_URL = os.environ.get("SYNTH_BACKEND_URL", "http://localhost:8000")
LOCAL_API_PORT = 8003

# Iris dataset config
APP_ID = "iris"
APP_NAME = "Iris Classification"

IRIS_LABELS = ["setosa", "versicolor", "virginica"]

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "iris_classify",
        "description": "Classify the iris species",
        "parameters": {
            "type": "object",
            "properties": {"species": {"type": "string"}},
            "required": ["species"],
        },
    },
}

BASELINE_SYSTEM_PROMPT = """You are a botany classification assistant. Based on the flower's measurements, classify the iris species. Respond with one of: setosa, versicolor, or virginica."""

USER_PROMPT = """Flower Measurements:
{features}

Classify this iris flower. Respond with one of: setosa, versicolor, or virginica."""

# Benchmark config
MODELS = ["gpt-4.1-nano", "gpt-5-nano", "gpt-4o-mini"]
RUNS_PER_MODEL = 3

# IMPORTANT: Iris dataset is ordered by class (0-49: setosa, 50-99: versicolor, 100-149: virginica)
# We need both training AND validation to include all 3 classes for proper evaluation!
# Training: 30 samples from each class = 90 total
TRAINING_SEEDS = (
    list(range(0, 30)) + list(range(50, 80)) + list(range(100, 130))
)  # 90 seeds, all classes
# Validation: 20 samples from each class = 60 total
VALIDATION_SEEDS = (
    list(range(30, 50)) + list(range(80, 100)) + list(range(130, 150))
)  # 60 seeds, all classes
ROLLOUT_BUDGET = 500  # Lower budget for smaller dataset


def format_features(row: dict) -> str:
    """Format iris features for the prompt."""
    return (
        f"Sepal Length: {row['SepalLengthCm']} cm\n"
        f"Sepal Width: {row['SepalWidthCm']} cm\n"
        f"Petal Length: {row['PetalLengthCm']} cm\n"
        f"Petal Width: {row['PetalWidthCm']} cm"
    )


def normalize_species(species: str) -> str:
    """Normalize species name for comparison."""
    species = species.lower().strip()
    if species.startswith("iris-"):
        species = species[5:]
    return species


class IrisDataset:
    def __init__(self):
        self._cache = None

    def _load(self):
        if self._cache is None:
            self._cache = load_dataset("scikit-learn/iris", split="train")
        return self._cache

    def ensure_ready(self):
        self._load()

    def size(self) -> int:
        return len(self._load())

    def sample(self, index: int) -> dict:
        ds = self._load()
        idx = index % len(ds)
        row = ds[idx]
        return {
            "index": idx,
            "features": format_features(row),
            "label": normalize_species(row["Species"]),
        }


async def classify_iris(
    features: str,
    system_prompt: str,
    model: str,
    api_key: str | None = None,
) -> str:
    """Classify iris using OpenAI API."""
    client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()

    user_msg = USER_PROMPT.format(features=features)

    # Model-specific configuration
    kwargs = {}
    if "gpt-5" in model:
        kwargs["temperature"] = 1.0
        kwargs["max_completion_tokens"] = 8192  # Reasoning models need more tokens
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
        tool_choice={"type": "function", "function": {"name": "iris_classify"}},
        **kwargs,
    )

    # Try to get tool call response first
    tool_calls = response.choices[0].message.tool_calls
    if tool_calls and len(tool_calls) > 0:
        args = json.loads(tool_calls[0].function.arguments)
        return normalize_species(args.get("species", ""))

    # Fallback: extract species from text content (for reasoning models like gpt-5-nano)
    content = response.choices[0].message.content or ""
    content_lower = content.lower()
    for species in IRIS_LABELS:
        if species in content_lower:
            return species

    # If no species found, return empty (will be marked as incorrect)
    print(f"WARNING: Could not extract species from response: {content[:100]}...")
    return ""


def create_iris_local_api(system_prompt: str, env_api_key: str):
    """Create iris classification local API."""
    os.environ["ENVIRONMENT_API_KEY"] = env_api_key

    dataset = IrisDataset()
    dataset.ensure_ready()

    async def run_rollout(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
        seed = request.env.seed or 0
        sample = dataset.sample(seed)

        inference_url = request.policy.config.get("inference_url") or ""
        os.environ["OPENAI_BASE_URL"] = inference_url
        api_key = request.policy.config.get("api_key")
        model = request.policy.config.get("model", "gpt-4o-mini")

        predicted = await classify_iris(
            features=sample["features"],
            system_prompt=system_prompt,
            model=model,
            api_key=api_key,
        )

        is_correct = predicted == sample["label"]
        reward = 1.0 if is_correct else 0.0

        return RolloutResponse(
            trace_correlation_id=request.trace_correlation_id or "",
            reward_info=RolloutMetrics(outcome_reward=reward),
            trace=None,
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
                task=TaskDescriptor(id=APP_ID, name=APP_NAME),
                dataset=DatasetInfo(id=APP_ID, split="train", index=sample["index"]),
                inference=InferenceInfo(tool="iris_classify"),
                limits=LimitsInfo(max_turns=1),
                task_metadata={"features": sample["features"], "expected_species": sample["label"]},
            )

    return create_local_api(
        LocalAPIConfig(
            app_id=APP_ID,
            name=APP_NAME,
            description=f"{APP_NAME} local API for classifying iris species.",
            provide_taskset_description=provide_taskset_description,
            provide_task_instances=provide_task_instances,
            rollout=run_rollout,
            cors_origins=["*"],
        )
    )


async def run_single_experiment(
    model: str,
    run_number: int,
    api_key: str,
    env_api_key: str,
    results_dir: Path,
    dry_run: bool = False,
) -> dict:
    """Run a single GEPA experiment."""
    print(f"\n{'=' * 60}")
    print(f"Running: {model} run {run_number}")
    print(f"{'=' * 60}")

    if dry_run:
        print(f"[DRY RUN] Would run {model} run {run_number}")
        return {"model": model, "run": run_number, "status": "dry_run"}

    start_time = time.time()

    # Create local API
    app = create_iris_local_api(BASELINE_SYSTEM_PROMPT, env_api_key)

    kill_port(LOCAL_API_PORT)
    run_server_background(app, LOCAL_API_PORT)

    print(f"Waiting for local API on port {LOCAL_API_PORT}...")
    await wait_for_health_check("localhost", LOCAL_API_PORT, env_api_key, timeout=30.0)
    print("Local API ready!")

    # Create tunnel (or use localhost directly for local backend)
    if "localhost" in BACKEND_URL or "127.0.0.1" in BACKEND_URL:
        # When running against local backend, no tunnel needed
        local_api_url = f"http://localhost:{LOCAL_API_PORT}"
        print(f"Using local URL directly: {local_api_url}")
    else:
        print("Provisioning Cloudflare tunnel...")
        tunnel = await TunneledLocalAPI.create(
            local_port=LOCAL_API_PORT,
            backend=TunnelBackend.CloudflareManagedTunnel,
            synth_user_key=api_key,
            localapi_key=env_api_key,
            synth_base_url=BACKEND_URL,
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
        policy_config["max_completion_tokens"] = 2048
    else:
        policy_config["temperature"] = 0.0
        policy_config["max_completion_tokens"] = 256

    config_body = {
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_id": APP_ID,
            "task_app_url": local_api_url,
            "initial_prompt": {
                "id": "iris_pattern",
                "name": "Iris Classification",
                "messages": [
                    {"role": "system", "order": 0, "pattern": BASELINE_SYSTEM_PROMPT},
                    {"role": "user", "order": 1, "pattern": USER_PROMPT},
                ],
                "wildcards": {"features": "REQUIRED"},
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

    print("Creating GEPA job...")

    pl_job = PromptLearningJob.from_dict(
        config_dict=config_body,
        synth_base_url=BACKEND_URL,
        synth_user_key=api_key,
        localapi_key=env_api_key,
        skip_health_check=True,
    )

    job_id = pl_job.submit()
    print(f"Job ID: {job_id}")

    result = pl_job.poll_until_complete(timeout=3600.0, interval=5.0, progress=True)

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

        # Get the optimized prompt
        pl_client = PromptLearningClient(synth_user_key=api_key, synth_base_url=BACKEND_URL)
        prompt_results = await pl_client.get_prompts(job_id)

        if prompt_results.top_prompts:
            top_prompt = prompt_results.top_prompts[0]

            # Try to get full prompt text using get_prompt_text (rank 0 or 1)
            optimized_full_text = None
            try:
                optimized_full_text = await pl_client.get_prompt_text(job_id, rank=0)
                if not optimized_full_text:
                    optimized_full_text = await pl_client.get_prompt_text(job_id, rank=1)
            except Exception as e:
                print(f"Warning: Could not get prompt text: {e}")

            # Fallback: extract from top_prompt data
            def extract_system_prompt(prompt_data) -> str:
                if isinstance(prompt_data, dict):
                    # Check for full_text first (preferred)
                    if prompt_data.get("full_text"):
                        return prompt_data["full_text"]
                    if "template" in prompt_data:
                        sections = prompt_data["template"].get("sections", [])
                        for s in sections:
                            if s.get("role") == "system":
                                return s.get("content", "")
                    if "system_prompt" in prompt_data:
                        return prompt_data["system_prompt"]
                    if "prompt" in prompt_data:
                        return prompt_data["prompt"]
                return str(prompt_data)

            if not optimized_full_text:
                optimized_full_text = extract_system_prompt(top_prompt)

            prompt_file = (
                results_dir / f"iris_{model.replace('-', '_')}_run{run_number}_prompt.json"
            )
            with open(prompt_file, "w") as f:
                json.dump(
                    {
                        "model": model,
                        "run": run_number,
                        "job_id": job_id,
                        "best_score": result.best_score,
                        "train_accuracy": top_prompt.get("train_accuracy")
                        if isinstance(top_prompt, dict)
                        else None,
                        "val_accuracy": top_prompt.get("val_accuracy")
                        if isinstance(top_prompt, dict)
                        else None,
                        "optimized_prompt_text": optimized_full_text,
                        "raw_prompt_data": top_prompt,
                    },
                    f,
                    indent=2,
                )
            print(f"Saved optimized prompt to {prompt_file}")
            if optimized_full_text:
                print(f"Optimized prompt preview: {optimized_full_text[:300]}...")
            experiment_result["prompt_file"] = str(prompt_file)
    else:
        print(f"\nGEPA failed: {result.error}")

    # Save result
    result_file = results_dir / f"iris_{model.replace('-', '_')}_run{run_number}_result.json"
    with open(result_file, "w") as f:
        json.dump(experiment_result, f, indent=2)
    print(f"Saved result to {result_file}")

    # Cleanup tunnel
    cleanup_all()
    kill_port(LOCAL_API_PORT)

    return experiment_result


async def main():
    global BACKEND_URL

    parser = argparse.ArgumentParser(description="Run Iris GEPA benchmark")
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
        api_key = mint_demo_api_key(synth_base_url=BACKEND_URL)
        print(f"Demo API Key: {api_key[:25]}...")

    # Mint environment key
    if not args.dry_run:
        env_api_key = mint_environment_api_key()
        print(f"Minted env key: {env_api_key[:12]}...{env_api_key[-4:]}")

        result = setup_environment_api_key(api_key, env_api_key, synth_base_url=BACKEND_URL)
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
                all_results.append(
                    {
                        "model": model,
                        "run": run,
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

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
        score = f"{r['best_score']:.1%}" if r.get("best_score") is not None else "N/A"
        print(f"{r['model']} run {r['run']}: {r['status']} (score: {score})")


if __name__ == "__main__":
    asyncio.run(main())
