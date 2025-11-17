#!/usr/bin/env python3
"""
Complete GEPA Prompt Optimization Example
==========================================

This script demonstrates the full pipeline for optimizing prompts using GEPA:

1. Evaluate initial/baseline prompts
2. Start in-process task app (with Cloudflare tunnel in prod, local in dev)
3. Run GEPA optimization with programmatic polling
4. Retrieve best prompts
5. Evaluate final prompts with optimized version

Usage:
    # Local development (uses localhost, no tunnel)
    SYNTH_TUNNEL_MODE=local uv run run_gepa_example.py

    # Production (uses Cloudflare tunnel)
    uv run run_gepa_example.py

Requirements:
    - GROQ_API_KEY in .env (for policy model)
    - OPENAI_API_KEY in .env (for mutation LLM)
    - SYNTH_API_KEY in .env (for backend API)
    - ENVIRONMENT_API_KEY in .env (for task app auth)
    - synth-ai backend running (localhost:8000 or BACKEND_BASE_URL)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv()

from synth_ai.task import InProcessTaskApp
from synth_ai.api.train.prompt_learning import PromptLearningJob
from synth_ai.learning.prompt_learning_client import PromptLearningClient
from synth_ai.api.train.utils import ensure_api_base
from task_app import build_config


async def evaluate_baseline(
    task_app_url: str,
    seeds: list[int],
    model: str = "openai/gpt-oss-20b",
    provider: str = "groq",
) -> dict[str, Any]:
    """Evaluate baseline prompt performance.

    Args:
        task_app_url: Task app URL
        seeds: List of seeds to evaluate
        model: Model to use
        provider: Provider to use

    Returns:
        Dictionary with accuracy and results
    """
    import httpx

    print(f"\n{'='*80}")
    print("Step 1: Evaluating Baseline Prompts")
    print(f"{'='*80}\n")

    api_key = os.getenv("ENVIRONMENT_API_KEY", "test")
    correct_count = 0
    results = []

    async with httpx.AsyncClient(timeout=300.0) as client:
        print(f"Evaluating {len(seeds)} seeds with baseline prompt...")

        for seed in seeds:
            rollout_request = {
                "run_id": f"baseline_{seed}",
                "env": {
                    "env_name": "heartdisease",
                    "seed": seed,
                    "config": {"split": "train"},
                },
                "policy": {
                    "policy_name": "baseline",
                    "config": {
                        "model": model,
                        "provider": provider,
                        "inference_url": f"https://api.{provider}.com/v1" if provider == "groq" else "https://api.openai.com/v1",
                        "temperature": 1.0,
                        "max_completion_tokens": 512,
                    },
                },
                "ops": ["policy"],
                "mode": "eval",
            }

            try:
                response = await client.post(
                    f"{task_app_url}/rollout",
                    json=rollout_request,
                    headers={"X-API-Key": api_key},
                )
                response.raise_for_status()
                data = response.json()

                metrics = data.get("metrics", {})
                reward = metrics.get("mean_return", 0.0)

                trajectories = data.get("trajectories", [])
                is_correct = False
                if trajectories:
                    steps = trajectories[0].get("steps", [])
                    if steps:
                        info = steps[0].get("info", {})
                        is_correct = info.get("label_correct", False)

                if is_correct:
                    correct_count += 1

                results.append({
                    "seed": seed,
                    "correct": is_correct,
                    "reward": reward,
                })

                print(f"  Seed {seed}: {'✓' if is_correct else '✗'} (reward={reward:.3f})")

            except Exception as e:
                print(f"  Seed {seed}: ERROR - {e}")
                results.append({"seed": seed, "error": str(e)})

    accuracy = correct_count / len(seeds) if seeds else 0.0

    print(f"\nBaseline Results:")
    print(f"  Accuracy: {accuracy:.2%} ({correct_count}/{len(seeds)})")
    print()

    return {
        "accuracy": accuracy,
        "num_correct": correct_count,
        "total": len(seeds),
        "results": results,
    }


async def run_gepa_optimization(
    task_app_url: str,
    backend_url: str,
    api_key: str,
    task_app_api_key: str,
    rollout_budget: int = 200,
) -> str:
    """Run GEPA optimization and return job ID.

    Args:
        task_app_url: Task app URL
        backend_url: Backend URL
        api_key: Backend API key
        task_app_api_key: Task app API key
        rollout_budget: Rollout budget

    Returns:
        Job ID
    """
    print(f"\n{'='*80}")
    print("Step 2: Running GEPA Optimization")
    print(f"{'='*80}\n")

    # Create a temporary config for GEPA
    import tempfile
    import toml

    config = {
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_url": task_app_url,
            "task_app_api_key": task_app_api_key,
            "env_name": "heartdisease",
            "initial_prompt": {
                "id": "heartdisease_pattern",
                "name": "Heart Disease Classification Pattern",
                "messages": [
                    {
                        "role": "system",
                        "pattern": "You are a medical classification assistant. Based on the patient's features, classify whether they have heart disease. Respond with '1' for heart disease or '0' for no heart disease.\n\nYou have access to the function `heart_disease_classify` which accepts your predicted classification. Call this tool with your classification when you're ready to submit your answer.",
                        "order": 0,
                    },
                    {
                        "role": "user",
                        "pattern": "Patient Features:\n{features}\n\nClassify: Does this patient have heart disease? Respond with '1' for yes or '0' for no.",
                        "order": 1,
                    },
                ],
                "wildcards": {
                    "features": "REQUIRED",
                },
            },
            "policy": {
                "model": "openai/gpt-oss-20b",
                "provider": "groq",
                "temperature": 1.0,
                "max_completion_tokens": 512,
            },
            "gepa": {
                "env_name": "heartdisease",
                "evaluation": {
                    "train_seeds": list(range(30)),
                },
                "rollout": {
                    "budget": rollout_budget,
                },
                "mutation": {
                    "rate": 0.3,
                    "llm_model": "openai/gpt-oss-20b",
                    "llm_provider": "groq",
                },
                "population": {
                    "initial_size": 10,
                    "num_generations": 10,
                    "children_per_generation": 8,
                },
            },
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        toml.dump(config, f)
        temp_config_path = f.name

    try:
        job = PromptLearningJob.from_config(
            config_path=temp_config_path,
            backend_url=backend_url,
            api_key=api_key,
            task_app_api_key=task_app_api_key,
        )

        print(f"Submitting GEPA job...")
        job_id = job.submit()
        print(f"✓ Job submitted: {job_id}\n")

        # Poll for completion
        start_time = time.time()
        last_status = None

        def on_status(status):
            nonlocal last_status
            elapsed = time.time() - start_time
            state = status.get("status", "unknown")

            if state != last_status or int(elapsed) % 10 == 0:
                timestamp = time.strftime("%H:%M:%S")
                progress = status.get("progress", {})
                best_score = status.get("best_score")

                if progress:
                    completed = progress.get("completed", 0)
                    total = progress.get("total", 0)
                    if total > 0:
                        pct = (completed / total) * 100
                        score_str = f" | Best: {best_score:.3f}" if best_score is not None else ""
                        print(
                            f"[{timestamp}] {elapsed:6.1f}s  Status: {state} ({completed}/{total} = {pct:.1f}%){score_str}"
                        )
                    else:
                        score_str = f" | Best: {best_score:.3f}" if best_score is not None else ""
                        print(f"[{timestamp}] {elapsed:6.1f}s  Status: {state}{score_str}")
                else:
                    score_str = f" | Best: {best_score:.3f}" if best_score is not None else ""
                    print(f"[{timestamp}] {elapsed:6.1f}s  Status: {state}{score_str}")
                last_status = state

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: job.poll_until_complete(
                timeout=3600.0,
                interval=5.0,
                on_status=on_status,
            ),
        )

        total_time = time.time() - start_time
        print(f"\n✓ GEPA optimization complete in {total_time:.1f}s\n")

        return job_id

    finally:
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)


async def evaluate_final(
    task_app_url: str,
    job_id: str,
    backend_url: str,
    api_key: str,
    seeds: list[int],
    model: str = "openai/gpt-oss-20b",
    provider: str = "groq",
) -> dict[str, Any]:
    """Evaluate final optimized prompts.

    Args:
        task_app_url: Task app URL
        job_id: GEPA job ID
        backend_url: Backend URL
        api_key: Backend API key
        seeds: List of seeds to evaluate
        model: Model to use
        provider: Provider to use

    Returns:
        Dictionary with accuracy and results
    """
    print(f"\n{'='*80}")
    print("Step 3: Evaluating Optimized Prompts")
    print(f"{'='*80}\n")

    # Get best prompts from job
    client = PromptLearningClient(
        ensure_api_base(backend_url),
        api_key,
    )
    prompt_results = await client.get_prompts(job_id)

    if not prompt_results.best_prompt:
        print("⚠️  No best prompt found - job may have failed")
        return {"accuracy": 0.0, "num_correct": 0, "total": len(seeds), "results": []}

    print(f"Best score from optimization: {prompt_results.best_score:.2%}")
    print(f"Using optimized prompt for final evaluation...\n")

    # For final evaluation, we need to use the interceptor pattern
    # The optimized prompt is registered with the interceptor automatically
    # when we use the inference_url from the backend

    import httpx

    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY", "test")
    correct_count = 0
    results = []

    # Get interceptor URL from backend
    # The interceptor is typically at the same base URL as the backend
    # For local dev, it's usually localhost:8000
    # For production, it might be interceptor.usesynth.ai or similar
    # We'll use the backend URL and let the backend route to interceptor
    interceptor_base = backend_url
    if "localhost" not in backend_url and "127.0.0.1" not in backend_url:
        # For production, interceptor might be on a subdomain
        # This is a simplified approach - adjust based on your setup
        pass

    async with httpx.AsyncClient(timeout=300.0) as client:
        print(f"Evaluating {len(seeds)} seeds with optimized prompt...")

        for seed in seeds:
            # Use interceptor URL with job_id
            # The interceptor pattern uses /v1/{trial_id}/chat/completions
            # For GEPA, trial_id is typically the job_id
            inference_url = f"{interceptor_base}/v1/{job_id}/chat/completions"

            rollout_request = {
                "run_id": f"final_{seed}",
                "env": {
                    "env_name": "heartdisease",
                    "seed": seed,
                    "config": {"split": "train"},
                },
                "policy": {
                    "policy_name": "optimized",
                    "config": {
                        "model": model,
                        "provider": provider,
                        "inference_url": inference_url,
                        "temperature": 1.0,
                        "max_completion_tokens": 512,
                    },
                },
                "ops": ["policy"],
                "mode": "eval",
            }

            try:
                response = await client.post(
                    f"{task_app_url}/rollout",
                    json=rollout_request,
                    headers={"X-API-Key": task_app_api_key},
                )
                response.raise_for_status()
                data = response.json()

                metrics = data.get("metrics", {})
                reward = metrics.get("mean_return", 0.0)

                trajectories = data.get("trajectories", [])
                is_correct = False
                if trajectories:
                    steps = trajectories[0].get("steps", [])
                    if steps:
                        info = steps[0].get("info", {})
                        is_correct = info.get("label_correct", False)

                if is_correct:
                    correct_count += 1

                results.append({
                    "seed": seed,
                    "correct": is_correct,
                    "reward": reward,
                })

                print(f"  Seed {seed}: {'✓' if is_correct else '✗'} (reward={reward:.3f})")

            except Exception as e:
                print(f"  Seed {seed}: ERROR - {e}")
                results.append({"seed": seed, "error": str(e)})

    accuracy = correct_count / len(seeds) if seeds else 0.0

    print(f"\nFinal Results:")
    print(f"  Accuracy: {accuracy:.2%} ({correct_count}/{len(seeds)})")
    print()

    return {
        "accuracy": accuracy,
        "num_correct": correct_count,
        "total": len(seeds),
        "results": results,
    }


async def main():
    """Main entry point."""
    print("\n" + "=" * 80)
    print("GEPA Prompt Optimization Example")
    print("=" * 80 + "\n")

    # Check requirements
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY required in .env")
        sys.exit(1)

    backend_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
    api_key = os.getenv("SYNTH_API_KEY", "test")
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY", "test")

    # Configuration
    rollout_budget = 200  # Adjust based on your needs
    eval_seeds = list(range(20))  # Seeds for baseline and final evaluation
    train_seeds = list(range(30))  # Seeds for GEPA optimization

    # Determine tunnel mode
    tunnel_mode = os.getenv("SYNTH_TUNNEL_MODE", "quick")
    if tunnel_mode == "local":
        print("Using local mode (no tunnel)\n")
    else:
        print("Using Cloudflare tunnel mode\n")

    try:
        # Start in-process task app
        async with InProcessTaskApp(
            config_factory=build_config,
            port=8114,
            tunnel_mode=tunnel_mode,
            api_key=task_app_api_key,
        ) as task_app:
            print(f"✅ Task app running at: {task_app.url}\n")

            # Step 1: Evaluate baseline
            baseline_results = await evaluate_baseline(
                task_app_url=task_app.url,
                seeds=eval_seeds,
            )

            # Step 2: Run GEPA optimization
            job_id = await run_gepa_optimization(
                task_app_url=task_app.url,
                backend_url=backend_url,
                api_key=api_key,
                task_app_api_key=task_app_api_key,
                rollout_budget=rollout_budget,
            )

            # Step 3: Evaluate final optimized prompts
            final_results = await evaluate_final(
                task_app_url=task_app.url,
                job_id=job_id,
                backend_url=backend_url,
                api_key=api_key,
                seeds=eval_seeds,
            )

            # Summary
            print("\n" + "=" * 80)
            print("Summary")
            print("=" * 80)
            print(f"Baseline Accuracy: {baseline_results['accuracy']:.2%}")
            print(f"Final Accuracy:    {final_results['accuracy']:.2%}")
            improvement = final_results['accuracy'] - baseline_results['accuracy']
            print(f"Improvement:       {improvement:+.2%}")
            print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

