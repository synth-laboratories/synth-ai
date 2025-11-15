#!/usr/bin/env python3
"""
Run Synth GEPA and DSPy GEPA side-by-side comparison on Heart Disease.
This script runs both optimizers with the same configuration and compares results.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Enable verbose logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add parent to path
parent_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from synth_ai.api.train.prompt_learning import PromptLearningJob

# Import DSPy adapter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "langprobe"))
from task_specific.heartdisease.dspy_heartdisease_adapter import run_dspy_gepa_heartdisease

# Load environment
load_dotenv()


async def main():
    """Run Synth GEPA and DSPy GEPA side-by-side."""

    print("\n" + "="*80)
    print("GEPA Comparison: Synth GEPA vs DSPy GEPA")
    print("Heart Disease Classification")
    print("="*80 + "\n")

    # Check environment
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY environment variable required")
        sys.exit(1)

    # Configuration
    config_path = Path(__file__).parent / "configs" / "heartdisease_gepa_local.toml"

    # Parse config to get seeds and budget
    import toml
    config = toml.load(config_path)
    train_seeds = config["prompt_learning"]["gepa"]["evaluation"]["train_seeds"]
    val_seeds = config["prompt_learning"]["gepa"]["evaluation"]["val_seeds"]
    rollout_budget = config["prompt_learning"]["gepa"]["rollout"]["budget"]

    backend_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
    api_key = os.getenv("SYNTH_API_KEY", "test")
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY", "test")

    print(f"Configuration:")
    print(f"  Config: {config_path.name}")
    print(f"  Backend: {backend_url}")
    print(f"  Task App: http://127.0.0.1:8114")
    print(f"  Training seeds: {len(train_seeds)} examples")
    print(f"  Validation seeds: {len(val_seeds)} examples")
    print(f"  Rollout budget: {rollout_budget}")
    print()

    # ========================================================================
    # PART 1: Run Synth GEPA
    # ========================================================================
    print("="*80)
    print("PART 1: Running Synth GEPA via API")
    print("="*80 + "\n")

    synth_start_time = time.time()

    try:
        job = PromptLearningJob.from_config(
            config_path=config_path,
            backend_url=backend_url,
            api_key=api_key,
            task_app_api_key=task_app_api_key,
        )

        job_id = job.submit()
        print(f"✅ Job submitted: {job_id}")
        print(f"⏳ Polling for completion...\n")

        def on_status(status):
            elapsed = time.time() - synth_start_time
            state = status.get("status", "unknown")
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {elapsed:6.1f}s  Status: {state}")

        result = job.poll_until_complete(
            timeout=3600.0,
            interval=5.0,
            on_status=on_status,
        )

        synth_total_time = time.time() - synth_start_time
        print(f"✅ Synth GEPA complete in {synth_total_time:.1f}s\n")

    except Exception as e:
        print(f"❌ Error running Synth GEPA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Extract Synth GEPA results (using async client directly since we're in async context)
    try:
        from synth_ai.learning.prompt_learning_client import PromptLearningClient
        from synth_ai.api.train.utils import ensure_api_base

        client = PromptLearningClient(
            ensure_api_base(backend_url),
            api_key,
        )
        prompt_results = await client.get_prompts(job._job_id)

        results = {
            "best_prompt": prompt_results.best_prompt,
            "best_score": prompt_results.best_score,
            "top_prompts": prompt_results.top_prompts,
            "optimized_candidates": prompt_results.optimized_candidates,
            "attempted_candidates": prompt_results.attempted_candidates,
            "validation_results": prompt_results.validation_results,
        }

        best_score = results.get("best_score", 0.0)
        best_prompt = results.get("best_prompt", {})
        top_prompts = results.get("top_prompts", [])
        attempted_candidates = results.get("attempted_candidates", [])

        # Helper to extract prompt text from various structures
        def extract_prompt_text(prompt_dict):
            """Extract full text from prompt structure."""
            if "prompt_sections" in prompt_dict:
                sections = prompt_dict["prompt_sections"]
                return "\n\n".join([s.get("content", "") for s in sections if s.get("content")])
            elif "object" in prompt_dict and "text_replacements" in prompt_dict["object"]:
                replacements = prompt_dict["object"]["text_replacements"]
                if replacements:
                    return replacements[0].get("new_text", "")
            return ""

        # Reconstruct prompts from available data (silently)
        if len(top_prompts) == 0 and attempted_candidates:
            sorted_candidates = sorted(attempted_candidates, key=lambda x: x.get("accuracy", 0))
            baseline_candidate = sorted_candidates[0]
            baseline_prompt = extract_prompt_text(baseline_candidate)
            baseline_score = baseline_candidate.get("accuracy", 0.0)

            best_candidate = sorted_candidates[-1]
            optimized_prompt = extract_prompt_text(best_candidate)
            optimized_score = best_candidate.get("accuracy", 0.0)

            if not optimized_prompt:
                optimized_prompt = extract_prompt_text(best_prompt)
                optimized_score = best_score

            top_prompts = [
                {
                    "rank": 1,
                    "full_text": baseline_prompt,
                    "val_accuracy": baseline_score,
                },
                {
                    "rank": 2,
                    "full_text": optimized_prompt,
                    "val_accuracy": optimized_score,
                }
            ]

        synth_baseline_score = top_prompts[0].get("val_accuracy", 0.0) if len(top_prompts) > 0 else 0.0
        synth_optimized_score = top_prompts[1].get("val_accuracy", best_score) if len(top_prompts) > 1 else best_score
        synth_baseline_prompt = top_prompts[0].get("full_text", "") if len(top_prompts) > 0 else ""
        synth_optimized_prompt = top_prompts[1].get("full_text", "") if len(top_prompts) > 1 else ""

        print(f"✅ Synth baseline: {synth_baseline_score:.4f}, optimized: {synth_optimized_score:.4f}")

    except Exception as e:
        print(f"❌ Error extracting Synth GEPA results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ========================================================================
    # PART 2: Run DSPy GEPA
    # ========================================================================
    print("\n" + "="*80)
    print("PART 2: Running DSPy GEPA (in-process)")
    print("="*80 + "\n")

    dspy_output_dir = Path(__file__).parent / "results" / "dspy_gepa_comparison"

    try:
        dspy_results = await run_dspy_gepa_heartdisease(
            task_app_url="http://127.0.0.1:8114",  # Not used but expected by function
            train_seeds=train_seeds,
            val_seeds=val_seeds,
            rollout_budget=rollout_budget,
            output_dir=dspy_output_dir,
        )

        dspy_best_score = dspy_results.get("best_score", 0.0)
        dspy_total_time = dspy_results.get("total_time", 0.0)

        # Load detailed results to get baseline and prompts
        detailed_file = dspy_output_dir / "dspy_gepa_detailed_results.json"
        if detailed_file.exists():
            with open(detailed_file) as f:
                dspy_detailed = json.load(f)

            dspy_baseline_score = dspy_detailed.get("baseline_score", 0.0)
            dspy_candidates = dspy_detailed.get("candidates", [])

            dspy_baseline_prompt = ""
            dspy_optimized_prompt = ""

            if dspy_candidates and len(dspy_candidates) >= 1:
                dspy_baseline_prompt = dspy_candidates[0].get("instructions", {}).get("predict.predict", "")
            if dspy_candidates and len(dspy_candidates) >= 2:
                dspy_optimized_prompt = dspy_candidates[1].get("instructions", {}).get("predict.predict", "")
            elif dspy_candidates and len(dspy_candidates) >= 1:
                dspy_optimized_prompt = dspy_candidates[0].get("instructions", {}).get("predict.predict", "")

        print(f"✅ DSPy baseline: {dspy_baseline_score:.4f}, optimized: {dspy_best_score:.4f}, time: {dspy_total_time:.1f}s")

    except Exception as e:
        print(f"❌ Error running DSPy GEPA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ========================================================================
    # PART 3: Comparison Display
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON: Synth GEPA vs DSPy GEPA")
    print("="*80 + "\n")

    # Comparison table
    print("┌" + "─"*78 + "┐")
    print("│" + " Framework Comparison".center(78) + "│")
    print("├" + "─"*25 + "┬" + "─"*25 + "┬" + "─"*26 + "┤")
    print("│" + " Metric".ljust(25) + "│" + " Synth GEPA".center(25) + "│" + " DSPy GEPA".center(26) + "│")
    print("├" + "─"*25 + "┼" + "─"*25 + "┼" + "─"*26 + "┤")

    # Baseline
    print("│" + " Baseline Score".ljust(25) + "│" +
          f" {synth_baseline_score:.4f} ({synth_baseline_score*100:.1f}%)".center(25) + "│" +
          f" {dspy_baseline_score:.4f} ({dspy_baseline_score*100:.1f}%)".center(26) + "│")

    # Optimized
    print("│" + " Optimized Score".ljust(25) + "│" +
          f" {synth_optimized_score:.4f} ({synth_optimized_score*100:.1f}%)".center(25) + "│" +
          f" {dspy_best_score:.4f} ({dspy_best_score*100:.1f}%)".center(26) + "│")

    # Improvement
    synth_improvement = synth_optimized_score - synth_baseline_score
    dspy_improvement = dspy_best_score - dspy_baseline_score

    print("│" + " Absolute Improvement".ljust(25) + "│" +
          f" {synth_improvement:+.4f} ({synth_improvement*100:+.1f}%)".center(25) + "│" +
          f" {dspy_improvement:+.4f} ({dspy_improvement*100:+.1f}%)".center(26) + "│")

    # Time
    print("│" + " Optimization Time".ljust(25) + "│" +
          f" {synth_total_time:.1f}s".center(25) + "│" +
          f" {dspy_total_time:.1f}s".center(26) + "│")

    # Rollouts
    print("│" + " Rollout Budget".ljust(25) + "│" +
          f" {rollout_budget}".center(25) + "│" +
          f" {rollout_budget}".center(26) + "│")

    print("└" + "─"*25 + "┴" + "─"*25 + "┴" + "─"*26 + "┘")

    # Prompt samples
    print("\n" + "="*80)
    print("BASELINE PROMPTS")
    print("="*80 + "\n")

    # Synth baseline
    print("Synth GEPA Baseline:")
    print("-" * 80)
    synth_baseline_lines = synth_baseline_prompt.split("\n")[:5] if synth_baseline_prompt else ["(empty)"]
    for line in synth_baseline_lines:
        print(line[:78])
    if len(synth_baseline_prompt.split("\n")) > 5:
        print("... [truncated]")
    print()

    # DSPy baseline
    print("DSPy GEPA Baseline:")
    print("-" * 80)
    dspy_baseline_lines = dspy_baseline_prompt.split("\n")[:5] if dspy_baseline_prompt else ["(empty)"]
    for line in dspy_baseline_lines:
        print(line[:78])
    if len(dspy_baseline_prompt.split("\n")) > 5:
        print("... [truncated]")
    print()

    print("="*80)
    print("OPTIMIZED PROMPTS")
    print("="*80 + "\n")

    # Synth optimized
    print("Synth GEPA Optimized:")
    print("-" * 80)
    synth_optimized_lines = synth_optimized_prompt.split("\n")[:10] if synth_optimized_prompt else ["(empty)"]
    for line in synth_optimized_lines:
        print(line[:78])
    if len(synth_optimized_prompt.split("\n")) > 10:
        print("... [truncated]")
    print()

    # DSPy optimized
    print("DSPy GEPA Optimized:")
    print("-" * 80)
    dspy_optimized_lines = dspy_optimized_prompt.split("\n")[:10] if dspy_optimized_prompt else ["(empty)"]
    for line in dspy_optimized_lines:
        print(line[:78])
    if len(dspy_optimized_prompt.split("\n")) > 10:
        print("... [truncated]")
    print()

    print("="*80)
    print("✅ Comparison Complete!")
    print("="*80 + "\n")

    print(f"📊 Synth GEPA: {synth_baseline_score*100:.1f}% → {synth_optimized_score*100:.1f}% ({synth_improvement*100:+.1f}%)")
    print(f"📊 DSPy GEPA: {dspy_baseline_score*100:.1f}% → {dspy_best_score*100:.1f}% ({dspy_improvement*100:+.1f}%)")
    print(f"⏱️  Time: Synth {synth_total_time:.1f}s vs DSPy {dspy_total_time:.1f}s")

if __name__ == "__main__":
    asyncio.run(main())
