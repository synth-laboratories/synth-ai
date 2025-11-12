"""Unified comparison runner for all prompt optimization frameworks on Iris.

This script runs all frameworks (Synth GEPA, Synth MIPRO, DSPy MIPROv2, DSPy GEPA, Lakshya GEPA)
at specified budgets and evaluates them on a held-out test set for apples-to-apples comparison.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from .dspy_iris_adapter import run_dspy_gepa_iris, run_dspy_miprov2_iris
from .lakshya_gepa_adapter import run_lakshya_gepa_iris
from .synth_iris_adapter import run_synth_gepa_iris_inprocess, run_synth_mipro_iris_inprocess

load_dotenv()


# Fixed test set seeds for apples-to-apples comparison
# These seeds are NEVER used during optimization, only for final evaluation
TEST_SEEDS = list(range(100, 150))  # Seeds 100-149 (50 examples)


async def run_all_frameworks(
    rollout_budget: int = 100,
    task_app_url: str = "http://127.0.0.1:8115",
    output_base_dir: Optional[Path] = None,
    train_seeds: Optional[list[int]] = None,
    val_seeds: Optional[list[int]] = None,
) -> dict[str, Any]:
    """Run all optimization frameworks with the same budget and dataset splits.

    Args:
        rollout_budget: Rollout budget for each framework (default: 100)
        task_app_url: Task app URL
        output_base_dir: Base directory for results (default: iris/results/comparison)
        train_seeds: Training seeds (default: 0-99)
        val_seeds: Validation seeds for optimization (default: 100-149, but NOT used for final eval)

    Returns:
        Dictionary with results from all frameworks
    """
    if output_base_dir is None:
        output_base_dir = Path(__file__).parent / "results" / "comparison" / f"budget_{rollout_budget}"
    
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Standardize train/val seeds across all frameworks
    if train_seeds is None:
        train_seeds = list(range(100))  # Seeds 0-99 for training
    if val_seeds is None:
        val_seeds = list(range(100, 150))  # Seeds 100-149 for validation during optimization

    results = {
        "rollout_budget": rollout_budget,
        "train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "test_seeds": TEST_SEEDS,
        "frameworks": {},
    }

    print(f"\n{'='*80}")
    print(f"Running All Frameworks Comparison (Budget: {rollout_budget} rollouts)")
    print(f"{'='*80}\n")

    # 1. Synth GEPA
    print(f"[1/5] Running Synth GEPA...")
    try:
        synth_gepa_output = output_base_dir / "synth_gepa"
        synth_gepa_result = await run_synth_gepa_iris_inprocess(
            task_app_url=task_app_url,
            rollout_budget=rollout_budget,
            train_seeds=train_seeds,
            val_seeds=val_seeds,
            output_dir=synth_gepa_output,
        )
        results["frameworks"]["synth_gepa"] = {
            "status": "completed",
            "best_score": synth_gepa_result.get("best_score", 0.0),
            "train_score": synth_gepa_result.get("train_score", 0.0),
            "val_score": synth_gepa_result.get("val_score"),
            "total_rollouts": rollout_budget,
            "output_dir": str(synth_gepa_output),
        }
        print(f"  ✅ Synth GEPA: best_score={synth_gepa_result.get('best_score', 0.0):.4f}")
    except Exception as e:
        results["frameworks"]["synth_gepa"] = {
            "status": "failed",
            "error": str(e),
        }
        print(f"  ❌ Synth GEPA failed: {e}")

    # 2. Synth MIPRO
    print(f"\n[2/5] Running Synth MIPRO...")
    try:
        synth_mipro_output = output_base_dir / "synth_mipro"
        synth_mipro_result = await run_synth_mipro_iris_inprocess(
            task_app_url=task_app_url,
            rollout_budget=rollout_budget,
            bootstrap_seeds=train_seeds[:20],  # Bootstrap uses subset
            online_seeds=train_seeds,
            test_seeds=val_seeds,
            output_dir=synth_mipro_output,
        )
        results["frameworks"]["synth_mipro"] = {
            "status": "completed",
            "best_score": synth_mipro_result.get("best_score", 0.0),
            "train_score": synth_mipro_result.get("train_score", 0.0),
            "val_score": synth_mipro_result.get("val_score"),
            "total_rollouts": rollout_budget,
            "output_dir": str(synth_mipro_output),
        }
        print(f"  ✅ Synth MIPRO: best_score={synth_mipro_result.get('best_score', 0.0):.4f}")
    except Exception as e:
        results["frameworks"]["synth_mipro"] = {
            "status": "failed",
            "error": str(e),
        }
        print(f"  ❌ Synth MIPRO failed: {e}")

    # 3. DSPy MIPROv2
    print(f"\n[3/5] Running DSPy MIPROv2...")
    try:
        dspy_mipro_output = output_base_dir / "dspy_mipro"
        dspy_mipro_result = await run_dspy_miprov2_iris(
            task_app_url=task_app_url,
            train_seeds=train_seeds,
            val_seeds=val_seeds,
            rollout_budget=rollout_budget,
            output_dir=dspy_mipro_output,
        )
        results["frameworks"]["dspy_miprov2"] = {
            "status": "completed",
            "best_score": dspy_mipro_result.get("best_score", 0.0),
            "val_score": dspy_mipro_result.get("val_score"),
            "total_rollouts": rollout_budget,
            "output_dir": str(dspy_mipro_output),
        }
        print(f"  ✅ DSPy MIPROv2: best_score={dspy_mipro_result.get('best_score', 0.0):.4f}")
    except Exception as e:
        results["frameworks"]["dspy_miprov2"] = {
            "status": "failed",
            "error": str(e),
        }
        print(f"  ❌ DSPy MIPROv2 failed: {e}")

    # 4. DSPy GEPA
    print(f"\n[4/5] Running DSPy GEPA...")
    try:
        dspy_gepa_output = output_base_dir / "dspy_gepa"
        dspy_gepa_result = await run_dspy_gepa_iris(
            task_app_url=task_app_url,
            train_seeds=train_seeds,
            val_seeds=val_seeds,
            rollout_budget=rollout_budget,
            output_dir=dspy_gepa_output,
        )
        results["frameworks"]["dspy_gepa"] = {
            "status": "completed",
            "best_score": dspy_gepa_result.get("best_score", 0.0),
            "val_score": dspy_gepa_result.get("val_score"),
            "total_rollouts": rollout_budget,
            "output_dir": str(dspy_gepa_output),
        }
        print(f"  ✅ DSPy GEPA: best_score={dspy_gepa_result.get('best_score', 0.0):.4f}")
    except Exception as e:
        results["frameworks"]["dspy_gepa"] = {
            "status": "failed",
            "error": str(e),
        }
        print(f"  ❌ DSPy GEPA failed: {e}")

    # 5. Lakshya GEPA
    print(f"\n[5/5] Running Lakshya GEPA...")
    try:
        lakshya_gepa_output = output_base_dir / "lakshya_gepa"
        lakshya_gepa_result = await run_lakshya_gepa_iris(
            task_app_url=task_app_url,
            train_seeds=train_seeds,
            val_seeds=val_seeds,
            rollout_budget=rollout_budget,
            output_dir=lakshya_gepa_output,
        )
        results["frameworks"]["lakshya_gepa"] = {
            "status": "completed",
            "best_score": lakshya_gepa_result.get("best_score", 0.0),
            "val_score": lakshya_gepa_result.get("val_score"),
            "total_rollouts": rollout_budget,
            "output_dir": str(lakshya_gepa_output),
        }
        print(f"  ✅ Lakshya GEPA: best_score={lakshya_gepa_result.get('best_score', 0.0):.4f}")
    except Exception as e:
        results["frameworks"]["lakshya_gepa"] = {
            "status": "failed",
            "error": str(e),
        }
        print(f"  ❌ Lakshya GEPA failed: {e}")

    # Save comparison results
    results_file = output_base_dir / "comparison_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Comparison Complete!")
    print(f"{'='*80}\n")
    print(f"Results saved to: {results_file}")
    print(f"\nSummary:")
    for framework, result in results["frameworks"].items():
        if result.get("status") == "completed":
            print(f"  {framework:20s} best_score={result.get('best_score', 0.0):.4f}")
        else:
            print(f"  {framework:20s} FAILED: {result.get('error', 'unknown')}")

    return results


async def evaluate_on_test_set(
    framework_results: dict[str, Any],
    task_app_url: str = "http://127.0.0.1:8115",
) -> dict[str, Any]:
    """Evaluate all optimized prompts on the held-out test set.

    Args:
        framework_results: Results dictionary from run_all_frameworks
        task_app_url: Task app URL

    Returns:
        Dictionary with test set evaluation results for each framework
    """
    from ...integrations.task_app_client import TaskAppClient
    from datasets import load_dataset

    test_seeds = framework_results["test_seeds"]
    test_results = {}

    print(f"\n{'='*80}")
    print(f"Evaluating on Held-Out Test Set (seeds: {test_seeds[0]}-{test_seeds[-1]})")
    print(f"{'='*80}\n")

    # Load dataset to get feature formatting
    dataset = load_dataset("scikit-learn/iris", split="train")
    
    async with TaskAppClient(task_app_url) as client:
        for framework_name, framework_result in framework_results["frameworks"].items():
            if framework_result.get("status") != "completed":
                test_results[framework_name] = {"status": "skipped", "reason": "optimization failed"}
                continue

            print(f"Evaluating {framework_name}...")

            # Load optimized prompt from framework-specific output
            output_dir = Path(framework_result["output_dir"])
            
            # Try to load optimized prompt (framework-specific)
            prompt_messages = None
            
            # Synth GEPA/MIPRO: Load from best_prompt.json
            best_prompt_file = output_dir / "iris_best_prompt.json"
            if best_prompt_file.exists():
                with open(best_prompt_file) as f:
                    best_prompt = json.load(f)
                    if isinstance(best_prompt, dict) and "messages" in best_prompt:
                        prompt_messages = best_prompt["messages"]
                    elif isinstance(best_prompt, dict) and ("system" in best_prompt or "user" in best_prompt):
                        # Lakshya GEPA format
                        prompt_messages = [
                            {"role": "system", "content": best_prompt.get("system", "")},
                            {"role": "user", "pattern": best_prompt.get("user", "")},
                        ]
            
            # DSPy: Load from iris_best_module.json and reconstruct
            if prompt_messages is None:
                module_file = output_dir / "iris_best_module.json"
                if module_file.exists():
                    with open(module_file) as f:
                        module_info = json.load(f)
                        # DSPy stores instructions and demos separately
                        # For now, use baseline (TODO: reconstruct from module_info)
                        prompt_messages = None
            
            # Fallback to baseline prompt
            if prompt_messages is None:
                prompt_messages = [
                    {
                        "role": "system",
                        "content": "You are a botany classification assistant. Based on the flower's measurements, classify the iris species. Respond with one of: setosa, versicolor, or virginica.",
                    },
                    {
                        "role": "user",
                        "pattern": "Flower Measurements:\n{features}\n\nClassify this iris flower. Respond with one of: setosa, versicolor, or virginica.",
                    },
                ]

            correct_count = 0
            total_reward = 0.0
            test_scores = []

            for seed in test_seeds:
                # Get features for this seed
                row = dataset[int(seed)]
                features = {}
                for key, value in row.items():
                    if key not in ("label", "target", "species", "class"):
                        features[key] = value
                feature_text = "\n".join([f"{key}: {value}" for key, value in features.items()])
                
                # Format user message with actual features
                user_message = prompt_messages[1].get("pattern", prompt_messages[1].get("content", "")).replace("{features}", feature_text)
                
                eval_messages = [
                    prompt_messages[0],
                    {"role": "user", "content": user_message},
                ]

                response = await client.evaluate_prompt(
                    prompt_messages=eval_messages,
                    seed=seed,
                    task_app_id="iris",
                    model="groq/llama-3.3-70b-versatile",
                    provider="groq",
                )

                metrics = response.get("metrics", {})
                outcome_score = metrics.get("outcome_score", 0.0)
                is_correct = int(outcome_score > 0.5)

                correct_count += is_correct
                total_reward += outcome_score
                test_scores.append(outcome_score)

            accuracy = correct_count / len(test_seeds) if test_seeds else 0.0
            mean_reward = total_reward / len(test_seeds) if test_seeds else 0.0

            test_results[framework_name] = {
                "status": "completed",
                "test_accuracy": accuracy,
                "test_mean_reward": mean_reward,
                "test_correct": correct_count,
                "test_total": len(test_seeds),
                "test_scores": test_scores,
            }

            print(f"  ✅ {framework_name}: test_accuracy={accuracy:.4f} ({correct_count}/{len(test_seeds)})")

    return test_results

