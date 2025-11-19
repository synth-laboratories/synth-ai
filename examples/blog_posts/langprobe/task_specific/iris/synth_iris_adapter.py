"""Synth GEPA and MIPRO adapters for Iris classification."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from ...integrations.synth_gepa_adapter_inprocess import SynthGEPAAdapterInProcess
from ...integrations.synth_mipro_adapter_inprocess import SynthMIPROAdapterInProcess

load_dotenv()


async def run_synth_gepa_iris_inprocess(
    task_app_url: str = "http://127.0.0.1:8115",
    train_seeds: Optional[list[int]] = None,
    val_seeds: Optional[list[int]] = None,
    rollout_budget: int = 400,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run Synth GEPA on Iris benchmark using in-process API.

    Args:
        task_app_url: Task app URL (local)
        train_seeds: Training seeds (default: auto-scale based on budget)
        val_seeds: Validation seeds for held-out evaluation (default: None)
        rollout_budget: Rollout budget (default: 400)
        output_dir: Output directory (default: results/synth_gepa/)

    Returns:
        Results dictionary
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "synth_gepa"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initial prompt messages (from iris_task_app.py)
    initial_prompt_messages = [
        {
            "role": "system",
            "pattern": (
                "You are a botany classification assistant. Based on the flower's measurements, "
                "classify the iris species. Respond with one of: setosa, versicolor, or virginica."
            ),
        },
        {
            "role": "user",
            "pattern": (
                "Flower Measurements:\n{features}\n\n"
                "Classify this iris flower. Respond with one of: setosa, versicolor, or virginica."
            ),
        },
    ]

    # Auto-determine validation seeds if not provided
    if val_seeds is None and train_seeds:
        # Use seeds after the max train seed for validation
        max_train_seed = max(train_seeds) if train_seeds else 99
        val_seeds = list(range(max_train_seed + 1, max_train_seed + 51))  # 50 validation seeds
    
    # Create adapter
    adapter = SynthGEPAAdapterInProcess(
        task_app_url=task_app_url,
        task_app_id="iris",
        initial_prompt_messages=initial_prompt_messages,
        rollout_budget=rollout_budget,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
    )

    # Run optimization
    results = await adapter.optimize()

    # Save results
    adapter.save_results(output_dir)

    return {
        "best_score": results.train_score,
        "train_score": results.train_score,
        "val_score": results.val_score,
        "total_rollouts": results.total_rollouts,
        "output_dir": str(output_dir),
    }


async def run_synth_mipro_iris_inprocess(
    task_app_url: str = "http://127.0.0.1:8115",
    bootstrap_seeds: Optional[list[int]] = None,
    online_seeds: Optional[list[int]] = None,
    test_seeds: Optional[list[int]] = None,
    rollout_budget: int = 400,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run Synth MIPRO on Iris benchmark using in-process API.

    Args:
        task_app_url: Task app URL (local)
        bootstrap_seeds: Bootstrap seeds (default: auto-scale based on budget)
        online_seeds: Online pool seeds (default: auto-scale based on budget)
        test_seeds: Test seeds for held-out evaluation (default: None)
        rollout_budget: Rollout budget (default: 400)
        output_dir: Output directory (default: results/synth_mipro/)

    Returns:
        Results dictionary
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "synth_mipro"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initial prompt messages (from iris_task_app.py)
    initial_prompt_messages = [
        {
            "role": "system",
            "pattern": (
                "You are a botany classification assistant. Based on the flower's measurements, "
                "classify the iris species. Respond with one of: setosa, versicolor, or virginica."
            ),
        },
        {
            "role": "user",
            "pattern": (
                "Flower Measurements:\n{features}\n\n"
                "Classify this iris flower. Respond with one of: setosa, versicolor, or virginica."
            ),
        },
    ]
    
    # Create adapter
    adapter = SynthMIPROAdapterInProcess(
        task_app_url=task_app_url,
        task_app_id="iris",
        initial_prompt_messages=initial_prompt_messages,
        rollout_budget=rollout_budget,
        bootstrap_seeds=bootstrap_seeds,
        online_seeds=online_seeds,
        test_seeds=test_seeds,
    )

    # Run optimization
    results = await adapter.optimize()

    # Save results
    adapter.save_results(output_dir)

    return {
        "best_score": results.train_score,
        "train_score": results.train_score,
        "val_score": results.test_score,  # MIPRO uses test_score for validation
        "total_rollouts": results.total_trials,
        "output_dir": str(output_dir),
    }

