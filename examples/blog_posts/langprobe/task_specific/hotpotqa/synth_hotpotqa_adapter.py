"""Synth GEPA and MIPRO adapters for HotpotQA multi-hop QA."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from ...integrations.synth_gepa_adapter_inprocess import SynthGEPAAdapterInProcess
from ...integrations.synth_mipro_adapter_inprocess import SynthMIPROAdapterInProcess

load_dotenv()


async def run_synth_gepa_hotpotqa_inprocess(
    task_app_url: str = "http://127.0.0.1:8110",
    train_seeds: Optional[list[int]] = None,
    val_seeds: Optional[list[int]] = None,
    rollout_budget: int = 400,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run Synth GEPA on HotpotQA benchmark using in-process API.

    Args:
        task_app_url: Task app URL (default: http://127.0.0.1:8110)
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

    # Initial prompt messages (from hotpotqa_task_app.py - must match exactly!)
    initial_prompt_messages = [
        {
            "role": "system",
            "pattern": (
                "You are a research assistant that answers multi-hop questions. "
                "Read the passages carefully and respond in the format:\n"
                "Answer: <short answer>\nSupport: <brief justification citing passages>."
            ),
        },
        {
            "role": "user",
            "pattern": "Question: {question}\n\nPassages:\n{context}\n\nProvide the final answer.",
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
        task_app_id="hotpotqa",
        initial_prompt_messages=initial_prompt_messages,
        rollout_budget=rollout_budget,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
    )

    # Run optimization
    results = await adapter.optimize()

    # Save results
    adapter.save_results(output_dir)

    # Print summary
    print(f"\n{'='*80}")
    print(f"✅ Synth GEPA Optimization Complete!")
    print(f"{'='*80}")
    # optimize() returns a dict, not a GEPAResult object
    print(f"Best score: {results.get('best_score', 0.0):.4f}")
    if results.get('val_score') is not None:
        print(f"Val score: {results.get('val_score'):.4f}")
    print(f"Total rollouts: {results.get('total_rollouts', 0)}")
    print(f"Results saved to: {output_dir}")
    
    if results.get('best_prompt'):
        print(f"\nBest prompt saved to: {output_dir / 'hotpotqa_best_prompt.json'}")
    
    return results


async def run_synth_mipro_hotpotqa_inprocess(
    task_app_url: str = "http://127.0.0.1:8110",
    bootstrap_seeds: Optional[list[int]] = None,
    online_seeds: Optional[list[int]] = None,
    test_seeds: Optional[list[int]] = None,
    rollout_budget: int = 400,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run Synth MIPRO on HotpotQA benchmark using in-process API.

    Args:
        task_app_url: Task app URL (default: http://127.0.0.1:8110)
        bootstrap_seeds: Bootstrap seeds for initial evaluation (default: auto-scale)
        online_seeds: Online seeds for optimization (default: auto-scale)
        test_seeds: Test seeds for held-out evaluation (default: None)
        rollout_budget: Rollout budget (default: 400)
        output_dir: Output directory (default: results/synth_mipro/)

    Returns:
        Results dictionary
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "synth_mipro"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initial prompt messages (from hotpotqa_task_app.py - must match exactly!)
    initial_prompt_messages = [
        {
            "role": "system",
            "pattern": (
                "You are a research assistant that answers multi-hop questions. "
                "Read the passages carefully and respond in the format:\n"
                "Answer: <short answer>\nSupport: <brief justification citing passages>."
            ),
        },
        {
            "role": "user",
            "pattern": "Question: {question}\n\nPassages:\n{context}\n\nProvide the final answer.",
        },
    ]

    # Auto-scale seeds if not provided
    if bootstrap_seeds is None:
        if rollout_budget < 50:
            bootstrap_seeds = list(range(5))
        elif rollout_budget < 100:
            bootstrap_seeds = list(range(10))
        else:
            bootstrap_seeds = list(range(20))
    
    if online_seeds is None:
        if rollout_budget < 50:
            online_seeds = list(range(20))
        elif rollout_budget < 100:
            online_seeds = list(range(50))
        else:
            online_seeds = list(range(100))
    
    # Create adapter
    adapter = SynthMIPROAdapterInProcess(
        task_app_url=task_app_url,
        task_app_id="hotpotqa",
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

    # Print summary
    print(f"\n{'='*80}")
    print(f"✅ Synth MIPRO Optimization Complete!")
    print(f"{'='*80}")
    # optimize() returns a dict, not a MIPROResult object
    print(f"Best score: {results.get('best_score', 0.0):.4f}")
    if results.get('test_score') is not None:
        print(f"Test score: {results.get('test_score'):.4f}")
    print(f"Total trials: {results.get('total_trials', 0)}")
    print(f"Results saved to: {output_dir}")
    
    if results.get('best_prompt'):
        print(f"\nBest prompt saved to: {output_dir / 'hotpotqa_best_prompt.json'}")
    
    return results

