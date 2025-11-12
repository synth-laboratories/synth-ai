#!/usr/bin/env python3
"""Run GEPA with judge support on Iris.

This example demonstrates first-class judge integration for automatic reward scoring.

Usage:
    python run_gepa_iris_with_judge.py --task-app-url http://127.0.0.1:8115 --rollout-budget 20
"""

import asyncio
import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Suppress verbose HTTP logs
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

# Add backend to path
script_path = Path(__file__).resolve()
monorepo_root = script_path.parent.parent.parent.parent.parent.parent.parent / "monorepo"
backend_path = monorepo_root / "backend"

if not backend_path.exists():
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        candidate = parent / "monorepo" / "backend"
        if candidate.exists():
            backend_path = candidate
            break

if not backend_path.exists():
    backend_path_env = os.getenv("MONOREPO_BACKEND_PATH")
    if backend_path_env:
        backend_path = Path(backend_path_env)

if backend_path.exists():
    monorepo_backend = backend_path.parent
    sys.path.insert(0, str(monorepo_backend))
    print(f"✓ Found backend at: {backend_path}")
else:
    print(f"ERROR: Could not find monorepo backend.")
    sys.exit(1)

from backend.app.routes.prompt_learning.algorithm.gepa import (
    GEPAOptimizer,
    GEPAConfig,
)
from backend.app.routes.prompt_learning.core.runtime import LocalRuntime
from backend.app.routes.prompt_learning.core.patterns import PromptPattern, MessagePattern
from backend.app.routes.prompt_learning.core.judge_config import JudgeConfig


def create_iris_gepa_config_with_judge(
    task_app_url: str,
    rollout_budget: int = 20,
    judge_enabled: bool = True,
    judge_backend_base: Optional[str] = None,
    judge_reward_source: str = "judge",
) -> GEPAConfig:
    """Create GEPA config for Iris classification with judge support."""
    
    # Auto-scale GEPA parameters based on budget
    initial_population_size = max(2, min(5, rollout_budget // 10))
    num_generations = max(2, min(5, rollout_budget // (initial_population_size * 2)))
    
    # Get API key
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("SYNTH_API_KEY")
    if not task_app_api_key:
        raise ValueError("ENVIRONMENT_API_KEY or SYNTH_API_KEY must be set")
    
    # Get Groq API key for policy
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY must be set")
    
    # Create initial pattern
    initial_pattern = PromptPattern(
        messages=[
            MessagePattern(
                role="system",
                pattern="You are a helpful assistant that classifies iris flowers into one of three species: setosa, versicolor, or virginica.",
                order=0,
            ),
            MessagePattern(
                role="user",
                pattern="Classify this iris flower with sepal length {sepal_length}, sepal width {sepal_width}, petal length {petal_length}, petal width {petal_width}.",
                order=1,
            ),
        ],
    )
    
    # Create judge config (optional, first-class)
    judge_config = None
    if judge_enabled:
        judge_backend_base = judge_backend_base or os.getenv("JUDGE_BACKEND_BASE", "https://judge.synth.ai")
        judge_api_key_env = os.getenv("JUDGE_API_KEY_ENV", "SYNTH_API_KEY")
        
        judge_config = JudgeConfig(
            enabled=True,
            reward_source=judge_reward_source,  # "task_app", "judge", or "fused"
            backend_base=judge_backend_base,
            backend_api_key_env=judge_api_key_env,
            backend_provider="groq",
            backend_model="llama-3.3-70b-versatile",
            backend_rubric_id="iris-rubric-v1",  # You'll need to create this rubric
            backend_event_enabled=True,
            backend_outcome_enabled=True,
            concurrency=8,
            timeout=60.0,
            weight_env=1.0,
            weight_event=0.0,
            weight_outcome=0.0,
            # Optional: spec support
            # spec_path="path/to/iris_spec.json",
            # spec_max_tokens=5000,
        )
    
    config = GEPAConfig(
        task_app_url=task_app_url,
        task_app_api_key=task_app_api_key,
        env_name="iris",
        rollout_budget=rollout_budget,
        initial_population_size=initial_population_size,
        num_generations=num_generations,
        mutation_rate=0.3,
        crossover_rate=0.5,
        minibatch_size=min(4, rollout_budget // 5),
        pareto_set_size=min(8, rollout_budget // 3),
        feedback_fraction=0.5,
        initial_pattern=initial_pattern,
        policy_config={
            "model": "openai/gpt-oss-20b",
            "provider": "groq",
            "inference_url": "https://api.groq.com/openai/v1",
            "api_key": groq_api_key,
        },
        max_concurrent_rollouts=5,
        judge=judge_config,  # Pass judge config
    )
    
    return config


async def main():
    parser = argparse.ArgumentParser(description="Run GEPA with judge support on Iris")
    parser.add_argument(
        "--task-app-url",
        type=str,
        default="http://127.0.0.1:8115",
        help="Task app URL",
    )
    parser.add_argument(
        "--rollout-budget",
        type=int,
        default=20,
        help="Maximum number of rollouts",
    )
    parser.add_argument(
        "--judge-enabled",
        action="store_true",
        default=True,
        help="Enable judge-based scoring",
    )
    parser.add_argument(
        "--judge-backend-base",
        type=str,
        default=None,
        help="Judge backend base URL (defaults to JUDGE_BACKEND_BASE env var or https://judge.synth.ai)",
    )
    parser.add_argument(
        "--judge-reward-source",
        type=str,
        choices=["task_app", "judge", "fused"],
        default="judge",
        help="Reward source: task_app (use task app rewards), judge (use judge rewards), or fused (weighted combination)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable judge (use task app rewards only)",
    )
    
    args = parser.parse_args()
    
    judge_enabled = args.judge_enabled and not args.no_judge
    
    print(f"\n{'='*60}")
    print(f"GEPA with Judge Support - Iris Classification")
    print(f"{'='*60}")
    print(f"Task App URL: {args.task_app_url}")
    print(f"Rollout Budget: {args.rollout_budget}")
    print(f"Judge Enabled: {judge_enabled}")
    if judge_enabled:
        print(f"Judge Reward Source: {args.judge_reward_source}")
        print(f"Judge Backend: {args.judge_backend_base or os.getenv('JUDGE_BACKEND_BASE', 'https://judge.synth.ai')}")
    print(f"{'='*60}\n")
    
    # Create config
    config = create_iris_gepa_config_with_judge(
        task_app_url=args.task_app_url,
        rollout_budget=args.rollout_budget,
        judge_enabled=judge_enabled,
        judge_backend_base=args.judge_backend_base,
        judge_reward_source=args.judge_reward_source,
    )
    
    # Create runtime
    runtime = LocalRuntime(
        interceptor_host="localhost",
        interceptor_port=8765,
        task_app_host="localhost",
        task_app_port=8115,
    )
    
    # Create optimizer
    optimizer = GEPAOptimizer(config=config, runtime=runtime)
    
    # Run optimization
    train_seeds = list(range(0, min(20, args.rollout_budget)))
    print(f"Training on seeds: {train_seeds[:10]}..." if len(train_seeds) > 10 else f"Training on seeds: {train_seeds}")
    
    try:
        best_template, best_score = await optimizer.optimize(
            initial_pattern=config.initial_pattern,
            train_seeds=train_seeds,
        )
        
        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"{'='*60}")
        print(f"Best Score: {best_score:.3f} ({best_score*100:.1f}%)")
        print(f"\nBest Prompt:")
        for section in best_template.sections:
            print(f"  [{section.role.value if hasattr(section.role, 'value') else section.role}]")
            print(f"  {section.content}")
            print()
        
        if judge_enabled:
            print(f"Note: Scores were computed using judge (reward_source={args.judge_reward_source})")
            print(f"      Task app rewards were {'fused' if args.judge_reward_source == 'fused' else 'replaced'} with judge scores")
        
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
    except Exception as e:
        print(f"\n\nError during optimization: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Stop interceptor
        await optimizer._stop_interceptor()


if __name__ == "__main__":
    asyncio.run(main())

