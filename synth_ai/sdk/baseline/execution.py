"""Execution engine for baseline evaluations."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from synth_ai.sdk.baseline.config import (
    BaselineConfig,
    BaselineTaskRunner,
    TaskResult,
)


def default_aggregator(results: List[TaskResult]) -> Dict[str, Any]:
    """Default result aggregation function.
    
    Computes mean, std, min, max, success rate, and other basic metrics.
    
    Args:
        results: List of TaskResult objects from all seeds
    
    Returns:
        Dict with aggregate metrics
    """
    successful_results = [r for r in results if r.success]
    outcome_rewards = [r.outcome_reward for r in successful_results]
    
    if not outcome_rewards:
        return {
            "mean_outcome_reward": 0.0,
            "std_outcome_reward": 0.0,
            "min_outcome_reward": 0.0,
            "max_outcome_reward": 0.0,
            "success_rate": 0.0,
            "total_tasks": len(results),
            "successful_tasks": 0,
            "failed_tasks": len(results),
        }
    
    mean_reward = sum(outcome_rewards) / len(outcome_rewards)
    
    # Calculate standard deviation
    variance = sum((x - mean_reward) ** 2 for x in outcome_rewards) / len(outcome_rewards)
    std_reward = variance ** 0.5
    
    return {
        "mean_outcome_reward": mean_reward,
        "std_outcome_reward": std_reward,
        "min_outcome_reward": min(outcome_rewards),
        "max_outcome_reward": max(outcome_rewards),
        "success_rate": len(successful_results) / len(results),
        "total_tasks": len(results),
        "successful_tasks": len(successful_results),
        "failed_tasks": len(results) - len(successful_results),
    }


def _is_class_based_runner(task_runner: Any) -> bool:
    """Check if task_runner is a class (not a function)."""
    return (
        isinstance(task_runner, type)
        and issubclass(task_runner, BaselineTaskRunner)
    )


async def run_baseline_evaluation(
    config: BaselineConfig,
    seeds: List[int],
    policy_config: Dict[str, Any],
    env_config: Dict[str, Any],
    concurrency: int = 4,
) -> List[TaskResult]:
    """Run baseline evaluation for given seeds.
    
    Args:
        config: BaselineConfig instance
        seeds: List of seeds to evaluate
        policy_config: Policy configuration (merged from defaults + overrides)
        env_config: Environment configuration (merged from defaults + overrides)
        concurrency: Maximum concurrent task executions
    
    Returns:
        List of TaskResult objects, one per seed
    """
    # Determine if we're using class-based or function-based runner
    is_class_based = _is_class_based_runner(config.task_runner)
    
    # Instantiate runner if class-based
    runner_instance: Optional[BaselineTaskRunner] = None
    if is_class_based:
        # task_runner is a class - instantiate with policy_config and env_config
        # as documented in BaselineConfig and BaselineTaskRunner
        runner_instance = config.task_runner(policy_config, env_config)  # type: ignore[call-arg]
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)
    
    async def run_task(seed: int) -> TaskResult:
        """Execute a single task with error handling."""
        async with semaphore:
            try:
                if is_class_based and runner_instance:
                    # Class-based: call run_task method
                    return await runner_instance.run_task(seed)
                else:
                    # Function-based: call function directly
                    task_runner_fn = config.task_runner
                    if callable(task_runner_fn):
                        result = task_runner_fn(seed, policy_config, env_config)  # type: ignore[call-arg]
                        # Handle both sync and async functions
                        if hasattr(result, "__await__"):
                            return await result
                        return result
                    raise RuntimeError("task_runner is not callable")
            except Exception as exc:
                # Return error result
                return TaskResult(
                    seed=seed,
                    success=False,
                    outcome_reward=0.0,
                    error=str(exc),
                )
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*[run_task(seed) for seed in seeds])
    return list(results)


def aggregate_results(
    config: BaselineConfig,
    results: List[TaskResult],
) -> Dict[str, Any]:
    """Aggregate results using custom aggregator or default.
    
    Args:
        config: BaselineConfig instance
        results: List of TaskResult objects
    
    Returns:
        Dict with aggregate metrics
    """
    if config.result_aggregator is None:
        return default_aggregator(results)
    
    # Check if aggregator is a class or function
    if isinstance(config.result_aggregator, type):
        # Class-based: instantiate and call aggregate()
        aggregator_instance = config.result_aggregator()
        return aggregator_instance.aggregate(results)
    else:
        # Function-based: call directly
        return config.result_aggregator(results)

