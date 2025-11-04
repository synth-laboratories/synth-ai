"""Unit tests for baseline task execution."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from synth_ai.baseline import BaselineConfig, BaselineTaskRunner, DataSplit, TaskResult
from synth_ai.baseline.execution import (
    aggregate_results,
    default_aggregator,
    run_baseline_evaluation,
)


class SimpleTaskRunner(BaselineTaskRunner):
    """Simple task runner for testing."""
    
    async def run_task(self, seed: int) -> TaskResult:
        return TaskResult(
            seed=seed,
            success=True,
            outcome_reward=float(seed),
            total_steps=1,
        )


class TestRunBaselineEvaluation:
    """Test baseline evaluation execution."""
    
    @pytest.mark.asyncio
    async def test_executes_single_task(self):
        """Test execution of single task."""
        task_runner = AsyncMock(return_value=TaskResult(
            seed=0,
            success=True,
            outcome_reward=1.0,
            total_steps=10,
        ))
        
        config = BaselineConfig(
            baseline_id="test",
            name="Test",
            task_runner=task_runner,
            splits={"train": DataSplit(name="train", seeds=[0])},
        )
        
        results = await run_baseline_evaluation(
            config=config,
            seeds=[0],
            policy_config={"model": "gpt-4o-mini"},
            env_config={},
            concurrency=1,
        )
        
        assert len(results) == 1
        assert results[0].seed == 0
        assert results[0].success is True
        assert results[0].outcome_reward == 1.0
        task_runner.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_executes_class_based_runner(self):
        """Test execution with class-based runner."""
        config = BaselineConfig(
            baseline_id="test",
            name="Test",
            task_runner=SimpleTaskRunner,
            splits={"train": DataSplit(name="train", seeds=[0, 1, 2])},
        )
        
        results = await run_baseline_evaluation(
            config=config,
            seeds=[0, 1, 2],
            policy_config={"model": "gpt-4o-mini"},
            env_config={},
            concurrency=3,
        )
        
        assert len(results) == 3
        assert all(r.success for r in results)
        assert results[0].outcome_reward == 0.0
        assert results[1].outcome_reward == 1.0
        assert results[2].outcome_reward == 2.0
    
    @pytest.mark.asyncio
    async def test_executes_multiple_tasks_concurrently(self):
        """Test concurrent execution of multiple tasks."""
        task_runner = AsyncMock(side_effect=[
            TaskResult(seed=i, success=True, outcome_reward=float(i))
            for i in range(5)
        ])
        
        config = BaselineConfig(
            baseline_id="test",
            name="Test",
            task_runner=task_runner,
            splits={"train": DataSplit(name="train", seeds=list(range(5)))},
        )
        
        results = await run_baseline_evaluation(
            config=config,
            seeds=list(range(5)),
            policy_config={"model": "gpt-4o-mini"},
            env_config={},
            concurrency=3,
        )
        
        assert len(results) == 5
        assert all(r.success for r in results)
        assert task_runner.call_count == 5
    
    @pytest.mark.asyncio
    async def test_handles_task_failure(self):
        """Test handling of task failure."""
        task_runner = AsyncMock(side_effect=ValueError("Task failed"))
        
        config = BaselineConfig(
            baseline_id="test",
            name="Test",
            task_runner=task_runner,
            splits={"train": DataSplit(name="train", seeds=[0])},
        )
        
        results = await run_baseline_evaluation(
            config=config,
            seeds=[0],
            policy_config={"model": "gpt-4o-mini"},
            env_config={},
            concurrency=1,
        )
        
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None
        assert "Task failed" in results[0].error


class TestDefaultAggregator:
    """Test default result aggregation."""
    
    def test_aggregates_successful_results(self):
        """Test aggregation of successful results."""
        results = [
            TaskResult(seed=i, success=True, outcome_reward=float(i))
            for i in range(1, 6)  # 1, 2, 3, 4, 5
        ]
        
        metrics = default_aggregator(results)
        
        assert metrics["mean_outcome_reward"] == 3.0  # (1+2+3+4+5)/5
        assert metrics["min_outcome_reward"] == 1.0
        assert metrics["max_outcome_reward"] == 5.0
        assert metrics["success_rate"] == 1.0
        assert metrics["total_tasks"] == 5
        assert metrics["successful_tasks"] == 5
        assert metrics["failed_tasks"] == 0
    
    def test_handles_mixed_success_failure(self):
        """Test aggregation with mixed success/failure."""
        results = [
            TaskResult(seed=0, success=True, outcome_reward=1.0),
            TaskResult(seed=1, success=False, outcome_reward=0.0, error="Failed"),
            TaskResult(seed=2, success=True, outcome_reward=2.0),
        ]
        
        metrics = default_aggregator(results)
        
        assert metrics["mean_outcome_reward"] == 1.5  # (1.0 + 2.0) / 2
        assert metrics["success_rate"] == 2.0 / 3.0
        assert metrics["successful_tasks"] == 2
        assert metrics["failed_tasks"] == 1
    
    def test_handles_all_failures(self):
        """Test aggregation when all tasks fail."""
        results = [
            TaskResult(seed=i, success=False, outcome_reward=0.0, error="Failed")
            for i in range(3)
        ]
        
        metrics = default_aggregator(results)
        
        assert metrics["mean_outcome_reward"] == 0.0
        assert metrics["success_rate"] == 0.0
        assert metrics["successful_tasks"] == 0
        assert metrics["failed_tasks"] == 3


class TestAggregateResults:
    """Test aggregate_results function."""
    
    def test_uses_default_aggregator_when_none_provided(self):
        """Test uses default aggregator when no custom aggregator."""
        config = BaselineConfig(
            baseline_id="test",
            name="Test",
            task_runner=AsyncMock(),
            splits={},
        )
        
        results = [
            TaskResult(seed=i, success=True, outcome_reward=float(i))
            for i in range(5)
        ]
        
        metrics = aggregate_results(config, results)
        
        assert "mean_outcome_reward" in metrics
        assert "success_rate" in metrics
    
    def test_uses_custom_function_aggregator(self):
        """Test uses custom function aggregator."""
        def custom_agg(results):
            return {"custom": len(results)}
        
        config = BaselineConfig(
            baseline_id="test",
            name="Test",
            task_runner=AsyncMock(),
            splits={},
            result_aggregator=custom_agg,
        )
        
        results = [TaskResult(seed=0, success=True, outcome_reward=1.0)]
        metrics = aggregate_results(config, results)
        
        assert metrics == {"custom": 1}
    
    def test_uses_custom_class_aggregator(self):
        """Test uses custom class aggregator."""
        class CustomAggregator:
            def aggregate(self, results):
                return {"custom": len(results)}
        
        config = BaselineConfig(
            baseline_id="test",
            name="Test",
            task_runner=AsyncMock(),
            splits={},
            result_aggregator=CustomAggregator,
        )
        
        results = [TaskResult(seed=0, success=True, outcome_reward=1.0)]
        metrics = aggregate_results(config, results)
        
        assert metrics == {"custom": 1}

