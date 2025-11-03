"""Integration tests for full baseline execution flow."""

from __future__ import annotations

import pytest
from synth_ai.baseline.discovery import discover_baseline_files, load_baseline_config_from_file
from synth_ai.baseline.execution import run_baseline_evaluation


class TestFullBaselineExecution:
    """Integration tests for complete baseline execution."""
    
    @pytest.mark.asyncio
    async def test_discovers_loads_and_executes_baseline(self, tmp_path):
        """Test full flow: discovery -> loading -> execution."""
        # Create a test baseline file
        baseline_file = tmp_path / "test_baseline.py"
        baseline_file.write_text("""
from synth_ai.baseline import BaselineConfig, BaselineTaskRunner, DataSplit, TaskResult

class SimpleTaskRunner(BaselineTaskRunner):
    async def run_task(self, seed: int) -> TaskResult:
        return TaskResult(
            seed=seed,
            success=True,
            outcome_reward=1.0,
            total_steps=1,
            metadata={"model": self.policy_config.get("model")},
        )

test_baseline = BaselineConfig(
    baseline_id="integration-test",
    name="Integration Test Baseline",
    task_runner=SimpleTaskRunner,
    splits={
        "train": DataSplit(name="train", seeds=[0, 1, 2]),
        "test": DataSplit(name="test", seeds=[3, 4]),
    },
    default_policy_config={"model": "gpt-4o-mini"},
)
""")
        
        # 1. Discovery
        choices = discover_baseline_files([tmp_path])
        assert len(choices) == 1
        assert choices[0].baseline_id == "integration-test"
        
        # 2. Loading
        config = load_baseline_config_from_file("integration-test", baseline_file)
        assert config.baseline_id == "integration-test"
        assert config.name == "Integration Test Baseline"
        
        # 3. Execution
        results = await run_baseline_evaluation(
            config=config,
            seeds=[0, 1],
            policy_config={"model": "gpt-4o-mini"},
            env_config={},
            concurrency=2,
        )
        
        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.outcome_reward == 1.0 for r in results)

