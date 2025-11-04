"""Simple example baseline file for testing."""

from __future__ import annotations

from synth_ai.baseline import BaselineConfig, BaselineTaskRunner, DataSplit, TaskResult


class SimpleTaskRunner(BaselineTaskRunner):
    """Simple task runner that returns success for testing."""
    
    async def run_task(self, seed: int) -> TaskResult:
        """Execute a simple task that always succeeds."""
        return TaskResult(
            seed=seed,
            success=True,
            outcome_reward=1.0,
            total_steps=1,
            metadata={
                "seed": seed,
                "message": f"Task completed successfully for seed {seed}",
            },
        )


# Define baseline config
simple_baseline = BaselineConfig(
    baseline_id="simple",
    name="Simple Baseline",
    description="A simple baseline for testing",
    task_runner=SimpleTaskRunner,
    splits={
        "train": DataSplit(
            name="train",
            seeds=list(range(10)),
            metadata={"difficulty": "easy"},
        ),
        "val": DataSplit(
            name="val",
            seeds=list(range(10, 15)),
            metadata={"difficulty": "medium"},
        ),
        "test": DataSplit(
            name="test",
            seeds=list(range(15, 20)),
            metadata={"difficulty": "hard"},
        ),
    },
    default_policy_config={
        "model": "gpt-4o-mini",
        "temperature": 0.0,
    },
    default_env_config={
        "max_steps": 10,
    },
)

