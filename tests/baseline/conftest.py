"""Pytest fixtures for baseline tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from synth_ai.sdk.baseline import BaselineConfig, DataSplit, TaskResult


@pytest.fixture
def simple_task_runner():
    """Simple task runner that returns success."""
    async def runner(seed: int, policy_config: dict, env_config: dict):
        return TaskResult(
            seed=seed,
            success=True,
            outcome_reward=1.0,
            total_steps=1,
        )
    return runner


@pytest.fixture
def simple_baseline_config(simple_task_runner):
    """Simple baseline config for testing."""
    return BaselineConfig(
        baseline_id="test",
        name="Test Baseline",
        task_runner=simple_task_runner,
        splits={
            "train": DataSplit(name="train", seeds=[0, 1, 2]),
            "val": DataSplit(name="val", seeds=[3, 4]),
            "test": DataSplit(name="test", seeds=[5, 6, 7, 8, 9]),
        },
        default_policy_config={"model": "gpt-4o-mini"},
    )


@pytest.fixture
def mock_inference_client():
    """Mock inference client."""
    client = AsyncMock()
    client.chat.return_value = {
        "content": "test response",
        "tool_calls": [],
    }
    return client

