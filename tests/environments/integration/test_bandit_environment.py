from __future__ import annotations

from uuid import uuid4

import pytest
from synth_ai.environments.examples.bandit import (
    BanditEnvironment,
    BanditTaskInstance,
    BanditTaskInstanceMetadata,
    create_bandit_taskset,
)
from synth_ai.environments.tasks.core import Impetus, Intent

pytestmark = pytest.mark.asyncio


def _make_instance(
    *,
    bandit_type: str = "bernoulli",
    arm_probabilities: list[float] | None = None,
    arm_means: list[float] | None = None,
    arm_stds: list[float] | None = None,
    max_steps: int = 3,
    seed: int | None = 5,
) -> BanditTaskInstance:
    metadata = BanditTaskInstanceMetadata(
        name="integration-bandit",
        bandit_type=bandit_type,
        arm_probabilities=arm_probabilities,
        arm_means=arm_means,
        arm_stds=arm_stds,
        max_steps=max_steps,
        seed=seed,
    )
    return BanditTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="Select an arm each step to gather reward."),
        intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
    )


async def test_bandit_taskset_default_configs():
    taskset = await create_bandit_taskset()
    assert len(taskset.instances) >= 3
    # Ensure split metadata populated
    assert taskset.split_info._is_split_defined is True
    assert taskset.split_info.val_instance_ids
    assert taskset.split_info.test_instance_ids


async def test_bandit_environment_episode_completes_after_max_steps():
    instance = _make_instance(arm_probabilities=[1.0, 0.5], max_steps=3, seed=11)
    env = BanditEnvironment(instance)

    obs = await env.initialize()
    assert obs["steps_taken"] == 0
    assert obs["terminated"] is False

    final_obs = obs
    for step in range(instance.metadata.max_steps):
        final_obs = await env.step({"arm": 0})
        assert final_obs["steps_taken"] == step + 1

    assert final_obs["terminated"] is True
    assert final_obs["status"] == "completed"
    assert final_obs["steps_taken"] == instance.metadata.max_steps
    assert final_obs["cumulative_reward"] >= instance.metadata.max_steps - 1


async def test_bandit_environment_checkpoint_and_serialize_roundtrip():
    instance = _make_instance(arm_probabilities=[0.9, 0.1], max_steps=4, seed=3)
    env = BanditEnvironment(instance)
    await env.initialize()
    await env.step({"arm": 0})

    checkpoint_obs = await env.checkpoint()
    assert checkpoint_obs["total_reward"] >= 0.0
    assert checkpoint_obs["steps_taken"] == 1

    snapshot = await env._serialize_engine()
    restored = await BanditEnvironment._deserialize_engine(snapshot, instance)

    restored_obs = await restored.step({"arm": 0})
    assert restored_obs["steps_taken"] == 2
    assert isinstance(restored_obs["cumulative_reward"], float)
