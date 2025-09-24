from uuid import uuid4

import pytest
from synth_ai.environments.examples.bandit import (
    BanditEngine,
    BanditEnvironment,
    BanditTaskInstance,
    BanditTaskInstanceMetadata,
)
from synth_ai.environments.tasks.core import Impetus, Intent

pytestmark = pytest.mark.asyncio


def _make_bandit_instance(
    *,
    bandit_type: str = "bernoulli",
    arm_probabilities: list[float] | None = None,
    arm_means: list[float] | None = None,
    arm_stds: list[float] | None = None,
    max_steps: int = 3,
    seed: int | None = 1,
) -> BanditTaskInstance:
    metadata = BanditTaskInstanceMetadata(
        name="test-bandit",
        bandit_type=bandit_type,
        arm_probabilities=arm_probabilities,
        arm_means=arm_means,
        arm_stds=arm_stds,
        max_steps=max_steps,
        seed=seed,
    )
    instance = BanditTaskInstance(
        id=uuid4(),
        impetus=Impetus(
            instructions="Interact with the bandit by choosing an arm index each step."
        ),
        intent=Intent(
            rubric={"goal": "Maximize cumulative reward"},
            gold_trajectories=None,
            gold_state_diff={},
        ),
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
    )
    return instance


async def test_bandit_engine_step_updates_counts_and_status():
    instance = _make_bandit_instance(
        arm_probabilities=[0.0, 1.0],
        max_steps=2,
        seed=123,
    )
    engine = BanditEngine(instance)

    priv, pub = await engine._reset_engine()
    assert pub.arm_count == 2
    assert priv.total_reward == pytest.approx(0.0)
    assert pub.status == "in_progress"

    priv, pub = await engine._step_engine(1)
    assert pub.step_count == 1
    assert pub.last_arm == 1
    assert priv.reward_last == pytest.approx(1.0)
    assert engine.arm_pull_counts == [0, 1]
    assert not pub.terminated

    priv, pub = await engine._step_engine(0)
    assert pub.step_count == 2
    assert engine.terminated is True
    assert pub.terminated is True
    assert pub.status == "completed"
    assert priv.total_reward == pytest.approx(1.0)


async def test_bandit_engine_serialize_deserialize_roundtrip():
    instance = _make_bandit_instance(arm_probabilities=[0.2, 0.8], max_steps=5, seed=7)
    engine = BanditEngine(instance)
    await engine._reset_engine()
    await engine._step_engine(1)

    snapshot = await engine._serialize_engine()
    restored = await BanditEngine._deserialize_engine(snapshot)

    # State properties should match after serialization round-trip
    assert restored.bandit_type == engine.bandit_type
    assert restored.arm_count == engine.arm_count
    assert restored.step_count == engine.step_count
    assert restored.total_reward == engine.total_reward
    assert restored.last_arm == engine.last_arm
    assert restored.reward_history == engine.reward_history
    assert restored.arm_pull_counts == engine.arm_pull_counts

    # RNG state restored -> subsequent rewards match
    priv_a, pub_a = await engine._step_engine(1)
    priv_b, pub_b = await restored._step_engine(1)
    assert priv_a.reward_last == pytest.approx(priv_b.reward_last)
    assert pub_a.last_reward == pytest.approx(pub_b.last_reward)


async def test_bandit_environment_supports_multiple_call_formats():
    instance = _make_bandit_instance(arm_probabilities=[0.5, 0.5], max_steps=3)
    env = BanditEnvironment(instance)
    await env.initialize()

    # Direct dict with tool key
    obs = await env.step({"tool": "pull_arm", "args": {"arm_index": 1}})
    assert obs["steps_taken"] == 1
    assert obs["last_arm"] == 1

    # OpenAI-style function call wrapper
    obs = await env.step(
        {
            "function": {
                "name": "pull_arm",
                "arguments": {"arm": 0},
            }
        }
    )
    assert obs["steps_taken"] == 2
    assert obs["last_arm"] == 0

    # Raw integer should also work
    obs = await env.step(1)
    assert obs["steps_taken"] == 3
    assert obs["terminated"] in {False, True}


async def test_bandit_environment_reports_errors_after_termination():
    instance = _make_bandit_instance(arm_probabilities=[1.0], max_steps=1)
    env = BanditEnvironment(instance)
    await env.initialize()
    await env.step({"arm": 0})

    obs = await env.step({"arm": 0})
    assert obs["error"]
    assert "terminated" in obs["error"].lower()


async def test_bandit_environment_validation_rejects_invalid_inputs():
    instance = _make_bandit_instance(arm_probabilities=[0.5], max_steps=2)
    env = BanditEnvironment(instance)

    with pytest.raises(ValueError, match="Unknown tool"):
        env.validate_tool_calls({"tool": "invalid", "args": {"arm": 0}})

    with pytest.raises(ValueError, match="Missing required 'arm'"):
        env.validate_tool_calls({"tool": "pull_arm", "args": {}})

    with pytest.raises(ValueError, match="must be an integer"):
        env.validate_tool_calls({"tool": "pull_arm", "args": {"arm": "left"}})

    with pytest.raises(ValueError, match="non-negative"):
        env.validate_tool_calls({"tool": "pull_arm", "args": {"arm": -1}})
