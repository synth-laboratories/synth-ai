"""Unit tests for MiniGrid environment."""

import asyncio
import pytest
import json

from synth_ai.environments.examples.minigrid.environment import (
    MiniGridEnvironment,
    MiniGridInteractTool,
)
from synth_ai.environments.examples.minigrid.taskset import DEFAULT_MINIGRID_TASK
from synth_ai.environments.environment.tools import EnvToolCall


@pytest.mark.asyncio
async def test_environment_initialization():
    """Test environment initialization."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)

    # Check initial setup
    assert env.name == "MiniGridEnvironment"
    assert env.task_instance == DEFAULT_MINIGRID_TASK

    # Initialize
    obs = await env.initialize()

    # Check observation
    assert isinstance(obs, dict)
    assert "observation" in obs
    assert "terminated" in obs
    assert "total_reward" in obs
    assert obs["terminated"] is False
    assert obs["total_reward"] == 0.0

    # Check observation text
    obs_text = obs["observation"]
    assert "Mission:" in obs_text
    assert "Grid:" in obs_text
    assert "Legend:" in obs_text


@pytest.mark.asyncio
async def test_environment_step():
    """Test environment step functionality."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    await env.initialize()

    # Test forward action
    tool_call = {"name": "minigrid_act", "args": {"action": "forward"}}
    obs = await env.step(tool_call)

    assert isinstance(obs, dict)
    assert "observation" in obs
    assert "reward_last" in obs
    assert obs["reward_last"] == -0.01  # Step penalty

    # Test turn action
    tool_call = {"name": "minigrid_act", "args": {"action": "right"}}
    obs = await env.step(tool_call)
    assert obs["reward_last"] == -0.01


@pytest.mark.asyncio
async def test_tool_validation():
    """Test tool call validation."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)

    # Test different input formats
    # Format 1: Dict with name and args
    call1 = env.validate_tool_calls({"name": "minigrid_act", "args": {"action": "forward"}})
    assert call1.tool == "minigrid_act"
    assert call1.args == {"action": "forward"}

    # Format 2: Dict with tool and args
    call2 = env.validate_tool_calls({"tool": "minigrid_act", "args": {"action": "left"}})
    assert call2.tool == "minigrid_act"
    assert call2.args == {"action": "left"}

    # Format 3: List of tool calls
    call3 = env.validate_tool_calls([{"name": "minigrid_act", "args": {"action": "right"}}])
    assert call3.tool == "minigrid_act"
    assert call3.args == {"action": "right"}

    # Format 4: With input field
    call4 = env.validate_tool_calls({"name": "minigrid_act", "input": {"action": "pickup"}})
    assert call4.tool == "minigrid_act"
    assert call4.args == {"action": "pickup"}

    # Format 5: With string input
    call5 = env.validate_tool_calls({"name": "minigrid_act", "input": '{"action": "drop"}'})
    assert call5.tool == "minigrid_act"
    assert call5.args == {"action": "drop"}

    # Test invalid tool name
    with pytest.raises(ValueError, match="Unknown tool"):
        env.validate_tool_calls({"name": "invalid_tool", "args": {"action": "forward"}})


@pytest.mark.asyncio
async def test_invalid_actions():
    """Test invalid action handling."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    await env.initialize()

    # Test invalid action
    tool_call = {"name": "minigrid_act", "args": {"action": "invalid_action"}}
    obs = await env.step(tool_call)

    # Should have error in observation
    assert "error" in obs
    assert "Invalid action" in obs["error"]


@pytest.mark.asyncio
async def test_checkpoint_and_terminate():
    """Test checkpoint and terminate functionality."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    await env.initialize()

    # Take some steps
    await env.step({"name": "minigrid_act", "args": {"action": "forward"}})
    await env.step({"name": "minigrid_act", "args": {"action": "right"}})

    # Checkpoint
    checkpoint = await env.checkpoint()
    assert isinstance(checkpoint, dict)
    assert "mission" in checkpoint
    assert "total_steps" in checkpoint
    assert "total_reward" in checkpoint

    # Terminate
    final_obs = await env.terminate()
    assert isinstance(final_obs, dict)
    assert "mission" in final_obs
    assert "final_position" in final_obs


@pytest.mark.asyncio
async def test_interact_tool():
    """Test the interact tool directly."""
    from synth_ai.environments.examples.minigrid.engine import MiniGridEngine

    engine = MiniGridEngine(DEFAULT_MINIGRID_TASK)
    await engine._reset_engine()

    tool = MiniGridInteractTool(engine)

    # Test valid action
    call = EnvToolCall(tool="minigrid_act", args={"action": "forward"})
    result = await tool(call)

    assert result.ok is True
    assert "message" in result.payload
    assert "public_state" in result.payload
    assert "private_state" in result.payload

    # Test invalid action
    call = EnvToolCall(tool="minigrid_act", args={"action": "invalid"})
    result = await tool(call)

    assert result.ok is False
    assert "Invalid action" in result.error


@pytest.mark.asyncio
async def test_observation_callables():
    """Test observation callable functionality."""
    from synth_ai.environments.examples.minigrid.engine import (
        MiniGridObservationCallable,
        MiniGridCheckpointObservationCallable,
    )

    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    await env.initialize()

    # Get states
    priv, pub = env.engine.get_current_states_for_observation()

    # Test step observation
    step_obs_callable = MiniGridObservationCallable()
    step_obs = await step_obs_callable.get_observation(pub, priv)

    assert "observation" in step_obs
    assert "terminated" in step_obs
    assert "reward_last" in step_obs

    # Test checkpoint observation
    ckpt_obs_callable = MiniGridCheckpointObservationCallable()
    ckpt_obs = await ckpt_obs_callable.get_observation(pub, priv)

    assert "mission" in ckpt_obs
    assert "final_position" in ckpt_obs
    assert "total_steps" in ckpt_obs


@pytest.mark.asyncio
async def test_full_episode():
    """Test a full episode from start to finish."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)

    # Initialize
    obs = await env.initialize()
    assert obs["terminated"] is False

    # Take multiple steps
    actions = ["forward", "forward", "right", "forward", "forward"]
    for action in actions:
        obs = await env.step({"name": "minigrid_act", "args": {"action": action}})
        if obs["terminated"]:
            break

    # Final observation should have accumulated rewards
    assert obs["total_reward"] < 0  # Should have step penalties

    # Terminate
    final = await env.terminate()
    assert "success" in final


@pytest.mark.asyncio
async def test_serialization():
    """Test environment serialization."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    await env.initialize()

    # Take some steps
    await env.step({"name": "minigrid_act", "args": {"action": "forward"}})

    # Serialize
    serialized = await env._serialize_engine()

    assert "task_instance_dict" in serialized
    assert "engine_snapshot" in serialized
    assert serialized["engine_snapshot"]["env_name"] == "MiniGrid-Empty-5x5-v0"


if __name__ == "__main__":
    asyncio.run(pytest.main([__file__, "-v"]))
