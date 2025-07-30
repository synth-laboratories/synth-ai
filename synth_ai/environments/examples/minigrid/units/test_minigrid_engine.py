"""Unit tests for MiniGrid engine."""

import asyncio
import pytest
import numpy as np

from synth_ai.environments.examples.minigrid.engine import (
    MiniGridEngine,
    MiniGridPublicState,
    MiniGridPrivateState,
    MiniGridStepPenaltyComponent,
)
from synth_ai.environments.examples.minigrid.taskset import DEFAULT_MINIGRID_TASK


@pytest.mark.asyncio
async def test_engine_initialization():
    """Test engine initialization."""
    engine = MiniGridEngine(DEFAULT_MINIGRID_TASK)

    # Check initial state
    assert engine.env_name == "MiniGrid-Empty-5x5-v0"
    assert engine.seed == 42
    assert engine.total_reward == 0.0
    assert not engine._initialized

    # Reset engine
    priv, pub = await engine._reset_engine()

    # Check reset state
    assert engine._initialized
    assert isinstance(priv, MiniGridPrivateState)
    assert isinstance(pub, MiniGridPublicState)
    assert priv.terminated is False
    assert priv.truncated is False
    assert priv.total_reward == 0.0
    assert pub.grid_array.shape == (5, 5, 3)
    assert pub.agent_pos == (1, 1)  # Default starting position
    assert pub.step_count == 0
    assert pub.mission == "get to the green goal square"


@pytest.mark.asyncio
async def test_engine_step():
    """Test engine step functionality."""
    engine = MiniGridEngine(DEFAULT_MINIGRID_TASK)
    await engine._reset_engine()

    # Test moving forward
    initial_pos = engine.env.unwrapped.agent_pos
    priv, pub = await engine._step_engine(2)  # Forward action

    # Check step results
    assert isinstance(priv, MiniGridPrivateState)
    assert isinstance(pub, MiniGridPublicState)
    assert pub.step_count == 1
    assert priv.reward_last == -0.01  # Step penalty
    assert priv.total_reward == -0.01

    # Test turning
    initial_dir = pub.agent_dir
    priv, pub = await engine._step_engine(0)  # Turn left
    assert pub.agent_dir == (initial_dir - 1) % 4
    assert pub.step_count == 2


@pytest.mark.asyncio
async def test_invalid_actions():
    """Test invalid action handling."""
    engine = MiniGridEngine(DEFAULT_MINIGRID_TASK)
    await engine._reset_engine()

    # Test invalid action values
    with pytest.raises(ValueError, match="Invalid action"):
        await engine._step_engine(-1)

    with pytest.raises(ValueError, match="Invalid action"):
        await engine._step_engine(7)

    with pytest.raises(ValueError, match="Invalid action"):
        await engine._step_engine("forward")


@pytest.mark.asyncio
async def test_grid_to_array():
    """Test grid to array conversion."""
    engine = MiniGridEngine(DEFAULT_MINIGRID_TASK)
    await engine._reset_engine()

    grid_array = engine._grid_to_array()

    # Check array properties
    assert isinstance(grid_array, np.ndarray)
    assert grid_array.shape == (5, 5, 3)
    assert grid_array.dtype == np.uint8

    # Check agent is in the grid
    agent_pos = engine.env.unwrapped.agent_pos
    agent_cell = grid_array[agent_pos[1], agent_pos[0]]
    assert agent_cell[0] == 9  # Agent object type


@pytest.mark.asyncio
async def test_state_diff():
    """Test state diff functionality."""
    engine = MiniGridEngine(DEFAULT_MINIGRID_TASK)
    priv1, pub1 = await engine._reset_engine()

    # Take a step
    priv2, pub2 = await engine._step_engine(2)  # Forward

    # Check public state diff
    diff = pub2.diff(pub1)
    assert "step_count" in diff
    assert diff["step_count"] == 1
    if pub1.agent_pos != pub2.agent_pos:
        assert "agent_pos" in diff

    # Check private state diff
    priv_diff = priv2.diff(priv1)
    assert "reward_last" in priv_diff
    assert "total_reward" in priv_diff


@pytest.mark.asyncio
async def test_reward_components():
    """Test reward components."""
    component = MiniGridStepPenaltyComponent()

    # Create a dummy state
    from synth_ai.environments.examples.minigrid.engine import MiniGridPublicState

    state = MiniGridPublicState(
        grid_array=np.zeros((5, 5, 3)),
        agent_pos=(1, 1),
        agent_dir=0,
        step_count=1,
        max_steps=100,
        mission="test",
    )

    # Test penalty
    reward = await component.score(state, 2)
    assert reward == -0.01


@pytest.mark.asyncio
async def test_serialization():
    """Test engine serialization."""
    engine = MiniGridEngine(DEFAULT_MINIGRID_TASK)
    await engine._reset_engine()

    # Take some steps
    await engine._step_engine(2)
    await engine._step_engine(1)

    # Serialize
    snapshot = await engine._serialize_engine()

    # Check snapshot
    assert snapshot.engine_snapshot["env_name"] == "MiniGrid-Empty-5x5-v0"
    assert snapshot.engine_snapshot["seed"] == 42
    assert snapshot.engine_snapshot["initialized"] is True
    assert "total_reward" in snapshot.engine_snapshot


@pytest.mark.asyncio
async def test_get_available_actions():
    """Test getting available actions."""
    engine = MiniGridEngine(DEFAULT_MINIGRID_TASK)
    actions = engine.get_available_actions()

    assert len(actions) == 7
    assert actions[0] == (0, "turn left")
    assert actions[2] == (2, "move forward")
    assert actions[3] == (3, "pickup")


@pytest.mark.asyncio
async def test_different_environments():
    """Test different MiniGrid environments."""
    from synth_ai.environments.examples.minigrid.taskset import (
        MiniGridTaskInstance,
        MiniGridTaskInstanceMetadata,
    )
    from synth_ai.environments.tasks.api import Impetus, Intent
    from uuid import uuid4

    # Test DoorKey environment
    task = MiniGridTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="Test"),
        intent=Intent(rubric={"goal": "Test"}, gold_trajectories=None, gold_state_diff={}),
        metadata=MiniGridTaskInstanceMetadata(
            env_name="MiniGrid-DoorKey-5x5-v0",
            grid_size=(5, 5),
            difficulty="medium",
            has_key=True,
            has_door=True,
        ),
        is_reproducible=True,
        initial_engine_snapshot=None,
    )

    engine = MiniGridEngine(task)
    priv, pub = await engine._reset_engine()

    # Check environment properties
    assert pub.mission == "open the door then get to the goal"
    assert pub.grid_array.shape == (5, 5, 3)


if __name__ == "__main__":
    asyncio.run(pytest.main([__file__, "-v"]))
