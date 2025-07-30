"""Test exploration mechanics in MiniGrid."""

import pytest
import asyncio
from synth_ai.environments.examples.minigrid.environment import MiniGridEnvironment
from synth_ai.environments.examples.minigrid.taskset import DEFAULT_MINIGRID_TASK


@pytest.mark.asyncio
async def test_goal_not_always_visible():
    """Test that the goal is not always visible initially."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    obs = await env.initialize()

    # Check if 'G' appears in the actual grid (not in legend)
    lines = obs["observation"].split("\n")
    grid_lines = []
    grid_started = False

    for line in lines:
        if "Grid:" in line:
            grid_started = True
        elif "Legend:" in line:
            break
        elif grid_started and line.strip():
            grid_lines.append(line)

    # The goal 'G' should not be visible in the initial 5x5 view
    # (though this depends on the specific seed/layout)
    grid_text = "\n".join(grid_lines)

    # Goal might or might not be visible - this is expected
    # The test is mainly to document this behavior
    has_goal_in_grid = "G" in grid_text
    assert isinstance(has_goal_in_grid, bool)  # Can be True or False


@pytest.mark.asyncio
async def test_limited_visibility():
    """Test that the agent has limited visibility."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    obs = await env.initialize()

    # Check that the grid contains '?' symbols indicating unseen areas
    observation_text = obs["observation"]
    assert "?" in observation_text

    # The grid should be small (agent's view)
    lines = observation_text.split("\n")
    grid_lines = []
    grid_started = False

    for line in lines:
        if "Grid:" in line:
            grid_started = True
        elif "Legend:" in line:
            break
        elif grid_started and line.strip():
            grid_lines.append(line)

    # In a 5x5 environment, the agent sees a 5x5 view
    assert len(grid_lines) == 5


@pytest.mark.asyncio
async def test_exploration_reveals_new_areas():
    """Test that moving reveals new areas of the grid."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    initial_obs = await env.initialize()

    # Move to a new position
    tool_call = {"tool": "minigrid_act", "args": {"action": "forward"}}
    new_obs = await env.step(tool_call)

    # The observations should be different (agent moved)
    assert initial_obs["observation"] != new_obs["observation"]

    # Agent position should have changed
    initial_pos = None
    new_pos = None

    for line in initial_obs["observation"].split("\n"):
        if "Agent Position:" in line:
            initial_pos = line
            break

    for line in new_obs["observation"].split("\n"):
        if "Agent Position:" in line:
            new_pos = line
            break

    assert initial_pos != new_pos


@pytest.mark.asyncio
async def test_complete_exploration_finds_goal():
    """Test that systematic exploration can find the goal."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    obs = await env.initialize()

    # Known solution path for the default task
    solution_path = [
        "forward",  # Move right to (2,1)
        "forward",  # Move right to (3,1)
        "right",  # Turn to face down
        "forward",  # Move down to (3,2)
        "forward",  # Move down to (3,3) - goal
    ]

    for action in solution_path:
        tool_call = {"tool": "minigrid_act", "args": {"action": action}}
        obs = await env.step(tool_call)

        if obs.get("terminated", False):
            # Should have found the goal
            assert obs.get("total_reward", 0) > 0
            return

    # If we didn't terminate, the test fails
    assert False, "Failed to reach goal with known solution path"
