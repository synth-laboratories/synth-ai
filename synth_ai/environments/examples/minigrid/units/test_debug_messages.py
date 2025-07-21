"""Test debug message functionality in MiniGrid."""

import pytest
import asyncio
from synth_ai.environments.examples.minigrid.environment import MiniGridEnvironment
from synth_ai.environments.examples.minigrid.taskset import DEFAULT_MINIGRID_TASK


@pytest.mark.asyncio
async def test_debug_messages_on_movement():
    """Test that debug messages are properly generated on movement."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    obs = await env.initialize()

    # Test successful forward movement
    tool_call = {"tool": "minigrid_act", "args": {"action": "forward"}}
    obs = await env.step(tool_call)

    assert "debug_message" in obs
    assert "last_action" in obs
    assert "last_action_result" in obs
    assert obs["last_action"] == "forward"
    assert obs["last_action_result"] == "moved"
    assert "Moved forward" in obs["debug_message"]


@pytest.mark.asyncio
async def test_debug_messages_on_blocked_movement():
    """Test debug messages when movement is blocked."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    obs = await env.initialize()

    # Move to a position where we'll be blocked
    # Move right twice to reach the edge
    for _ in range(2):
        tool_call = {"tool": "minigrid_act", "args": {"action": "forward"}}
        obs = await env.step(tool_call)

    # Now try to move forward again - should be blocked
    tool_call = {"tool": "minigrid_act", "args": {"action": "forward"}}
    obs = await env.step(tool_call)

    assert obs["last_action_result"] in ["blocked_by_wall", "blocked_by_boundary"]
    assert "blocked" in obs["debug_message"].lower()


@pytest.mark.asyncio
async def test_debug_messages_on_turn():
    """Test debug messages when turning."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    obs = await env.initialize()

    # Test turning left
    tool_call = {"tool": "minigrid_act", "args": {"action": "left"}}
    obs = await env.step(tool_call)

    assert obs["last_action"] == "left"
    assert obs["last_action_result"] == "turned"
    assert "Turned left" in obs["debug_message"]

    # Test turning right
    tool_call = {"tool": "minigrid_act", "args": {"action": "right"}}
    obs = await env.step(tool_call)

    assert obs["last_action"] == "right"
    assert obs["last_action_result"] == "turned"
    assert "Turned right" in obs["debug_message"]


@pytest.mark.asyncio
async def test_debug_messages_in_observation_text():
    """Test that debug messages appear in the observation text."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    obs = await env.initialize()

    # Move forward
    tool_call = {"tool": "minigrid_act", "args": {"action": "forward"}}
    obs = await env.step(tool_call)

    # Check that debug info appears in the text observation
    observation_text = obs["observation"]
    assert "Debug:" in observation_text
    assert "Last action result:" in observation_text or obs["last_action_result"] == "moved"
