"""Test actual action behavior to debug left/right turn issues."""

import pytest
from synth_ai.environments.examples.minigrid.environment import MiniGridEnvironment
from synth_ai.environments.examples.minigrid.taskset import DEFAULT_MINIGRID_TASK


@pytest.mark.asyncio
async def test_initial_state():
    """Test the initial state of the agent."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    obs = await env.initialize()

    # Check initial state
    assert "Agent Position: (1, 1)" in obs["observation"]
    assert "Agent Direction: →" in obs["observation"]
    print(f"✓ Initial state verified: position (1,1), direction →")


@pytest.mark.asyncio
async def test_right_turn_action():
    """Test what happens when we send 'right' action."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    await env.initialize()

    # Send 'right' action
    tool_call = {"tool": "minigrid_act", "args": {"action": "right"}}
    obs = await env.step(tool_call)

    # Extract direction from observation
    obs_text = obs["observation"]
    if "Agent Direction: ↓" in obs_text:
        actual_dir = "↓ (down)"
        expected = True
    elif "Agent Direction: ↑" in obs_text:
        actual_dir = "↑ (up)"
        expected = False
    elif "Agent Direction: ←" in obs_text:
        actual_dir = "← (left)"
        expected = False
    elif "Agent Direction: →" in obs_text:
        actual_dir = "→ (right)"
        expected = False
    else:
        actual_dir = "unknown"
        expected = False

    print(f"RIGHT action result: {actual_dir}")
    print(f"Expected: ↓ (down) for clockwise turn")
    print(f"✓ RIGHT turn working correctly: {expected}")

    assert expected, f"RIGHT turn failed: expected ↓ (down), got {actual_dir}"


@pytest.mark.asyncio
async def test_left_turn_action():
    """Test what happens when we send 'left' action."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    await env.initialize()

    # Send 'left' action
    tool_call = {"tool": "minigrid_act", "args": {"action": "left"}}
    obs = await env.step(tool_call)

    # Extract direction from observation
    obs_text = obs["observation"]
    if "Agent Direction: ↑" in obs_text:
        actual_dir = "↑ (up)"
        expected = True
    elif "Agent Direction: ↓" in obs_text:
        actual_dir = "↓ (down)"
        expected = False
    elif "Agent Direction: ←" in obs_text:
        actual_dir = "← (left)"
        expected = False
    elif "Agent Direction: →" in obs_text:
        actual_dir = "→ (right)"
        expected = False
    else:
        actual_dir = "unknown"
        expected = False

    print(f"LEFT action result: {actual_dir}")
    print(f"Expected: ↑ (up) for counter-clockwise turn")
    print(f"✓ LEFT turn working correctly: {expected}")

    assert expected, f"LEFT turn failed: expected ↑ (up), got {actual_dir}"


@pytest.mark.asyncio
async def test_full_rotation_sequence():
    """Test a full sequence of turns to verify direction logic."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    await env.initialize()

    directions = []

    # Get initial direction
    obs = await env.checkpoint()
    if "Agent Direction: →" in obs["observation"]:
        directions.append("→")

    # Do 4 right turns (should return to initial direction)
    for i in range(4):
        tool_call = {"tool": "minigrid_act", "args": {"action": "right"}}
        obs = await env.step(tool_call)

        # Extract direction
        obs_text = obs["observation"]
        if "Agent Direction: ↓" in obs_text:
            directions.append("↓")
        elif "Agent Direction: ←" in obs_text:
            directions.append("←")
        elif "Agent Direction: ↑" in obs_text:
            directions.append("↑")
        elif "Agent Direction: →" in obs_text:
            directions.append("→")

    print(f"Full rotation sequence: {' -> '.join(directions)}")
    print(f"Expected clockwise: → -> ↓ -> ← -> ↑ -> →")

    expected_sequence = ["→", "↓", "←", "↑", "→"]
    assert directions == expected_sequence, f"Rotation sequence wrong: {directions}"


@pytest.mark.asyncio
async def test_forward_movement():
    """Test forward movement in different directions."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    await env.initialize()

    # Test forward when facing right (initial direction)
    tool_call = {"tool": "minigrid_act", "args": {"action": "forward"}}
    obs = await env.step(tool_call)

    # Should move from (1,1) to (2,1)
    assert "Agent Position: (2, 1)" in obs["observation"]
    print("✓ Forward movement verified: (1,1) -> (2,1)")

    # Move to (3,1)
    obs = await env.step(tool_call)
    assert "Agent Position: (3, 1)" in obs["observation"]
    print("✓ Forward movement verified: (2,1) -> (3,1)")

    # Try to move forward again (should hit wall)
    obs = await env.step(tool_call)
    assert "Agent Position: (3, 1)" in obs["observation"]  # Should stay at (3,1)
    assert obs.get("last_action_result") == "blocked_by_wall"
    print("✓ Wall blocking verified: stayed at (3,1)")


@pytest.mark.asyncio
async def test_turn_then_move_sequence():
    """Test the critical sequence: move to (3,1), turn right, then move toward goal."""
    env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)
    await env.initialize()

    # Move to (3,1)
    forward_call = {"tool": "minigrid_act", "args": {"action": "forward"}}
    await env.step(forward_call)  # (1,1) -> (2,1)
    await env.step(forward_call)  # (2,1) -> (3,1)

    # Verify at (3,1) facing right
    obs = await env.checkpoint()
    assert "Agent Position: (3, 1)" in obs["observation"]
    assert "Agent Direction: →" in obs["observation"]
    print("✓ At position (3,1) facing right")

    # Turn right (should face down toward goal)
    right_call = {"tool": "minigrid_act", "args": {"action": "right"}}
    obs = await env.step(right_call)

    direction_after_right = None
    if "Agent Direction: ↓" in obs["observation"]:
        direction_after_right = "↓ (down)"
        facing_goal = True
    elif "Agent Direction: ↑" in obs["observation"]:
        direction_after_right = "↑ (up)"
        facing_goal = False
    else:
        direction_after_right = "other"
        facing_goal = False

    print(f"After RIGHT turn at (3,1): facing {direction_after_right}")
    print(f"Goal is at (3,3), so agent should face ↓ (down)")
    print(f"✓ Facing toward goal: {facing_goal}")

    # If facing down, try to move toward goal
    if facing_goal:
        obs = await env.step(forward_call)
        if "Agent Position: (3, 2)" in obs["observation"]:
            print("✓ Successfully moved toward goal: (3,1) -> (3,2)")

            # Try to reach goal
            obs = await env.step(forward_call)
            if "Agent Position: (3, 3)" in obs["observation"]:
                print("✓ SUCCESS: Reached goal at (3,3)!")
                return True

    return False


if __name__ == "__main__":
    import asyncio

    async def run_tests():
        print("=== TESTING MINIGRID ACTION BEHAVIOR ===")

        try:
            await test_initial_state()
            await test_right_turn_action()
            await test_left_turn_action()
            await test_full_rotation_sequence()
            await test_forward_movement()
            success = await test_turn_then_move_sequence()

            print(f"\n=== SUMMARY ===")
            print(f"Goal reached successfully: {success}")

        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(run_tests())
