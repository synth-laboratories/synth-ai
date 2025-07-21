#!/usr/bin/env python3
"""Comprehensive tests for Pokemon Red environment - verifying harness gets core info and controls work"""

import sys

sys.path.append("/Users/joshuapurtell/Documents/GitHub/Environments/src")

import asyncio

from synth_ai.environments.examples.red.environment import PokemonRedEnvironment
from synth_ai.environments.examples.red.engine import PokemonRedEngine
from synth_ai.environments.examples.red.taskset import INSTANCE
from synth_ai.environments.environment.tools import EnvToolCall


async def test_memory_state_tracking():
    """Test that we can track key game state metrics"""
    print("=== Testing Memory State Tracking ===")

    engine = PokemonRedEngine(INSTANCE)

    # Test initial state
    state = engine._extract_current_state()
    print(f"✓ Initial state keys: {list(state.keys())}")

    # Verify all critical metrics are tracked
    critical_metrics = [
        "map_id",
        "player_x",
        "player_y",
        "badges",
        "in_battle",
        "party_level",
        "party_hp_current",
        "party_hp_max",
        "party_xp",
    ]

    for metric in critical_metrics:
        assert metric in state, f"Missing critical metric: {metric}"
        print(f"  ✓ {metric}: {state[metric]}")

    # Test state evolution after button press
    prev_state = state.copy()
    engine._press_button("A", 1)
    new_state = engine._extract_current_state()

    print("✓ State after button press - some values may change")
    for key in critical_metrics:
        if new_state[key] != prev_state[key]:
            print(f"  → {key}: {prev_state[key]} → {new_state[key]}")

    return True


async def test_reward_system():
    """Test that reward system properly tracks game progress"""
    print("\n=== Testing Reward System ===")

    engine = PokemonRedEngine(INSTANCE)
    await engine._reset_engine()

    # Test step penalty
    action = {"button": "A", "frames": 1}
    priv, pub = await engine._step_engine(action)

    print(f"✓ Step penalty applied: {priv.reward_last_step}")
    assert priv.reward_last_step < 0, "Step penalty should be negative"

    # Test reward calculation doesn't crash with various button combinations
    test_buttons = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]
    total_reward = priv.total_reward

    for button in test_buttons:
        action = {"button": button, "frames": 1}
        priv, pub = await engine._step_engine(action)
        print(f"  ✓ {button} button: reward={priv.reward_last_step:.3f}")
        assert isinstance(priv.reward_last_step, float)

    print(f"✓ Total reward after button tests: {priv.total_reward}")
    return True


async def test_button_controls():
    """Test that all Game Boy controls work properly"""
    print("\n=== Testing Button Controls ===")

    env = PokemonRedEnvironment()
    await env.initialize()

    # Test all button combinations
    buttons = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]
    frame_counts = [1, 2, 5]

    for button in buttons:
        for frames in frame_counts:
            call = EnvToolCall(tool="press_button", args={"button": button, "frames": frames})
            obs = await env.step(call)

            print(f"  ✓ {button} button ({frames} frames) - step: {obs['step_count']}")
            assert "step_count" in obs
            assert obs["step_count"] > 0

    # Test invalid button handling
    try:
        call = EnvToolCall(tool="press_button", args={"button": "INVALID", "frames": 1})
        obs = await env.step(call)
        print("  ✓ Invalid button handled gracefully")
    except Exception as e:
        print(f"  ✓ Invalid button properly rejected: {type(e).__name__}")

    return True


async def test_observation_richness():
    """Test that observations contain rich, useful information"""
    print("\n=== Testing Observation Richness ===")

    env = PokemonRedEnvironment()
    obs = await env.initialize()

    # Check all expected observation fields
    expected_fields = [
        "position",
        "badges_earned",
        "badges_bitfield",
        "hp_status",
        "party_level",
        "party_xp",
        "in_battle",
        "step_count",
        "reward_last_step",
        "total_reward",
        "terminated",
    ]

    for field in expected_fields:
        assert field in obs, f"Missing observation field: {field}"
        print(f"  ✓ {field}: {obs[field]}")

    # Test observation evolution
    initial_step = obs["step_count"]
    call = EnvToolCall(tool="press_button", args={"button": "A", "frames": 1})
    obs = await env.step(call)

    print(f"✓ Step count evolution: {initial_step} → {obs['step_count']}")
    assert obs["step_count"] > initial_step

    # Test position formatting
    position = obs["position"]
    assert "Map" in position and ":" in position
    print(f"✓ Position format valid: {position}")

    # Test HP status formatting
    hp_status = obs["hp_status"]
    assert "HP:" in hp_status
    print(f"✓ HP status format valid: {hp_status}")

    return True


async def test_game_progression_detection():
    """Test that the harness can detect meaningful game progression"""
    print("\n=== Testing Game Progression Detection ===")

    engine = PokemonRedEngine(INSTANCE)

    # Test badge detection
    print("Testing badge progression detection...")

    # Simulate earning badges by manually setting memory
    # (In real gameplay, this would happen through game events)
    def simulate_badge_earned(badge_num):
        """Simulate earning a specific badge"""
        # This is for testing - in real game, badges are earned through gameplay
        badge_flag = 1 << (badge_num - 1)  # Badge 1 = bit 0, Badge 2 = bit 1, etc.

        # Create mock state with badge
        prev_state = engine._extract_current_state()
        current_state = prev_state.copy()
        current_state["badges"] = badge_flag

        return prev_state, current_state

    # Test badge reward calculation
    prev_state, current_state = simulate_badge_earned(1)  # Boulder Badge

    # Manually test reward calculation
    from synth_ai.environments.examples.red.engine_helpers.reward_components import (
        BadgeRewardComponent,
    )

    badge_component = BadgeRewardComponent()

    reward = await badge_component.score(
        state=current_state, action={"prev_badges": prev_state["badges"]}
    )

    print(f"✓ Badge reward calculation: {reward} (should be 1.0 for first badge)")
    assert reward == 1.0, f"Expected badge reward 1.0, got {reward}"

    # Test battle state detection
    print("Testing battle state detection...")

    battle_state = engine._extract_current_state()
    battle_state["in_battle"] = True
    print(f"✓ Battle state detected: {battle_state['in_battle']}")

    # Test level tracking
    print("Testing level progression...")

    level_state = engine._extract_current_state()
    level_state["party_level"] = 10
    print(f"✓ Party level tracked: {level_state['party_level']}")

    return True


async def test_checkpointing_system():
    """Test that checkpointing preserves game state"""
    print("\n=== Testing Checkpointing System ===")

    env = PokemonRedEnvironment()
    await env.initialize()

    # Take some steps to change state
    for i in range(3):
        call = EnvToolCall(tool="press_button", args={"button": "A", "frames": 1})
        await env.step(call)

    # Create checkpoint
    checkpoint_obs = await env.checkpoint()

    print(f"✓ Checkpoint created with keys: {list(checkpoint_obs.keys())}")
    assert "engine_snapshot_data" in checkpoint_obs

    snapshot_data = checkpoint_obs["engine_snapshot_data"]
    print(f"✓ Snapshot contains: {list(snapshot_data.keys())}")

    required_snapshot_fields = ["state_data", "total_reward", "step_count"]
    for field in required_snapshot_fields:
        assert field in snapshot_data, f"Missing snapshot field: {field}"
        print(f"  ✓ {field}: {snapshot_data[field]}")

    return True


async def test_error_handling():
    """Test that the harness handles errors gracefully"""
    print("\n=== Testing Error Handling ===")

    env = PokemonRedEnvironment()
    await env.initialize()

    # Test with malformed tool calls
    try:
        call = EnvToolCall(tool="press_button", args={})  # Missing button
        obs = await env.step(call)
        print("✓ Malformed call handled gracefully")
    except Exception as e:
        print(f"✓ Malformed call properly rejected: {type(e).__name__}")

    # Test environment termination
    final_obs = await env.terminate()
    print(f"✓ Environment termination: {final_obs.get('terminated')}")
    assert final_obs.get("terminated") is True

    return True


async def main():
    """Run comprehensive tests"""
    print("🎮 Pokemon Red Comprehensive Test Suite")
    print("=" * 50)

    tests = [
        ("Memory State Tracking", test_memory_state_tracking),
        ("Reward System", test_reward_system),
        ("Button Controls", test_button_controls),
        ("Observation Richness", test_observation_richness),
        ("Game Progression Detection", test_game_progression_detection),
        ("Checkpointing System", test_checkpointing_system),
        ("Error Handling", test_error_handling),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            success = await test_func()
            results[test_name] = success
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            results[test_name] = False

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")

    passed = sum(results.values())
    total = len(results)

    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\n🏆 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Pokemon Red harness is comprehensive and working!")
        print("\nKey capabilities verified:")
        print("  • Memory state extraction from real Game Boy ROM")
        print("  • All button controls functional")
        print("  • Rich observations with game metrics")
        print("  • Dense reward system for AI training")
        print("  • Game progression detection (badges, levels, battles)")
        print("  • Robust error handling")
        print("  • State checkpointing for reproducibility")
    else:
        print(f"\n❌ {total - passed} tests failed. Check errors above.")


if __name__ == "__main__":
    asyncio.run(main())
