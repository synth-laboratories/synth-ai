#!/usr/bin/env python3
"""Basic test to verify Pokemon Red environment works with real ROM"""

import sys

sys.path.append("/Users/joshuapurtell/Documents/GitHub/Environments/src")

import asyncio

# Test memory extraction functions
from synth_ai.environments.examples.red.engine_helpers.state_extraction import (
    get_badge_count,
    format_position,
    format_hp_status,
)


def test_memory_functions():
    """Test basic memory extraction functions"""
    print("Testing memory extraction functions...")

    # Test badge counting
    assert get_badge_count(0x00) == 0
    assert get_badge_count(0x01) == 1  # Boulder Badge
    assert get_badge_count(0xFF) == 8  # All badges
    print("✓ Badge counting works")

    # Test position formatting
    pos = format_position(10, 8, 3)
    assert pos == "Map03:(10,8)"
    print("✓ Position formatting works")

    # Test HP formatting
    hp = format_hp_status(25, 50)
    assert "25/50" in hp and "50%" in hp
    print("✓ HP formatting works")

    print("All memory functions working!")


async def test_engine_with_rom():
    """Test engine initialization with real ROM"""
    print("\nTesting engine with real ROM...")

    try:
        from synth_ai.environments.examples.red.engine import PokemonRedEngine
        from synth_ai.environments.examples.red.taskset import INSTANCE

        # Try to initialize engine
        engine = PokemonRedEngine(INSTANCE)
        print("✓ Engine initialized successfully with ROM")

        # Test state extraction
        state = engine._extract_current_state()
        print(f"✓ Initial state extracted: {state}")

        # Test reset
        priv, pub = await engine._reset_engine()
        print("✓ Engine reset successful")
        print(f"  Position: {format_position(pub.player_x, pub.player_y, pub.map_id)}")
        print(f"  Badges: {get_badge_count(pub.badges)}")
        print(f"  HP: {format_hp_status(pub.party_hp_current, pub.party_hp_max)}")
        print(f"  Level: {pub.party_level}")

        # Test a button press
        print("\nTesting button press...")
        action = {"button": "A", "frames": 1}
        priv, pub = await engine._step_engine(action)
        print(f"✓ Button press executed, step count: {pub.step_count}")
        print(f"  Reward: {priv.reward_last_step}")
        print(f"  Total reward: {priv.total_reward}")

        return True

    except Exception as e:
        print(f"✗ Engine test failed: {e}")
        return False


async def test_environment():
    """Test full environment"""
    print("\nTesting full environment...")

    try:
        from synth_ai.environments.examples.red.environment import PokemonRedEnvironment
        from synth_ai.environments.environment.tools import EnvToolCall

        env = PokemonRedEnvironment()
        print("✓ Environment created")

        # Initialize
        obs = await env.initialize()
        print("✓ Environment initialized")
        print(f"  Observation keys: {list(obs.keys())}")
        print(f"  Position: {obs.get('position')}")
        print(f"  Badges: {obs.get('badges_earned')}")

        # Test button press
        call = EnvToolCall(tool="press_button", args={"button": "A", "frames": 1})
        obs = await env.step(call)
        print("✓ Step executed via tool")
        print(f"  Step count: {obs.get('step_count')}")
        print(f"  Total reward: {obs.get('total_reward')}")

        return True

    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("=== Pokemon Red Environment Tests ===\n")

    # Test 1: Basic memory functions
    try:
        test_memory_functions()
    except Exception as e:
        print(f"✗ Memory function tests failed: {e}")
        return

    # Test 2: Engine with ROM
    engine_success = await test_engine_with_rom()

    # Test 3: Full environment
    if engine_success:
        env_success = await test_environment()
    else:
        print("Skipping environment test due to engine failure")
        env_success = False

    print("\n=== Results ===")
    print("Memory functions: ✓")
    print(f"Engine with ROM: {'✓' if engine_success else '✗'}")
    print(f"Full environment: {'✓' if env_success else '✗'}")

    if engine_success and env_success:
        print("\n🎉 All tests passed! Pokemon Red environment is working!")
    else:
        print("\n❌ Some tests failed. Check the errors above.")


if __name__ == "__main__":
    asyncio.run(main())
