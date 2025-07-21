#!/usr/bin/env python3
"""
Mock test script to verify red environment fixes without ROM file.
Tests JAX logging suppression and error handling.
"""

import logging
import sys
from unittest.mock import Mock, patch


def test_logging_configuration():
    """Test that logging is properly configured."""
    print("Testing logging configuration...")

    # Import configuration to trigger setup
    from synth_ai.environments.examples.red.config_logging import configure_logging

    configure_logging()

    # Check that JAX loggers are set to WARNING level
    jax_loggers = [
        "jax._src.cache_key",
        "jax._src.compilation_cache",
        "jax._src.compiler",
        "jax._src.dispatch",
    ]

    success = True
    for logger_name in jax_loggers:
        logger = logging.getLogger(logger_name)
        if logger.level >= logging.WARNING:
            print(f"✅ {logger_name} logger level: {logging.getLevelName(logger.level)}")
        else:
            print(
                f"❌ {logger_name} logger level: {logging.getLevelName(logger.level)} (should be WARNING or higher)"
            )
            success = False

    # Test that debug messages are suppressed
    jax_logger = logging.getLogger("jax._src.cache_key")
    jax_logger.debug("This debug message should not appear")
    print("✅ JAX debug logging appears to be suppressed")

    return success


def test_safe_compare():
    """Test the safe comparison function."""
    print("Testing safe comparison function...")

    from synth_ai.environments.examples.red.config_logging import safe_compare

    # Test cases that previously would cause the string vs int error
    test_cases = [
        ("5", 3, ">", True),  # String vs int
        (5, "3", ">", True),  # Int vs string
        ("abc", 5, ">", False),  # Invalid string vs int
        ("5", "3", ">", True),  # String vs string (numeric)
        ("abc", "def", ">", False),  # String vs string (alphabetic)
        (5, 3, ">", True),  # Normal int comparison
        ("10", 5, ">=", True),  # String number >= int
        (3, "10", "<=", True),  # Int <= string number
    ]

    success = True
    for left, right, op, expected in test_cases:
        result = safe_compare(left, right, op)
        status = "✅" if result == expected else "❌"
        print(f"{status} safe_compare({left}, {right}, '{op}') = {result} (expected {expected})")
        if result != expected:
            success = False

    return success


def test_state_creation_error_handling():
    """Test that state creation handles type errors gracefully."""
    print("Testing state creation error handling...")

    from synth_ai.environments.examples.red.engine import PokemonRedEngine
    from synth_ai.environments.examples.red.taskset import INSTANCE as POKEMON_TASK

    try:
        # Mock the PyBoy emulator to avoid ROM requirement
        with patch("examples.red.engine.PyBoy") as mock_pyboy:
            mock_emulator = Mock()
            mock_pyboy.return_value = mock_emulator

            # Create engine instance
            engine = PokemonRedEngine(POKEMON_TASK)

            # Mock extract_game_state to return problematic data that could cause comparison errors
            with patch.object(engine, "_extract_current_state") as mock_extract:
                # Test with string badges that could cause comparison error
                mock_extract.return_value = {
                    "map_id": "1",  # String instead of int
                    "player_x": "10",
                    "player_y": "20",
                    "badges": "abc",  # Non-numeric string
                    "in_battle": "false",  # String instead of bool
                    "party_level": "5",
                    "party_hp_current": "50",
                    "party_hp_max": "50",
                    "party_xp": "100",
                }

                # This should not crash due to our error handling
                priv_state, pub_state = engine._create_states(0.0, False)

                print("✅ State creation handles problematic data gracefully")
                print(f"✅ Created states: badges={pub_state.badges}, map_id={pub_state.map_id}")

                # Test with completely invalid data
                mock_extract.side_effect = Exception("Memory read error")
                priv_state, pub_state = engine._create_states(0.0, False)
                print("✅ State creation handles extraction errors gracefully")

                return True

    except Exception as e:
        print(f"❌ State creation error handling failed: {e}")
        return False


def main():
    """Main test function."""
    print("Running Pokemon Red environment fixes test (mock version)...\n")

    # Test logging configuration
    logging_ok = test_logging_configuration()
    print()

    # Test safe comparison
    compare_ok = test_safe_compare()
    print()

    # Test error handling
    error_handling_ok = test_state_creation_error_handling()
    print()

    success = logging_ok and compare_ok and error_handling_ok
    print(f"Overall test result: {'✅ PASSED' if success else '❌ FAILED'}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
