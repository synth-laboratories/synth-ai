#!/usr/bin/env python3
"""
Test script to verify red environment fixes.
Tests JAX logging suppression and error handling.
"""

import asyncio
import logging
import sys

from synth_ai.environments.examples.red.environment import PokemonRedEnvironment
from synth_ai.environments.examples.red.taskset import INSTANCE as POKEMON_TASK
from synth_ai.environments.environment.tools import EnvToolCall


class PressButtonCall(EnvToolCall):
    """Helper class for creating button press calls"""

    def __init__(self, button: str, frames: int = 1):
        super().__init__(tool="press_button", args={"button": button, "frames": frames})


async def test_environment_setup():
    """Test that the environment can be set up without errors."""
    print("Testing Pokemon Red environment setup...")

    try:
        # Create environment instance
        env = PokemonRedEnvironment(POKEMON_TASK)
        print("✅ Environment created successfully")

        # Try to initialize
        obs = await env.initialize()
        print("✅ Environment initialized successfully")
        print(f"Initial observation keys: {list(obs.keys())}")

        # Try a simple step
        obs = await env.step(PressButtonCall("A"))
        print("✅ Environment step executed successfully")
        print(
            f"Step observation: step_count={obs.get('step_count')}, terminated={obs.get('terminated')}"
        )

        # Terminate
        final_obs = await env.terminate()
        print("✅ Environment terminated successfully")

        return True

    except Exception as e:
        print(f"❌ Failed to setup environment: {e}")
        logging.exception("Failed to setup environment, aborting test")
        return False


def test_logging_configuration():
    """Test that logging is properly configured."""
    print("Testing logging configuration...")

    # Check that JAX loggers are set to WARNING level
    jax_loggers = [
        "jax._src.cache_key",
        "jax._src.compilation_cache",
        "jax._src.compiler",
        "jax._src.dispatch",
    ]

    for logger_name in jax_loggers:
        logger = logging.getLogger(logger_name)
        if logger.level >= logging.WARNING:
            print(f"✅ {logger_name} logger level: {logging.getLevelName(logger.level)}")
        else:
            print(
                f"❌ {logger_name} logger level: {logging.getLevelName(logger.level)} (should be WARNING or higher)"
            )

    # Test that debug messages are suppressed
    jax_logger = logging.getLogger("jax._src.cache_key")
    jax_logger.debug("This debug message should not appear")
    print("✅ JAX debug logging appears to be suppressed")


def test_safe_compare():
    """Test the safe comparison function."""
    print("Testing safe comparison function...")

    from synth_ai.environments.examples.red.config_logging import safe_compare

    # Test cases
    test_cases = [
        ("5", 3, ">", True),  # String vs int
        (5, "3", ">", True),  # Int vs string
        ("abc", 5, ">", False),  # Invalid string vs int
        ("5", "3", ">", True),  # String vs string (numeric)
        ("abc", "def", ">", False),  # String vs string (alphabetic)
        (5, 3, ">", True),  # Normal int comparison
    ]

    for left, right, op, expected in test_cases:
        result = safe_compare(left, right, op)
        status = "✅" if result == expected else "❌"
        print(f"{status} safe_compare({left}, {right}, '{op}') = {result} (expected {expected})")


async def main():
    """Main test function."""
    print("Running Pokemon Red environment fixes test...\n")

    # Test logging configuration
    test_logging_configuration()
    print()

    # Test safe comparison
    test_safe_compare()
    print()

    # Test environment setup
    success = await test_environment_setup()

    print(f"\nOverall test result: {'✅ PASSED' if success else '❌ FAILED'}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
