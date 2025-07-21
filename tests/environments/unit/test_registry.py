"""
Unit tests for the environment registry functionality.
"""

import pytest
from synth_ai.environments.environment.registry import (
    register_environment,
    get_environment_cls,
    list_supported_env_types,
    ENV_REGISTRY,
)
from synth_ai.environments.stateful.core import StatefulEnvironment
from synth_ai.environments.examples.tictactoe.environment import TicTacToeEnvironment


class MockEnvironment(StatefulEnvironment):
    """Mock environment for testing."""

    def __init__(self, task_instance):
        self.task_instance = task_instance

    async def initialize(self):
        return {"observation": "initialized"}

    async def step(self, tool_calls):
        return {"observation": "stepped", "done": False}

    async def terminate(self):
        return {"observation": "terminated"}

    async def checkpoint(self):
        return {"state": "checkpointed"}


@pytest.mark.fast
class TestRegistry:
    """Test the environment registry functions."""

    def setup_method(self):
        """Clear the registry before each test."""
        self.original_registry = ENV_REGISTRY.copy()
        ENV_REGISTRY.clear()

    def teardown_method(self):
        """Restore the original registry after each test."""
        ENV_REGISTRY.clear()
        ENV_REGISTRY.update(self.original_registry)

    def test_register_environment(self):
        """Test registering a new environment."""
        register_environment("TestEnv", MockEnvironment)
        assert "TestEnv" in ENV_REGISTRY
        assert ENV_REGISTRY["TestEnv"] == MockEnvironment

    def test_get_environment_cls(self):
        """Test retrieving a registered environment class."""
        register_environment("TestEnv", MockEnvironment)
        cls = get_environment_cls("TestEnv")
        assert cls == MockEnvironment

    def test_get_environment_cls_not_found(self):
        """Test retrieving a non-existent environment raises error."""
        with pytest.raises(ValueError, match="Unsupported environment type: NonExistent"):
            get_environment_cls("NonExistent")

    def test_list_supported_env_types(self):
        """Test listing all registered environment types."""
        register_environment("TestEnv1", MockEnvironment)
        register_environment("TestEnv2", MockEnvironment)

        env_types = list_supported_env_types()
        assert len(env_types) == 2
        assert "TestEnv1" in env_types
        assert "TestEnv2" in env_types

    def test_register_multiple_environments(self):
        """Test registering multiple environments."""

        class AnotherMockEnv(MockEnvironment):
            pass

        register_environment("TestEnv1", MockEnvironment)
        register_environment("TestEnv2", AnotherMockEnv)

        assert get_environment_cls("TestEnv1") == MockEnvironment
        assert get_environment_cls("TestEnv2") == AnotherMockEnv

    def test_overwrite_environment(self):
        """Test overwriting an existing environment registration."""

        class UpdatedMockEnv(MockEnvironment):
            pass

        register_environment("TestEnv", MockEnvironment)
        register_environment("TestEnv", UpdatedMockEnv)

        assert get_environment_cls("TestEnv") == UpdatedMockEnv

    def test_register_environment_tictactoe(self):
        """Test registering TicTacToe environment."""
        register_environment("TestEnv", TicTacToeEnvironment)
        assert "TestEnv" in list_supported_env_types()

    def test_get_environment_cls_tictactoe(self):
        """Test retrieving TicTacToe environment class."""
        register_environment("TestEnv", TicTacToeEnvironment)
        env_cls = get_environment_cls("TestEnv")
        assert env_cls == TicTacToeEnvironment

    def test_get_environment_cls_not_found_tictactoe(self):
        """Test error when environment not found."""
        with pytest.raises(ValueError, match="Unsupported environment type: NonExistent"):
            get_environment_cls("NonExistent")

    def test_list_supported_env_types_tictactoe(self):
        """Test listing multiple TicTacToe environments."""
        register_environment("TestEnv1", TicTacToeEnvironment)
        register_environment("TestEnv2", TicTacToeEnvironment)

        env_types = list_supported_env_types()
        assert "TestEnv1" in env_types
        assert "TestEnv2" in env_types

    def test_register_multiple_environments_tictactoe(self):
        """Test registering multiple TicTacToe environments."""
        register_environment("TestEnv1", TicTacToeEnvironment)
        register_environment("TestEnv2", TicTacToeEnvironment)

        assert len(list_supported_env_types()) >= 2

    def test_overwrite_environment_tictactoe(self):
        """Test overwriting a TicTacToe environment."""
        register_environment("TestEnv", TicTacToeEnvironment)

        # Should be able to overwrite
        register_environment("TestEnv", TicTacToeEnvironment)
        assert "TestEnv" in list_supported_env_types()
