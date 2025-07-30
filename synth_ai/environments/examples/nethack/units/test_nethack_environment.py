"""Unit tests for NetHack environment."""

import pytest
import asyncio
from uuid import uuid4

from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent

from synth_ai.environments.examples.nethack.environment import (
    NetHackEnvironment,
    NetHackInteractTool,
)
from synth_ai.environments.examples.nethack.taskset import (
    NetHackTaskInstanceMetadata,
    NetHackTaskInstance,
)


class TestNetHackEnvironment:
    """Test cases for NetHack environment."""

    @pytest.fixture
    def mock_task_instance(self):
        """Create a mock task instance for testing."""
        metadata = NetHackTaskInstanceMetadata(
            character_role="knight",
            starting_level=1,
            target_depth=3,
            time_limit=500,
            difficulty="beginner",
            special_objectives=["Defeat 10 monsters"],
            seed=123,
        )

        return NetHackTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions="Test knight adventure"),
            intent=Intent(
                rubric={"goal": "Reach depth 3"},
                gold_trajectories=None,
                gold_state_diff={},
            ),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,
        )

    @pytest.mark.asyncio
    async def test_environment_initialization(self, mock_task_instance):
        """Test environment initialization."""
        env = NetHackEnvironment(mock_task_instance)

        assert env.name == "NetHack"
        assert env.task_instance == mock_task_instance
        assert env.engine is not None

        # Initialize and check observation
        obs = await env.initialize()

        assert isinstance(obs, dict)
        assert "ascii_map" in obs
        assert "message" in obs
        assert "character_stats" in obs
        assert "terminated" in obs
        assert obs["terminated"] is False

    @pytest.mark.asyncio
    async def test_step_with_valid_action(self, mock_task_instance):
        """Test stepping with valid actions."""
        env = NetHackEnvironment(mock_task_instance)
        await env.initialize()

        # Test simple string action
        obs = await env.step("north")
        assert "last_action" in obs
        assert obs["last_action"] == "north"
        assert obs["turn_count"] == 1

        # Test another movement
        obs = await env.step("east")
        assert obs["last_action"] == "east"
        assert obs["turn_count"] == 2

    @pytest.mark.asyncio
    async def test_step_with_invalid_action(self, mock_task_instance):
        """Test stepping with invalid actions."""
        env = NetHackEnvironment(mock_task_instance)
        await env.initialize()

        # Test invalid action
        obs = await env.step("invalid_action_xyz")
        assert "error" in obs
        assert "Unknown action" in obs["error"]

    @pytest.mark.asyncio
    async def test_tool_call_formats(self, mock_task_instance):
        """Test various tool call input formats."""
        env = NetHackEnvironment(mock_task_instance)
        await env.initialize()

        # Test dict with action key
        obs = await env.step({"action": "wait"})
        assert obs["last_action"] == "wait"

        # Test EnvToolCall format
        tool_call = EnvToolCall(tool="interact", args={"action": "search"})
        obs = await env.step(tool_call)
        assert obs["last_action"] == "search"

        # Test list format
        obs = await env.step([{"action": "inventory"}])
        assert obs["last_action"] == "inventory"

        # Test nested tool_calls format
        obs = await env.step({"tool_calls": [{"args": {"action": "look"}}]})
        assert obs["last_action"] == "look"

    @pytest.mark.asyncio
    async def test_checkpoint(self, mock_task_instance):
        """Test checkpoint functionality."""
        env = NetHackEnvironment(mock_task_instance)
        await env.initialize()

        # Take some actions
        await env.step("north")
        await env.step("east")

        # Create checkpoint
        checkpoint_obs = await env.checkpoint()

        assert "final_score" in checkpoint_obs
        assert "max_depth" in checkpoint_obs
        assert "turn_count_final" in checkpoint_obs
        assert "total_reward" in checkpoint_obs

    @pytest.mark.asyncio
    async def test_terminate(self, mock_task_instance):
        """Test environment termination."""
        env = NetHackEnvironment(mock_task_instance)
        await env.initialize()

        # Take an action
        await env.step("wait")

        # Terminate
        final_obs = await env.terminate()

        assert final_obs["terminated"] is True
        assert "final_score" in final_obs
        assert "total_reward" in final_obs

    @pytest.mark.asyncio
    async def test_validate_tool_calls_edge_cases(self, mock_task_instance):
        """Test tool call validation edge cases."""
        env = NetHackEnvironment(mock_task_instance)

        # Test empty list
        with pytest.raises(ValueError, match="Empty tool calls list"):
            env.validate_tool_calls([])

        # Test invalid format
        with pytest.raises(ValueError, match="Invalid tool call format"):
            env.validate_tool_calls(123)  # type: ignore[arg-type]  # Not a valid format

        # Test nested args
        call = env.validate_tool_calls({"args": {"action": "north"}})
        assert call.args["action"] == "north"

        # Test parameters key
        call = env.validate_tool_calls({"parameters": {"action": "south"}})
        assert call.args["action"] == "south"

    @pytest.mark.asyncio
    async def test_available_actions(self, mock_task_instance):
        """Test getting available actions."""
        env = NetHackEnvironment(mock_task_instance)

        actions = env.get_available_actions()
        assert isinstance(actions, list)
        assert "north" in actions
        assert "inventory" in actions
        assert "a" in actions  # Menu action

        descriptions = env.get_action_descriptions()
        assert isinstance(descriptions, dict)
        assert descriptions["north"] == "move north"
        assert descriptions["inventory"] == "check inventory"


class TestNetHackInteractTool:
    """Test cases for NetHack interact tool."""

    @pytest.fixture
    def mock_task_instance(self):
        """Create a mock task instance for testing."""
        metadata = NetHackTaskInstanceMetadata(
            character_role="knight",
            starting_level=1,
            target_depth=3,
            time_limit=500,
            difficulty="beginner",
            special_objectives=["Defeat 10 monsters"],
            seed=123,
        )

        return NetHackTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions="Test knight adventure"),
            intent=Intent(
                rubric={"goal": "Test objectives"},
                gold_trajectories=None,
                gold_state_diff={},
            ),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,
        )

    @pytest.fixture
    def mock_engine(self, mock_task_instance):
        """Create a mock engine for testing."""
        from synth_ai.environments.examples.nethack.engine import NetHackEngine

        return NetHackEngine(mock_task_instance)

    @pytest.mark.asyncio
    async def test_interact_tool_valid_action(self, mock_engine):
        """Test interact tool with valid action."""
        await mock_engine._reset_engine()
        tool = NetHackInteractTool(mock_engine)

        call = EnvToolCall(tool="interact", args={"action": "wait"})
        result = await tool(call)

        assert result.ok is True
        assert "public_state" in result.payload
        assert "private_state" in result.payload
        assert result.payload["public_state"].last_action == "wait"

    @pytest.mark.asyncio
    async def test_interact_tool_no_action(self, mock_engine):
        """Test interact tool with missing action."""
        await mock_engine._reset_engine()
        tool = NetHackInteractTool(mock_engine)

        call = EnvToolCall(tool="interact", args={})
        result = await tool(call)

        assert result.ok is False
        # KeyError is caught and returned as string
        assert "'action'" in result.error

    @pytest.mark.asyncio
    async def test_interact_tool_invalid_action(self, mock_engine):
        """Test interact tool with invalid action."""
        await mock_engine._reset_engine()
        tool = NetHackInteractTool(mock_engine)

        call = EnvToolCall(tool="interact", args={"action": "fly"})
        result = await tool(call)

        assert result.ok is False
        assert "Unknown action" in result.error

    @pytest.mark.asyncio
    async def test_interact_tool_game_over_validation(self, mock_engine):
        """Test interact tool validation when game is over."""
        await mock_engine._reset_engine()
        tool = NetHackInteractTool(mock_engine)

        # Manually terminate the game
        mock_engine.public_state.terminated = True
        mock_engine.private_state.terminated = True

        # Try non-quit action
        call = EnvToolCall(tool="interact", args={"action": "north"})
        result = await tool(call)

        assert result.ok is False
        assert "Game is over" in result.error
