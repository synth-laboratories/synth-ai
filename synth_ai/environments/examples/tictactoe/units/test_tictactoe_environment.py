import pytest
import numpy as np
from uuid import uuid4

from synth_ai.environments.environment.tools import EnvToolCall, ToolResult
from synth_ai.environments.tasks.core import TaskInstance, Impetus, Intent
from synth_ai.environments.examples.tictactoe.environment import (
    TicTacToeEnvironment,
    TicTacToeActionInput,
    TicTacToeInteractTool,
)
from synth_ai.environments.examples.tictactoe.engine import TicTacToeEngine
from synth_ai.environments.examples.tictactoe.taskset import (
    TicTacToeTaskInstance,
    TicTacToeTaskInstanceMetadata,
)


@pytest.fixture
def simple_task_instance():
    """Create a simple task instance for testing."""
    metadata = TicTacToeTaskInstanceMetadata(
        starting_player="X",
        opening_moves=[],
        optimal_outcome="draw",
        position_complexity=0,
        shortest_win_length=5,
    )

    return TicTacToeTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="Test TicTacToe game"),
        intent=Intent(rubric={"goal": "Test game"}, gold_trajectories=None, gold_state_diff={}),
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
    )


class TestTicTacToeEnvironment:
    @pytest.mark.asyncio
    async def test_environment_initialization(self, simple_task_instance):
        """Test environment initializes correctly."""
        env = TicTacToeEnvironment(simple_task_instance)

        assert env.name == "TicTacToe"
        assert env.task_instance == simple_task_instance
        assert env.engine is not None
        assert env._interact_tool is not None

        # Test initial observation
        obs = await env.initialize()

        assert "board_text" in obs
        assert "current_player" in obs
        assert obs["current_player"] == "X"
        assert obs["move_count"] == 0
        assert obs["terminated"] == False

    @pytest.mark.asyncio
    async def test_step_with_valid_move(self, simple_task_instance):
        """Test stepping with valid moves."""
        env = TicTacToeEnvironment(simple_task_instance)
        await env.initialize()

        # Test dictionary format
        obs = await env.step({"action": "B2"})

        assert obs["last_move"] == "B2"
        assert obs["current_player"] == "O"
        assert obs["move_count"] == 1
        assert "X" in obs["board_text"]

        # Test EnvToolCall format
        tool_call = EnvToolCall(tool="interact", args={"action": "A1"})
        obs = await env.step(tool_call)

        assert obs["last_move"] == "A1"
        assert obs["current_player"] == "X"
        assert obs["move_count"] == 2

    @pytest.mark.asyncio
    async def test_step_with_invalid_move(self, simple_task_instance):
        """Test stepping with invalid moves."""
        env = TicTacToeEnvironment(simple_task_instance)
        await env.initialize()

        # Make a valid move first
        await env.step({"action": "B2"})

        # Try to make move in occupied cell
        obs = await env.step({"action": "B2"})

        assert obs["terminated"] == True
        assert obs["reward_last"] == -1.0

    @pytest.mark.asyncio
    async def test_checkpoint(self, simple_task_instance):
        """Test checkpoint functionality."""
        env = TicTacToeEnvironment(simple_task_instance)
        await env.initialize()

        # Make some moves
        await env.step({"action": "B2"})
        await env.step({"action": "A1"})

        # Get checkpoint
        checkpoint = await env.checkpoint()

        assert "board_text_final" in checkpoint
        assert "winner_final" in checkpoint
        assert "move_count_final" in checkpoint
        assert checkpoint["move_count_final"] == 2
        assert checkpoint["total_reward"] == 0.0

    @pytest.mark.asyncio
    async def test_terminate(self, simple_task_instance):
        """Test terminate functionality."""
        env = TicTacToeEnvironment(simple_task_instance)
        await env.initialize()

        obs = await env.terminate()

        assert obs["terminated"] == True
        assert "board_text_final" in obs

    @pytest.mark.asyncio
    async def test_validate_tool_calls_various_formats(self, simple_task_instance):
        """Test tool call validation with various input formats."""
        env = TicTacToeEnvironment(simple_task_instance)

        # Test EnvToolCall format
        call = EnvToolCall(tool="interact", args={"action": "A1"})
        validated = env.validate_tool_calls(call)
        assert validated.tool == "interact"
        assert validated.args["letter"] == "A"
        assert validated.args["number"] == 1

        # Test dict with tool/args
        validated = env.validate_tool_calls({"tool": "interact", "args": {"action": "B2"}})
        assert validated.tool == "interact"
        assert validated.args["letter"] == "B"
        assert validated.args["number"] == 2

        # Test dict with name/parameters (legacy)
        validated = env.validate_tool_calls({"name": "interact", "parameters": {"action": "C3"}})
        assert validated.tool == "interact"
        assert validated.args["letter"] == "C"
        assert validated.args["number"] == 3

        # Test OpenAI function format
        validated = env.validate_tool_calls(
            {"function": {"name": "interact", "arguments": {"action": "A2"}}}
        )
        assert validated.tool == "interact"
        assert validated.args["letter"] == "A"
        assert validated.args["number"] == 2

        # Test bare dict (assumed to be args)
        validated = env.validate_tool_calls({"action": "B1"})
        assert validated.tool == "interact"
        assert validated.args["letter"] == "B"
        assert validated.args["number"] == 1

        # Test list format
        validated = env.validate_tool_calls([{"tool": "interact", "args": {"action": "C1"}}])
        assert validated.tool == "interact"
        assert validated.args["letter"] == "C"
        assert validated.args["number"] == 1

        # Test string conversion
        validated = env.validate_tool_calls("A3")
        assert validated.tool == "interact"
        assert validated.args["letter"] == "A"
        assert validated.args["number"] == 3

    @pytest.mark.asyncio
    async def test_validate_tool_calls_errors(self, simple_task_instance):
        """Test tool call validation error cases."""
        env = TicTacToeEnvironment(simple_task_instance)

        # Test wrong tool name
        with pytest.raises(ValueError, match="Unknown tool"):
            env.validate_tool_calls({"tool": "wrong_tool", "args": {}})

        # Test empty list
        with pytest.raises(ValueError, match="Empty tool calls list"):
            env.validate_tool_calls([])

    @pytest.mark.asyncio
    async def test_serialization(self, simple_task_instance):
        """Test environment serialization and deserialization."""
        env = TicTacToeEnvironment(simple_task_instance)
        await env.initialize()

        # Make some moves
        await env.step({"action": "B2"})
        await env.step({"action": "A1"})

        # Serialize
        snapshot = await env._serialize_engine()

        # Deserialize
        restored_env = await TicTacToeEnvironment._deserialize_engine(
            snapshot, simple_task_instance
        )

        # Check state is preserved
        assert restored_env.engine.current_player == env.engine.current_player
        assert restored_env.engine.move_count == env.engine.move_count
        assert np.array_equal(restored_env.engine.board, env.engine.board)

    @pytest.mark.asyncio
    async def test_full_game_to_win(self, simple_task_instance):
        """Test playing a full game to win."""
        env = TicTacToeEnvironment(simple_task_instance)
        await env.initialize()

        # X wins in top row
        moves = [
            ("A1", "X", "O"),
            ("B1", "O", "X"),
            ("A2", "X", "O"),
            ("B2", "O", "X"),
            ("A3", "X", None),  # X wins
        ]

        for move, player_before, player_after in moves:
            obs = await env.step({"action": move})

            if player_after:
                assert obs["current_player"] == player_after
            else:
                assert obs["terminated"] == True
                assert obs["winner"] == "X"
                assert obs["reward_last"] == 1.0

    @pytest.mark.asyncio
    async def test_full_game_to_draw(self, simple_task_instance):
        """Test playing a full game to draw."""
        env = TicTacToeEnvironment(simple_task_instance)
        await env.initialize()

        # Play a draw game
        moves = ["A1", "B2", "A2", "A3", "B3", "B1", "C1", "C3", "C2"]

        for i, move in enumerate(moves):
            obs = await env.step({"action": move})

            if i < len(moves) - 1:
                assert not obs["terminated"]
            else:
                assert obs["terminated"]
                assert obs["winner"] == "draw"
                assert obs["move_count"] == 9
                assert obs["reward_last"] == 0.0


class TestTicTacToeInteractTool:
    @pytest.mark.asyncio
    async def test_interact_tool_valid_action(self, simple_task_instance):
        """Test interact tool with valid action."""
        engine = TicTacToeEngine(simple_task_instance)
        tool = TicTacToeInteractTool(engine)

        call = EnvToolCall(tool="interact", args={"letter": "B", "number": 2})
        result = await tool(call)

        assert result.ok
        assert "public_state" in result.payload
        assert "private_state" in result.payload

        pub_state = result.payload["public_state"]
        assert pub_state.last_move == "B2"
        assert pub_state.current_player == "O"

    @pytest.mark.asyncio
    async def test_interact_tool_invalid_action(self, simple_task_instance):
        """Test interact tool with invalid action."""
        engine = TicTacToeEngine(simple_task_instance)
        tool = TicTacToeInteractTool(engine)

        # Make a move first
        await tool(EnvToolCall(tool="interact", args={"letter": "B", "number": 2}))

        # Try same cell again
        call = EnvToolCall(tool="interact", args={"letter": "B", "number": 2})
        result = await tool(call)

        assert result.ok
        assert result.payload["public_state"].terminated
        assert result.payload["private_state"].reward_last == -1.0

    @pytest.mark.asyncio
    async def test_interact_tool_no_action(self, simple_task_instance):
        """Test interact tool with missing action."""
        engine = TicTacToeEngine(simple_task_instance)
        tool = TicTacToeInteractTool(engine)

        call = EnvToolCall(tool="interact", args={})
        result = await tool(call)

        assert not result.ok
        assert result.error == "Both letter and number parameters are required"

    @pytest.mark.asyncio
    async def test_interact_tool_exception_handling(self, simple_task_instance):
        """Test interact tool exception handling."""
        engine = TicTacToeEngine(simple_task_instance)
        tool = TicTacToeInteractTool(engine)

        # Force an exception by passing invalid data type
        call = EnvToolCall(tool="interact", args={"letter": "A", "number": None})
        result = await tool(call)

        assert not result.ok
        assert result.error == "Both letter and number parameters are required"


class TestTicTacToeActionInput:
    def test_action_input_model(self):
        """Test TicTacToeActionInput Pydantic model."""
        # Valid input
        input_model = TicTacToeActionInput(letter="A", number=1)
        assert input_model.letter == "A"
        assert input_model.number == 1

        # Test schema
        schema = TicTacToeActionInput.model_json_schema()
        assert "properties" in schema
        assert "letter" in schema["properties"]
        assert "number" in schema["properties"]
        assert schema["properties"]["letter"]["type"] == "string"
        assert schema["properties"]["number"]["type"] == "integer"


class TestTicTacToeValidation:
    """Test validation fixes for TicTacToe environment."""

    @pytest.mark.asyncio
    async def test_position_validation_valid_positions(self, simple_task_instance):
        """Test that valid positions (0-8) are correctly converted."""
        env = TicTacToeEnvironment(simple_task_instance)

        # Test all valid positions
        valid_positions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        expected_conversions = [
            ("A", 1),
            ("A", 2),
            ("A", 3),
            ("B", 1),
            ("B", 2),
            ("B", 3),
            ("C", 1),
            ("C", 2),
            ("C", 3),
        ]

        for pos, (expected_letter, expected_number) in zip(valid_positions, expected_conversions):
            validated_call = env.validate_tool_calls(
                {"tool": "interact", "args": {"position": pos}}
            )
            assert validated_call.args["letter"] == expected_letter
            assert validated_call.args["number"] == expected_number

    @pytest.mark.asyncio
    async def test_position_validation_invalid_positions(self, simple_task_instance):
        """Test that invalid positions are properly rejected."""
        env = TicTacToeEnvironment(simple_task_instance)

        invalid_positions = [-1, 9, 10, 100, -5]

        for pos in invalid_positions:
            with pytest.raises(ValueError, match=f"Position {pos} must be between 0 and 8"):
                env.validate_tool_calls({"tool": "interact", "args": {"position": pos}})

    @pytest.mark.asyncio
    async def test_coordinate_validation_valid_coordinates(self, simple_task_instance):
        """Test that valid coordinate strings are correctly converted."""
        env = TicTacToeEnvironment(simple_task_instance)

        valid_coords = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]

        for coord in valid_coords:
            validated_call = env.validate_tool_calls(
                {"tool": "interact", "args": {"action": coord}}
            )
            expected_letter = coord[0]
            expected_number = int(coord[1])
            assert validated_call.args["letter"] == expected_letter
            assert validated_call.args["number"] == expected_number

    @pytest.mark.asyncio
    async def test_coordinate_validation_invalid_coordinates(self, simple_task_instance):
        """Test that invalid coordinate strings are properly rejected."""
        env = TicTacToeEnvironment(simple_task_instance)

        # Test invalid numbers
        with pytest.raises(ValueError, match="Number must be 1, 2, or 3, got 0"):
            env.validate_tool_calls({"tool": "interact", "args": {"action": "A0"}})

        with pytest.raises(ValueError, match="Number must be 1, 2, or 3, got 4"):
            env.validate_tool_calls({"tool": "interact", "args": {"action": "A4"}})

        # Test invalid letters
        with pytest.raises(ValueError, match="Letter must be A, B, or C, got 'D'"):
            env.validate_tool_calls({"tool": "interact", "args": {"action": "D1"}})

        with pytest.raises(ValueError, match="Letter must be A, B, or C, got 'Z'"):
            env.validate_tool_calls({"tool": "interact", "args": {"action": "Z9"}})

        # Test invalid format
        with pytest.raises(ValueError, match="Action '' must be 2 characters"):
            env.validate_tool_calls({"tool": "interact", "args": {"action": ""}})

        with pytest.raises(ValueError, match="Action '1A' must have a numeric second character"):
            env.validate_tool_calls({"tool": "interact", "args": {"action": "1A"}})

        with pytest.raises(ValueError, match="Action 'BB' must have a numeric second character"):
            env.validate_tool_calls({"tool": "interact", "args": {"action": "BB"}})

    @pytest.mark.asyncio
    async def test_direct_letter_number_validation_valid(self, simple_task_instance):
        """Test that valid letter/number combinations work."""
        env = TicTacToeEnvironment(simple_task_instance)

        valid_combinations = [
            ("A", 1),
            ("A", 2),
            ("A", 3),
            ("B", 1),
            ("B", 2),
            ("B", 3),
            ("C", 1),
            ("C", 2),
            ("C", 3),
        ]

        for letter, number in valid_combinations:
            validated_call = env.validate_tool_calls(
                {"tool": "interact", "args": {"letter": letter, "number": number}}
            )
            assert validated_call.args["letter"] == letter
            assert validated_call.args["number"] == number

    @pytest.mark.asyncio
    async def test_direct_letter_number_validation_invalid(self, simple_task_instance):
        """Test that invalid letter/number combinations are rejected."""
        env = TicTacToeEnvironment(simple_task_instance)

        # Test invalid numbers
        with pytest.raises(ValueError, match="Number must be 1, 2, or 3, got 0"):
            env.validate_tool_calls({"tool": "interact", "args": {"letter": "A", "number": 0}})

        with pytest.raises(ValueError, match="Number must be 1, 2, or 3, got 4"):
            env.validate_tool_calls({"tool": "interact", "args": {"letter": "A", "number": 4}})

        # Test invalid letters
        with pytest.raises(ValueError, match="Letter must be A, B, or C, got 'D'"):
            env.validate_tool_calls({"tool": "interact", "args": {"letter": "D", "number": 1}})

        with pytest.raises(ValueError, match="Letter must be A, B, or C, got 'a'"):
            env.validate_tool_calls({"tool": "interact", "args": {"letter": "a", "number": 1}})

        with pytest.raises(ValueError, match="Letter must be A, B, or C, got ''"):
            env.validate_tool_calls({"tool": "interact", "args": {"letter": "", "number": 1}})

        with pytest.raises(ValueError, match="Letter must be A, B, or C, got 'AB'"):
            env.validate_tool_calls({"tool": "interact", "args": {"letter": "AB", "number": 1}})

    @pytest.mark.asyncio
    async def test_tool_validates_before_execution(self, simple_task_instance):
        """Test that the tool validates inputs before execution."""
        env = TicTacToeEnvironment(simple_task_instance)

        # Test invalid letter directly on tool
        result = await env._interact_tool(
            EnvToolCall(tool="interact", args={"letter": "Z", "number": 1})
        )
        assert not result.ok
        assert "Letter must be A, B, or C, got 'Z'" in result.error

        # Test invalid number directly on tool
        result = await env._interact_tool(
            EnvToolCall(tool="interact", args={"letter": "A", "number": 0})
        )
        assert not result.ok
        assert "Number must be 1, 2, or 3, got 0" in result.error

        # Test missing parameters
        result = await env._interact_tool(EnvToolCall(tool="interact", args={}))
        assert not result.ok
        assert "Both letter and number parameters are required" in result.error
