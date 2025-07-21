import pytest
import numpy as np
from uuid import uuid4

from synth_ai.environments.tasks.core import TaskInstance, Impetus, Intent
from synth_ai.environments.examples.tictactoe.engine import (
    TicTacToeEngine,
    TicTacToePublicState,
    TicTacToePrivateState,
    TicTacToeWinComponent,
    TicTacToeDrawComponent,
    TicTacToeIllegalMoveComponent,
    COORD_TO_IDX,
    IDX_TO_COORD,
    WIN_PATTERNS,
    PLAYER_MARKS,
    MARK_TO_PLAYER,
)
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


@pytest.fixture
def task_with_premoves():
    """Create a task instance with pre-moves."""
    metadata = TicTacToeTaskInstanceMetadata(
        starting_player="O",
        opening_moves=["A1", "B2"],
        optimal_outcome="win",
        position_complexity=2,
        shortest_win_length=3,
    )

    return TicTacToeTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="Test TicTacToe with premoves"),
        intent=Intent(rubric={"goal": "Test game"}, gold_trajectories=None, gold_state_diff={}),
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
    )


class TestTicTacToeEngine:
    @pytest.mark.asyncio
    async def test_engine_initialization(self, simple_task_instance):
        """Test engine initializes correctly."""
        engine = TicTacToeEngine(simple_task_instance)

        assert engine.current_player == "X"
        assert engine.move_count == 0
        assert engine.winner is None
        assert not engine.terminated
        assert engine.total_reward == 0.0
        assert np.all(engine.board == 0)

    @pytest.mark.asyncio
    async def test_engine_with_premoves(self, task_with_premoves):
        """Test engine applies pre-moves correctly."""
        engine = TicTacToeEngine(task_with_premoves)

        # Check pre-moves were applied
        assert engine.board[COORD_TO_IDX["A1"]] == PLAYER_MARKS["X"]
        assert engine.board[COORD_TO_IDX["B2"]] == PLAYER_MARKS["O"]
        assert engine.current_player == "X"  # After 2 moves, back to X
        assert engine.move_count == 2

    @pytest.mark.asyncio
    async def test_reset_engine(self, simple_task_instance):
        """Test engine reset functionality."""
        engine = TicTacToeEngine(simple_task_instance)

        # Make a move first
        await engine._step_engine("B2")

        # Reset
        priv, pub = await engine._reset_engine()

        assert pub.current_player == "X"
        assert pub.move_count == 0
        assert pub.winner is None
        assert not pub.terminated
        assert np.all(pub.board == 0)
        assert priv.total_reward == 0.0

    @pytest.mark.asyncio
    async def test_valid_moves(self, simple_task_instance):
        """Test making valid moves."""
        engine = TicTacToeEngine(simple_task_instance)

        # Make first move
        priv, pub = await engine._step_engine("B2")

        assert pub.last_move == "B2"
        assert pub.board[COORD_TO_IDX["B2"]] == PLAYER_MARKS["X"]
        assert pub.current_player == "O"
        assert pub.move_count == 1
        assert not pub.terminated

        # Make second move
        priv, pub = await engine._step_engine("A1")

        assert pub.last_move == "A1"
        assert pub.board[COORD_TO_IDX["A1"]] == PLAYER_MARKS["O"]
        assert pub.current_player == "X"
        assert pub.move_count == 2

    @pytest.mark.asyncio
    async def test_invalid_moves(self, simple_task_instance):
        """Test handling of invalid moves."""
        engine = TicTacToeEngine(simple_task_instance)

        # Make a valid move first
        await engine._step_engine("B2")

        # Try to make move in occupied cell
        priv, pub = await engine._step_engine("B2")

        assert pub.terminated
        assert priv.reward_last == -1.0  # Illegal move penalty

        # Test invalid coordinate
        engine = TicTacToeEngine(simple_task_instance)
        priv, pub = await engine._step_engine("Z9")

        assert pub.terminated
        assert priv.reward_last == -1.0

    @pytest.mark.asyncio
    async def test_win_detection_row(self, simple_task_instance):
        """Test detecting wins in rows."""
        engine = TicTacToeEngine(simple_task_instance)

        # X wins in top row
        await engine._step_engine("A1")  # X
        await engine._step_engine("B1")  # O
        await engine._step_engine("A2")  # X
        await engine._step_engine("B2")  # O
        priv, pub = await engine._step_engine("A3")  # X wins

        assert pub.winner == "X"
        assert pub.terminated
        assert priv.reward_last == 1.0  # Win reward

    @pytest.mark.asyncio
    async def test_win_detection_column(self, simple_task_instance):
        """Test detecting wins in columns."""
        engine = TicTacToeEngine(simple_task_instance)

        # X wins in first column
        await engine._step_engine("A1")  # X
        await engine._step_engine("A2")  # O
        await engine._step_engine("B1")  # X
        await engine._step_engine("B2")  # O
        priv, pub = await engine._step_engine("C1")  # X wins

        assert pub.winner == "X"
        assert pub.terminated

    @pytest.mark.asyncio
    async def test_win_detection_diagonal(self, simple_task_instance):
        """Test detecting wins in diagonals."""
        engine = TicTacToeEngine(simple_task_instance)

        # X wins in main diagonal
        await engine._step_engine("A1")  # X
        await engine._step_engine("A2")  # O
        await engine._step_engine("B2")  # X
        await engine._step_engine("B1")  # O
        priv, pub = await engine._step_engine("C3")  # X wins

        assert pub.winner == "X"
        assert pub.terminated

    @pytest.mark.asyncio
    async def test_draw_detection(self, simple_task_instance):
        """Test detecting draws."""
        engine = TicTacToeEngine(simple_task_instance)

        # Play a game that ends in draw
        moves = ["A1", "B2", "A2", "A3", "B3", "B1", "C1", "C3", "C2"]
        for move in moves:
            priv, pub = await engine._step_engine(move)

        assert pub.winner == "draw"
        assert pub.terminated
        assert pub.move_count == 9
        assert priv.reward_last == 0.0  # Draw reward

    @pytest.mark.asyncio
    async def test_board_text_representation(self, simple_task_instance):
        """Test board text representation."""
        engine = TicTacToeEngine(simple_task_instance)

        await engine._step_engine("B2")
        await engine._step_engine("A1")

        priv, pub = engine.get_current_states_for_observation()
        board_text = pub.board_text

        assert "  A B C" in board_text
        assert "1 O    " in board_text
        assert "2   X  " in board_text
        assert "3      " in board_text

    @pytest.mark.asyncio
    async def test_serialization(self, simple_task_instance):
        """Test engine serialization and deserialization."""
        engine = TicTacToeEngine(simple_task_instance)

        # Make some moves
        await engine._step_engine("B2")
        await engine._step_engine("A1")

        # Serialize
        snapshot = await engine._serialize_engine()

        assert snapshot.engine_snapshot["current_player"] == "X"
        assert snapshot.engine_snapshot["move_count"] == 2
        assert snapshot.engine_snapshot["last_move"] == "A1"

        # Deserialize
        restored_engine = await TicTacToeEngine._deserialize_engine(snapshot)

        assert restored_engine.current_player == engine.current_player
        assert restored_engine.move_count == engine.move_count
        assert np.array_equal(restored_engine.board, engine.board)

    @pytest.mark.asyncio
    async def test_state_diff(self, simple_task_instance):
        """Test state diff functionality."""
        engine = TicTacToeEngine(simple_task_instance)

        priv1, pub1 = await engine._reset_engine()
        priv2, pub2 = await engine._step_engine("B2")

        # Test public state diff
        diff = pub2.diff(pub1)
        assert "board" in diff
        assert "current_player" in diff
        assert "last_move" in diff
        assert "move_count" in diff

        # Test private state diff
        priv_diff = priv2.diff(priv1)
        # reward_last might be 0.0 in both states, so it won't appear in diff
        # Check that diff works by modifying reward
        priv2.reward_last = 1.0
        priv_diff = priv2.diff(priv1)
        assert "reward_last" in priv_diff


class TestRewardComponents:
    @pytest.mark.asyncio
    async def test_win_component(self):
        """Test win reward component."""
        component = TicTacToeWinComponent(player_mark="X")

        # Test win for X
        state = TicTacToePublicState(
            board=np.zeros(9),
            current_player="O",
            last_move="A3",
            winner="X",
            move_count=5,
            max_moves=9,
            terminated=True,
        )
        score = await component.score(state, "A3")
        assert score == 1.0

        # Test loss (O wins)
        state.winner = "O"
        score = await component.score(state, "A3")
        assert score == -1.0

        # Test no winner yet
        state.winner = None
        score = await component.score(state, "A3")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_draw_component(self):
        """Test draw reward component."""
        component = TicTacToeDrawComponent()

        state = TicTacToePublicState(
            board=np.ones(9),
            current_player="X",
            last_move="C3",
            winner="draw",
            move_count=9,
            max_moves=9,
            terminated=True,
        )

        score = await component.score(state, "C3")
        assert score == 0.0

        # Test non-draw
        state.winner = "X"
        score = await component.score(state, "C3")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_illegal_move_component(self):
        """Test illegal move reward component."""
        component = TicTacToeIllegalMoveComponent()

        state = TicTacToePublicState(
            board=np.zeros(9),
            current_player="X",
            last_move="A1",
            winner=None,
            move_count=1,
            max_moves=9,
            terminated=False,
        )

        # Test no illegal move
        score = await component.score(state, "A1")
        assert score == 0.0

        # Test illegal move
        component.illegal_move_attempted = True
        score = await component.score(state, "A1")
        assert score == -1.0
        assert not component.illegal_move_attempted  # Should reset


class TestConstants:
    def test_coordinate_mappings(self):
        """Test coordinate to index mappings."""
        assert len(COORD_TO_IDX) == 9
        assert len(IDX_TO_COORD) == 9

        # Test all coordinates map correctly
        for coord, idx in COORD_TO_IDX.items():
            assert IDX_TO_COORD[idx] == coord

        # Test specific mappings
        assert COORD_TO_IDX["A1"] == 0
        assert COORD_TO_IDX["B2"] == 4
        assert COORD_TO_IDX["C3"] == 8

    def test_win_patterns(self):
        """Test win patterns cover all possibilities."""
        assert len(WIN_PATTERNS) == 8  # 3 rows, 3 cols, 2 diagonals

        # Test rows
        assert [0, 1, 2] in WIN_PATTERNS
        assert [3, 4, 5] in WIN_PATTERNS
        assert [6, 7, 8] in WIN_PATTERNS

        # Test columns
        assert [0, 3, 6] in WIN_PATTERNS
        assert [1, 4, 7] in WIN_PATTERNS
        assert [2, 5, 8] in WIN_PATTERNS

        # Test diagonals
        assert [0, 4, 8] in WIN_PATTERNS
        assert [2, 4, 6] in WIN_PATTERNS

    def test_player_mappings(self):
        """Test player mark mappings."""
        assert PLAYER_MARKS["X"] == 1
        assert PLAYER_MARKS["O"] == 2
        assert MARK_TO_PLAYER[0] == " "
        assert MARK_TO_PLAYER[1] == "X"
        assert MARK_TO_PLAYER[2] == "O"
