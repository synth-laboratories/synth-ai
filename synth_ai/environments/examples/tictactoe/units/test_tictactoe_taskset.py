import pytest
import numpy as np
from uuid import UUID

from synth_ai.environments.examples.tictactoe.taskset import (
    create_tictactoe_taskset,
    TicTacToeTaskInstance,
    TicTacToeTaskInstanceMetadata,
    _evaluate_position,
    _count_shortest_win,
    COORD_TO_IDX,
    PLAYER_MARKS,
)


class TestTasksetGeneration:
    @pytest.mark.asyncio
    async def test_create_taskset(self):
        """Test taskset creation."""
        taskset = await create_tictactoe_taskset()

        assert taskset.name == "TicTacToe Procedural TaskSet"
        assert len(taskset.instances) == 50  # 10 + 15 + 15 + 10
        assert taskset.split_info._is_split_defined

        # Check that we have instances of each complexity
        complexities = [inst.metadata.position_complexity for inst in taskset.instances]
        assert 0 in complexities  # opening positions
        assert 1 in complexities  # early positions
        assert 2 in complexities  # mid positions
        assert 3 in complexities  # complex positions

    @pytest.mark.asyncio
    async def test_task_instance_metadata(self):
        """Test task instance metadata."""
        taskset = await create_tictactoe_taskset()

        for instance in taskset.instances:
            metadata = instance.metadata

            # Check metadata fields
            assert metadata.starting_player in ["X", "O"]
            assert isinstance(metadata.opening_moves, list)
            assert metadata.optimal_outcome in ["win", "draw", "loss"]
            assert metadata.position_complexity >= 0
            assert metadata.shortest_win_length >= 1

            # Check opening moves match complexity
            assert len(metadata.opening_moves) == metadata.position_complexity

    @pytest.mark.asyncio
    async def test_task_instance_structure(self):
        """Test task instance structure."""
        taskset = await create_tictactoe_taskset()
        instance = taskset.instances[0]

        # Check instance has required attributes
        assert isinstance(instance.id, UUID)
        assert instance.impetus is not None
        assert instance.intent is not None
        assert instance.metadata is not None
        assert instance.is_reproducible == True
        assert instance.initial_engine_snapshot is None

        # Check impetus
        assert "TicTacToe" in instance.impetus.instructions
        assert instance.metadata.starting_player in instance.impetus.instructions

        # Check intent
        assert "goal" in instance.intent.rubric
        assert instance.intent.gold_trajectories is None
        assert isinstance(instance.intent.gold_state_diff, dict)

    @pytest.mark.asyncio
    async def test_splits(self):
        """Test train/val/test splits."""
        taskset = await create_tictactoe_taskset()

        val_ids = taskset.split_info.val_instance_ids
        test_ids = taskset.split_info.test_instance_ids
        all_ids = {inst.id for inst in taskset.instances}

        # Check splits are disjoint
        assert len(val_ids & test_ids) == 0

        # Check splits are subsets of all instances
        assert val_ids.issubset(all_ids)
        assert test_ids.issubset(all_ids)

        # Check we have some instances in each split
        assert len(val_ids) > 0
        assert len(test_ids) > 0

        # Train should be everything not in val/test
        train_ids = all_ids - val_ids - test_ids
        assert len(train_ids) > 0

    @pytest.mark.asyncio
    async def test_serialization(self):
        """Test task instance serialization."""
        taskset = await create_tictactoe_taskset()
        instance = taskset.instances[0]

        # Serialize
        data = await instance.serialize()

        assert "id" in data
        assert "impetus" in data
        assert "intent" in data
        assert "metadata" in data
        assert "is_reproducible" in data

        # Check metadata serialization
        assert data["metadata"]["starting_player"] == instance.metadata.starting_player
        assert data["metadata"]["opening_moves"] == instance.metadata.opening_moves

        # Deserialize
        restored = await TicTacToeTaskInstance.deserialize(data)

        assert str(restored.id) == str(instance.id)
        assert restored.impetus.instructions == instance.impetus.instructions
        assert restored.metadata.starting_player == instance.metadata.starting_player
        assert restored.metadata.opening_moves == instance.metadata.opening_moves

    @pytest.mark.asyncio
    async def test_opening_moves_validity(self):
        """Test that opening moves are valid."""
        taskset = await create_tictactoe_taskset()

        for instance in taskset.instances:
            # Check all moves are valid coordinates
            for move in instance.metadata.opening_moves:
                assert move in COORD_TO_IDX

            # Check no duplicate moves
            assert len(instance.metadata.opening_moves) == len(set(instance.metadata.opening_moves))

            # Simulate the moves to check they're valid
            board = np.zeros(9, dtype=int)
            current_player = "X"

            for move in instance.metadata.opening_moves:
                idx = COORD_TO_IDX[move]
                assert board[idx] == 0  # Cell should be empty
                board[idx] = PLAYER_MARKS[current_player]
                current_player = "O" if current_player == "X" else "X"


class TestHelperFunctions:
    def test_evaluate_position_wins(self):
        """Test position evaluation for wins."""
        # X wins in top row
        board = np.array([1, 1, 1, 2, 2, 0, 0, 0, 0])
        assert _evaluate_position(board, 1) == "win"
        assert _evaluate_position(board, 2) == "loss"

        # O wins in first column
        board = np.array([2, 1, 1, 2, 1, 0, 2, 0, 0])
        assert _evaluate_position(board, 1) == "loss"
        assert _evaluate_position(board, 2) == "win"

    def test_evaluate_position_draw(self):
        """Test position evaluation for draws."""
        # Full board with no winner - fixed to actually be a draw
        # X O X
        # X O X
        # O X O
        board = np.array([1, 2, 1, 1, 2, 1, 2, 1, 2])
        assert _evaluate_position(board, 1) == "draw"
        assert _evaluate_position(board, 2) == "draw"

    def test_evaluate_position_ongoing(self):
        """Test position evaluation for ongoing games."""
        # Game still in progress
        board = np.array([1, 2, 0, 0, 1, 0, 0, 0, 0])
        # For simplicity, our implementation returns "draw" for non-terminal
        assert _evaluate_position(board, 1) == "draw"

    def test_count_shortest_win(self):
        """Test shortest win calculation."""
        # Empty board
        board = np.zeros(9, dtype=int)
        assert _count_shortest_win(board, 1) == 4  # 9 empty cells / 2

        # Partially filled board
        board = np.array([1, 2, 0, 0, 1, 0, 0, 0, 0])
        assert _count_shortest_win(board, 1) == 3  # 6 empty cells / 2

        # Almost full board
        board = np.array([1, 2, 1, 2, 1, 2, 2, 1, 0])
        assert _count_shortest_win(board, 1) == 1  # max(1, 1/2)
