from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List
import numpy as np

from uuid import uuid4
from synth_ai.environments.tasks.core import (
    TaskInstance,
    TaskInstanceMetadata,
    TaskInstanceSet,
    Impetus,
    Intent,
    SplitInfo,
)

from .engine import COORD_TO_IDX, WIN_PATTERNS, PLAYER_MARKS


@dataclass
class TicTacToeTaskInstanceMetadata(TaskInstanceMetadata):
    starting_player: str  # "X" or "O"
    opening_moves: List[str]  # Pre-made moves to create position
    optimal_outcome: str  # "win", "draw", "loss" for starting player
    position_complexity: int  # Number of pre-moves made
    shortest_win_length: int  # Min moves to force win/draw


@dataclass
class TicTacToeTaskInstance(TaskInstance):
    async def serialize(self) -> dict:
        return {
            "id": str(self.id),
            "impetus": {"instructions": self.impetus.instructions},
            "intent": {
                "rubric": self.intent.rubric,
                "gold_trajectories": self.intent.gold_trajectories,
                "gold_state_diff": self.intent.gold_state_diff,
            },
            "metadata": {
                "starting_player": self.metadata.starting_player,
                "opening_moves": self.metadata.opening_moves,
                "optimal_outcome": self.metadata.optimal_outcome,
                "position_complexity": self.metadata.position_complexity,
                "shortest_win_length": self.metadata.shortest_win_length,
            },
            "is_reproducible": self.is_reproducible,
            "initial_engine_snapshot": self.initial_engine_snapshot,
        }

    @classmethod
    async def deserialize(cls, data: dict) -> "TicTacToeTaskInstance":
        from uuid import UUID

        metadata = TicTacToeTaskInstanceMetadata(
            starting_player=data["metadata"]["starting_player"],
            opening_moves=data["metadata"]["opening_moves"],
            optimal_outcome=data["metadata"]["optimal_outcome"],
            position_complexity=data["metadata"]["position_complexity"],
            shortest_win_length=data["metadata"]["shortest_win_length"],
        )

        return cls(
            id=UUID(data["id"]),
            impetus=Impetus(instructions=data["impetus"]["instructions"]),
            intent=Intent(
                rubric=data["intent"]["rubric"],
                gold_trajectories=data["intent"]["gold_trajectories"],
                gold_state_diff=data["intent"]["gold_state_diff"],
            ),
            metadata=metadata,
            is_reproducible=data["is_reproducible"],
            initial_engine_snapshot=data["initial_engine_snapshot"],
        )


def _evaluate_position(board: np.ndarray, player: int) -> str:
    """Simple evaluation of position outcome with perfect play"""
    # Check for immediate win
    for pattern in WIN_PATTERNS:
        values = [board[i] for i in pattern]
        if values.count(player) == 3:
            return "win"
        if values.count(3 - player) == 3:
            return "loss"

    # Check if board is full
    if np.all(board != 0):
        return "draw"

    # For simplicity, assume draw for non-terminal positions
    # In a real implementation, this would use minimax
    return "draw"


def _count_shortest_win(board: np.ndarray, player: int) -> int:
    """Count minimum moves to force a win/draw"""
    # Simplified: return remaining empty cells
    empty_cells = sum(1 for i in range(9) if board[i] == 0)
    return max(1, empty_cells // 2)


async def create_tictactoe_taskset() -> TaskInstanceSet:
    """Generate diverse TicTacToe starting positions"""
    instances = []

    # Configuration for different position types
    POSITION_CONFIGS = {
        "opening": {"pre_moves": 0, "count": 10},  # Fresh games
        "early": {"pre_moves": 1, "count": 15},  # After 1 move
        "mid": {"pre_moves": 2, "count": 15},  # After 2 moves
        "complex": {"pre_moves": 3, "count": 10},  # After 3 moves
    }

    all_coords = list(COORD_TO_IDX.keys())

    for config_name, config in POSITION_CONFIGS.items():
        for i in range(config["count"]):
            # Generate random opening moves
            opening_moves = []
            board = np.zeros(9, dtype=int)
            current_player = "X"

            # Make pre-moves
            available_coords = all_coords.copy()
            for move_idx in range(config["pre_moves"]):
                if not available_coords:
                    break

                # Random move
                move = random.choice(available_coords)
                opening_moves.append(move)
                available_coords.remove(move)

                # Update board
                board[COORD_TO_IDX[move]] = PLAYER_MARKS[current_player]
                current_player = "O" if current_player == "X" else "X"

            # Evaluate position
            starting_player = current_player
            optimal_outcome = _evaluate_position(board, PLAYER_MARKS[starting_player])
            shortest_win = _count_shortest_win(board, PLAYER_MARKS[starting_player])

            # Create metadata
            metadata = TicTacToeTaskInstanceMetadata(
                starting_player=starting_player,
                opening_moves=opening_moves,
                optimal_outcome=optimal_outcome,
                position_complexity=config["pre_moves"],
                shortest_win_length=shortest_win,
            )

            # Create instance
            impetus = Impetus(
                instructions=(
                    f"You are playing TicTacToe as {starting_player}. "
                    + "The game is played on a 3x3 grid with cells labeled A1-A3, B1-B3, C1-C3. "
                    + (
                        f"The game has already had {len(opening_moves)} moves."
                        if opening_moves
                        else "This is a fresh game."
                    )
                    + f" You must place your mark ({starting_player}) in an empty cell. "
                    + "Win by getting three of your marks in a row (horizontally, vertically, or diagonally)."
                )
            )

            intent = Intent(
                rubric={"goal": f"Win the game as {starting_player}, or at least force a draw"},
                gold_trajectories=None,
                gold_state_diff={"optimal_outcome": optimal_outcome},
            )

            instance = TicTacToeTaskInstance(
                id=uuid4(),
                impetus=impetus,
                intent=intent,
                metadata=metadata,
                is_reproducible=True,
                initial_engine_snapshot=None,
            )

            instances.append(instance)

    # Shuffle instances
    random.shuffle(instances)

    # Define splits based on complexity
    val_ids = {inst.id for inst in instances if inst.metadata.position_complexity == 1}
    test_ids = {inst.id for inst in instances if inst.metadata.position_complexity >= 2}

    # If not enough instances for splits, use simple division
    if len(val_ids) == 0 or len(test_ids) == 0:
        total = len(instances)
        val_end = int(total * 0.15)
        test_end = int(total * 0.30)
        val_ids = {instances[i].id for i in range(val_end)}
        test_ids = {instances[i].id for i in range(val_end, test_end)}

    split_info = SplitInfo(
        val_instance_ids=val_ids, test_instance_ids=test_ids, _is_split_defined=True
    )

    return TaskInstanceSet(
        name="TicTacToe Procedural TaskSet",
        description="Procedurally generated TicTacToe tasks with varying starting positions.",
        instances=instances,
        split_info=split_info,
    )


# Make taskset available as module attribute
taskset = create_tictactoe_taskset
