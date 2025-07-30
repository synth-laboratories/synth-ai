from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from synth_ai.environments.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from synth_ai.environments.reproducibility.core import IReproducibleEngine
from synth_ai.environments.environment.rewards.core import RewardStack, RewardComponent
from synth_ai.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_ai.environments.tasks.core import TaskInstance


# Action mapping: coordinate strings to board indices
COORD_TO_IDX = {
    "A1": 0,
    "A2": 1,
    "A3": 2,
    "B1": 3,
    "B2": 4,
    "B3": 5,
    "C1": 6,
    "C2": 7,
    "C3": 8,
}
IDX_TO_COORD = {v: k for k, v in COORD_TO_IDX.items()}

# Win condition patterns (row, col, diagonal indices)
WIN_PATTERNS = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],  # rows
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],  # columns
    [0, 4, 8],
    [2, 4, 6],  # diagonals
]

# Player mappings
PLAYER_MARKS = {"X": 1, "O": 2}
MARK_TO_PLAYER = {1: "X", 2: "O", 0: " "}


@dataclass
class TicTacToePublicState:
    board: np.ndarray  # 3x3 array: 0=empty, 1=X, 2=O
    current_player: str  # "X" or "O"
    last_move: Optional[str]  # "A1", "B2", etc.
    winner: Optional[str]  # None, "X", "O", or "draw"
    move_count: int  # Number of moves made
    max_moves: int  # Always 9 for TicTacToe
    terminated: bool  # Game finished

    def diff(self, prev_state: "TicTacToePublicState") -> Dict[str, Any]:
        differences = {}
        if not np.array_equal(self.board, prev_state.board):
            differences["board"] = self.board.tolist()
        if self.current_player != prev_state.current_player:
            differences["current_player"] = self.current_player
        if self.last_move != prev_state.last_move:
            differences["last_move"] = self.last_move
        if self.winner != prev_state.winner:
            differences["winner"] = self.winner
        if self.move_count != prev_state.move_count:
            differences["move_count"] = self.move_count
        if self.terminated != prev_state.terminated:
            differences["terminated"] = self.terminated
        return differences

    @property
    def board_text(self) -> str:
        lines = []
        lines.append("  A B C")
        for i in range(3):
            row_marks = []
            for j in range(3):
                mark = MARK_TO_PLAYER[self.board[i * 3 + j]]
                row_marks.append(mark)
            lines.append(f"{i + 1} {' '.join(row_marks)}")
        return "\n".join(lines)


@dataclass
class TicTacToePrivateState:
    reward_last: float
    total_reward: float
    terminated: bool
    truncated: bool

    def diff(self, prev_state: "TicTacToePrivateState") -> Dict[str, Any]:
        differences = {}
        if self.reward_last != prev_state.reward_last:
            differences["reward_last"] = self.reward_last
        if self.total_reward != prev_state.total_reward:
            differences["total_reward"] = self.total_reward
        if self.terminated != prev_state.terminated:
            differences["terminated"] = self.terminated
        if self.truncated != prev_state.truncated:
            differences["truncated"] = self.truncated
        return differences


@dataclass
class TicTacToeEngineSnapshot(StatefulEngineSnapshot):
    task_instance_dict: Dict
    engine_snapshot: Dict


class TicTacToeWinComponent(RewardComponent):
    def __init__(self, player_mark: str = "X"):
        super().__init__()
        self.player_mark = player_mark

    async def score(self, state: TicTacToePublicState, action: Any) -> float:
        if state.winner == self.player_mark:
            return 1.0
        elif state.winner and state.winner != "draw":
            return -1.0  # Opponent won
        return 0.0


class TicTacToeDrawComponent(RewardComponent):
    async def score(self, state: TicTacToePublicState, action: Any) -> float:
        if state.winner == "draw":
            return 0.0
        return 0.0


class TicTacToeIllegalMoveComponent(RewardComponent):
    def __init__(self):
        self.illegal_move_attempted = False

    async def score(self, state: TicTacToePublicState, action: Any) -> float:
        if self.illegal_move_attempted:
            self.illegal_move_attempted = False
            return -1.0
        return 0.0


class TicTacToeEngine(StatefulEngine, IReproducibleEngine):
    def __init__(self, task_instance: TaskInstance):
        self.task_instance = task_instance
        self.illegal_move_component = TicTacToeIllegalMoveComponent()

        # Determine which player the agent is controlling
        agent_player = "X"  # Default to X
        if hasattr(task_instance, "metadata") and hasattr(
            task_instance.metadata, "starting_player"
        ):
            agent_player = task_instance.metadata.starting_player

        self.reward_stack = RewardStack(
            [
                TicTacToeWinComponent(player_mark=agent_player),
                TicTacToeDrawComponent(),
                self.illegal_move_component,
            ]
        )

        # Initialize game state
        self.board = np.zeros(9, dtype=int)
        self.current_player = "X"
        self.last_move = None
        self.winner = None
        self.move_count = 0
        self.terminated = False
        self.total_reward = 0.0

        # Apply any pre-moves from task instance metadata
        if hasattr(task_instance, "metadata") and hasattr(task_instance.metadata, "opening_moves"):
            for move in task_instance.metadata.opening_moves:
                self._apply_move(move)

    async def _reset_engine(
        self, *, seed: int | None = None
    ) -> Tuple[TicTacToePrivateState, TicTacToePublicState]:
        self.board = np.zeros(9, dtype=int)
        self.current_player = "X"
        self.last_move = None
        self.winner = None
        self.move_count = 0
        self.terminated = False
        self.total_reward = 0.0

        # Apply any pre-moves from task instance metadata
        if hasattr(self.task_instance, "metadata") and hasattr(
            self.task_instance.metadata, "opening_moves"
        ):
            for move in self.task_instance.metadata.opening_moves:
                self._apply_move(move)

        public_state = TicTacToePublicState(
            board=self.board.copy(),
            current_player=self.current_player,
            last_move=self.last_move,
            winner=self.winner,
            move_count=self.move_count,
            max_moves=9,
            terminated=self.terminated,
        )

        private_state = TicTacToePrivateState(
            reward_last=0.0,
            total_reward=self.total_reward,
            terminated=self.terminated,
            truncated=False,
        )

        return private_state, public_state

    async def _step_engine(self, action: str) -> Tuple[TicTacToePrivateState, TicTacToePublicState]:
        # Validate and apply move
        if not self._is_valid_move(action, self.board):
            self.illegal_move_component.illegal_move_attempted = True
            self.terminated = True
        else:
            self._apply_move(action)

        # Create public state
        public_state = TicTacToePublicState(
            board=self.board.copy(),
            current_player=self.current_player,
            last_move=self.last_move,
            winner=self.winner,
            move_count=self.move_count,
            max_moves=9,
            terminated=self.terminated,
        )

        # Calculate rewards
        reward = await self.reward_stack.step_reward(public_state, action)
        self.total_reward += reward

        # Create private state
        private_state = TicTacToePrivateState(
            reward_last=reward,
            total_reward=self.total_reward,
            terminated=self.terminated,
            truncated=False,
        )

        return private_state, public_state

    def _apply_move(self, coord: str):
        if coord not in COORD_TO_IDX:
            return

        idx = COORD_TO_IDX[coord]
        if self.board[idx] == 0:
            self.board[idx] = PLAYER_MARKS[self.current_player]
            self.last_move = coord
            self.move_count += 1

            # Check for winner
            self.winner = self._check_winner(self.board)

            # Check if game is over
            if self.winner is not None or self.move_count >= 9:
                self.terminated = True
            else:
                # Switch players
                self.current_player = "O" if self.current_player == "X" else "X"

    def _check_winner(self, board: np.ndarray) -> Optional[str]:
        # Check all win patterns
        for pattern in WIN_PATTERNS:
            values = [board[i] for i in pattern]
            if values[0] != 0 and values[0] == values[1] == values[2]:
                return MARK_TO_PLAYER[values[0]]

        # Check for draw
        if np.all(board != 0):
            return "draw"

        return None

    def _is_valid_move(self, coord: str, board: np.ndarray) -> bool:
        if coord not in COORD_TO_IDX:
            return False
        idx = COORD_TO_IDX[coord]
        return board[idx] == 0

    async def _serialize_engine(self) -> TicTacToeEngineSnapshot:
        return TicTacToeEngineSnapshot(
            task_instance_dict=await self.task_instance.serialize(),
            engine_snapshot={
                "board": self.board.tolist(),
                "current_player": self.current_player,
                "last_move": self.last_move,
                "winner": self.winner,
                "move_count": self.move_count,
                "terminated": self.terminated,
                "total_reward": self.total_reward,
            },
        )

    @classmethod
    async def _deserialize_engine(cls, snapshot: TicTacToeEngineSnapshot) -> "TicTacToeEngine":
        task_instance = await TaskInstance.deserialize(snapshot.task_instance_dict)
        engine = cls(task_instance)

        # Restore state
        engine.board = np.array(snapshot.engine_snapshot["board"])
        engine.current_player = snapshot.engine_snapshot["current_player"]
        engine.last_move = snapshot.engine_snapshot["last_move"]
        engine.winner = snapshot.engine_snapshot["winner"]
        engine.move_count = snapshot.engine_snapshot["move_count"]
        engine.terminated = snapshot.engine_snapshot["terminated"]
        engine.total_reward = snapshot.engine_snapshot["total_reward"]

        return engine

    def get_current_states_for_observation(
        self,
    ) -> Tuple[TicTacToePrivateState, TicTacToePublicState]:
        public_state = TicTacToePublicState(
            board=self.board.copy(),
            current_player=self.current_player,
            last_move=self.last_move,
            winner=self.winner,
            move_count=self.move_count,
            max_moves=9,
            terminated=self.terminated,
        )

        private_state = TicTacToePrivateState(
            reward_last=0.0,
            total_reward=self.total_reward,
            terminated=self.terminated,
            truncated=False,
        )

        return private_state, public_state


class SynthTicTacToeObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: TicTacToePublicState, priv: TicTacToePrivateState
    ) -> InternalObservation:
        observation: InternalObservation = {
            "board_text": pub.board_text,
            "current_player": pub.current_player,
            "move_count": pub.move_count,
            "last_move": pub.last_move,
            "winner": pub.winner,
            "terminated": pub.terminated,
            "reward_last": priv.reward_last,
            "total_reward": priv.total_reward,
        }
        return observation


class SynthTicTacToeCheckpointObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: TicTacToePublicState, priv: TicTacToePrivateState
    ) -> InternalObservation:
        observation: InternalObservation = {
            "board_text_final": pub.board_text,
            "winner_final": pub.winner,
            "move_count_final": pub.move_count,
            "total_reward": priv.total_reward,
            "terminated": pub.terminated,
        }
        return observation
