from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from synth_ai.environments.environment.rewards.core import RewardComponent, RewardStack
from synth_ai.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_ai.environments.examples.sokoban.engine_helpers.vendored.envs.sokoban_env import (
    ACTION_LOOKUP,
)
from synth_ai.environments.examples.sokoban.engine_helpers.vendored.envs.sokoban_env import (
    SokobanEnv as GymSokobanEnv,
)
from synth_ai.environments.examples.sokoban.taskset import (
    SokobanTaskInstance,
)  # Assuming this is where SokobanTaskInstance is defined
from synth_ai.environments.reproducibility.core import IReproducibleEngine  # Added import
from synth_ai.environments.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from synth_ai.environments.tasks.core import TaskInstance

# No monkey-patch needed - we fixed the vendored code directly

# Configure logging for debugging SokobanEngine steps
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
# Suppress verbose PIL debug logs
logging.getLogger("PIL").setLevel(logging.WARNING)

# --- Action Mapping ---
ACTION_STRING_TO_INT: Dict[str, int] = {
    "no operation": 0,
    "push up": 1,
    "push down": 2,
    "push left": 3,
    "push right": 4,
    "move up": 5,
    "move down": 6,
    "move left": 7,
    "move right": 8,
}
INT_TO_ACTION_STRING: Dict[int, str] = {v: k for k, v in ACTION_STRING_TO_INT.items()}


@dataclass
class SokobanEngineSnapshot(StatefulEngineSnapshot):
    task_instance_dict: Dict
    engine_snapshot: Dict


@dataclass
class SokobanPublicState:
    dim_room: Tuple[int, int]
    room_fixed: np.ndarray  # numpy kinda sucks
    room_state: np.ndarray
    player_position: Tuple[int, int]
    boxes_on_target: int
    num_steps: int
    max_steps: int
    last_action_name: str
    num_boxes: int
    error_info: Optional[str] = None

    def diff(self, prev_state: "SokobanPublicState") -> Dict[str, Any]:
        changes: Dict[str, Any] = {}
        for field in self.__dataclass_fields__:  # type: ignore[attr-defined]
            new_v, old_v = getattr(self, field), getattr(prev_state, field)
            if isinstance(new_v, np.ndarray):
                if not np.array_equal(new_v, old_v):
                    changes[field] = True
            elif new_v != old_v:
                changes[field] = (old_v, new_v)
        return changes

    @property
    def room_text(self) -> str:
        """ASCII visualization of the room state"""
        return _grid_to_text(self.room_state)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper numpy array serialization."""
        return {
            "dim_room": self.dim_room,
            "room_fixed": self.room_fixed.tolist(),  # Convert numpy array to list
            "room_state": self.room_state.tolist(),  # Convert numpy array to list
            "player_position": self.player_position,
            "boxes_on_target": self.boxes_on_target,
            "num_steps": self.num_steps,
            "max_steps": self.max_steps,
            "last_action_name": self.last_action_name,
            "num_boxes": self.num_boxes,
            "error_info": self.error_info,
        }

    def __repr__(self) -> str:
        """Safe string representation that avoids numpy array recursion."""
        return f"SokobanPublicState(dim_room={self.dim_room}, num_steps={self.num_steps}, boxes_on_target={self.boxes_on_target})"

    def __str__(self) -> str:
        """Safe string representation that avoids numpy array recursion."""
        return self.__repr__()


@dataclass
class SokobanPrivateState:
    reward_last: float
    total_reward: float
    terminated: bool
    truncated: bool
    rng_state: dict | None = None

    def diff(self, prev_state: "SokobanPrivateState") -> Dict[str, Any]:
        changes: Dict[str, Any] = {}
        for field in self.__dataclass_fields__:  # type: ignore[attr-defined]
            new_v, old_v = getattr(self, field), getattr(prev_state, field)
            if new_v != old_v:
                changes[field] = (old_v, new_v)
        return changes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        return {
            "reward_last": self.reward_last,
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "rng_state": self.rng_state,
        }

    def __repr__(self) -> str:
        """Safe string representation."""
        return f"SokobanPrivateState(reward_last={self.reward_last}, total_reward={self.total_reward}, terminated={self.terminated})"

    def __str__(self) -> str:
        """Safe string representation."""
        return self.__repr__()


# Note - just how we roll! Show your agent whatever state you want
# Close to original
def _grid_to_text(grid: np.ndarray) -> str:
    """Pretty 3-char glyphs for each cell – same lookup the legacy renderer used."""
    return "\n".join(
        "".join(GRID_LOOKUP.get(int(cell), "?") for cell in row)  # type: ignore[arg-type]
        for row in grid
    )


class SynthSokobanObservationCallable(GetObservationCallable):
    def __init__(self):
        pass

    async def get_observation(
        self, pub: SokobanPublicState, priv: SokobanPrivateState
    ) -> InternalObservation:  # type: ignore[override]
        board_txt = _grid_to_text(pub.room_state)
        return {
            "room_text": board_txt,
            "player_position": tuple(map(int, pub.player_position)),
            "boxes_on_target": int(pub.boxes_on_target),
            "steps_taken": int(pub.num_steps),
            "max_steps": int(pub.max_steps),
            "last_action": pub.last_action_name,
            "reward_last": float(priv.reward_last),
            "total_reward": float(priv.total_reward),
            "terminated": bool(priv.terminated),
            "truncated": bool(priv.truncated),
            "num_boxes": int(pub.num_boxes),
        }


# Close to original
class SynthSokobanCheckpointObservationCallable(GetObservationCallable):
    """
    Snapshot emitted once after the episode finishes.
    Mirrors the legacy 'final_observation' concept: full board + final tallies.
    """

    def __init__(self):
        pass

    async def get_observation(
        self, pub: SokobanPublicState, priv: SokobanPrivateState
    ) -> InternalObservation:  # type: ignore[override]
        board_txt = _grid_to_text(pub.room_state)
        return {
            "room_text_final": board_txt,
            "boxes_on_target_final": int(pub.boxes_on_target),
            "steps_taken_final": int(pub.num_steps),
            "total_reward": float(priv.total_reward),
            "terminated": bool(priv.terminated),
            "truncated": bool(priv.truncated),
            "num_boxes": int(pub.num_boxes),
        }


# Think of engine as the actual logic, then with hooks to update the public and private state
# Note - I don't really want to split up the transformation/engine logic from the instance information. There's already quite a bit of abstraction, so let's make the hard call here. I observe that this class does combine the responsibility of tracking engine state AND containing dynamics, but I think it's fine.


GRID_LOOKUP = {0: " # ", 1: " _ ", 2: " O ", 3: " √ ", 4: " X ", 5: " P ", 6: " S "}


def _count_boxes_on_target(room_state: np.ndarray) -> int:
    """Return number of boxes currently sitting on target tiles."""
    return int(np.sum(room_state == 3))


def package_sokoban_env_from_engine_snapshot(
    engine_snapshot: Dict[str, Any],
) -> GymSokobanEnv:
    """Instantiate SokobanEnv and load every field from a saved-state dict."""
    # 1. create empty env (skip reset)
    env = GymSokobanEnv(
        dim_room=tuple(engine_snapshot["dim_room"]),
        max_steps=engine_snapshot.get("max_steps", 120),
        num_boxes=engine_snapshot.get("num_boxes", 1),
        reset=False,
    )

    # 2. restore core grids
    env.room_fixed = np.asarray(engine_snapshot["room_fixed"], dtype=int)
    env.room_state = np.asarray(engine_snapshot["room_state"], dtype=int)

    # 3. restore auxiliary data
    raw_map = engine_snapshot.get("box_mapping", {})
    if isinstance(raw_map, list):  # list-of-dict form
        env.box_mapping = {tuple(e["original"]): tuple(e["current"]) for e in raw_map}
    else:  # string-keyed dict form
        env.box_mapping = {
            tuple(map(int, k.strip("[]").split(","))): tuple(v) for k, v in raw_map.items()
        }

    env.player_position = np.argwhere(env.room_state == 5)[0]
    env.num_env_steps = engine_snapshot.get("num_env_steps", 0)
    env.boxes_on_target = engine_snapshot.get("boxes_on_target", int(np.sum(env.room_state == 3)))
    env.reward_last = engine_snapshot.get("reward_last", 0)

    # 4. restore RNG (if provided)
    rng = engine_snapshot.get("np_random_state")
    if rng:
        env.seed()  # init env.np_random
        env.np_random.set_state(
            (
                rng["key"],
                np.array(rng["state"], dtype=np.uint32),
                rng["pos"],
                0,  # has_gauss
                0.0,  # cached_gaussian
            )
        )

    return env


# --- Reward Components ---
class SokobanGoalAchievedComponent(RewardComponent):
    async def score(self, state: "SokobanPublicState", action: Any) -> float:
        if state.boxes_on_target == state.num_boxes:
            return 1.0
        return 0.0


class SokobanStepPenaltyComponent(RewardComponent):
    def __init__(self, penalty: float = -0.01):
        super().__init__()
        self.penalty = penalty
        self.weight = 1.0

    async def score(self, state: Any, action: Any) -> float:
        return self.penalty


class SokobanEngine(StatefulEngine, IReproducibleEngine):
    task_instance: TaskInstance
    package_sokoban_env: GymSokobanEnv

    # sokoban stuff

    def __init__(self, task_instance: TaskInstance):
        self.task_instance = task_instance
        self._total_reward = 0.0  # Initialize total_reward
        self._current_action_for_reward: Optional[int] = None
        self.reward_stack = RewardStack(
            components=[
                SokobanGoalAchievedComponent(),
                SokobanStepPenaltyComponent(penalty=-0.01),
            ]
        )

        init_snap: dict | None = getattr(self.task_instance, "initial_engine_snapshot", None)

        if init_snap:
            # Initialize package_sokoban_env here using the snapshot
            self.package_sokoban_env = package_sokoban_env_from_engine_snapshot(init_snap)
            # Ensure counters are consistent with the snapshot state
            self.package_sokoban_env.boxes_on_target = _count_boxes_on_target(
                self.package_sokoban_env.room_state
            )
        else:
            # No initial snapshot - this should not happen with the new pre-generated puzzle system
            # Create a minimal default environment as fallback
            logger.warning(
                "No initial_engine_snapshot provided - this should not happen with verified puzzles"
            )
            self.package_sokoban_env = GymSokobanEnv(
                dim_room=(5, 5),
                max_steps=50,
                num_boxes=1,
                reset=False,  # Don't reset during creation to avoid generation
            )

    # gives the observation!
    # also final rewards when those are passed in
    async def _render(
        self,
        private_state: SokobanPrivateState,
        public_state: SokobanPublicState,
        get_observation: Optional[GetObservationCallable] = None,
    ) -> str:
        """
        1. choose the observation callable (default = SynthSokobanObservationCallable)
        2. fetch obs via callable(pub, priv)
        3. if callable returned a dict -> pretty-print board + footer
           if str -> forward unchanged
        """
        # 1 – pick callable
        obs_cb = get_observation or SynthSokobanObservationCallable()

        # 2 – pull observation
        obs = await obs_cb.get_observation(public_state, private_state)

        # 3 – stringify
        if isinstance(obs, str):
            return obs

        if isinstance(obs, dict):
            board_txt = (
                obs.get("room_text")
                or obs.get("room_text_final")
                or _grid_to_text(public_state.room_state)
            )
            footer = (
                f"steps: {public_state.num_steps}/{public_state.max_steps} | "
                f"boxes✓: {public_state.boxes_on_target} | "
                f"last_r: {private_state.reward_last:.2f} | "
                f"total_r: {private_state.total_reward:.2f}"
            )
            return f"{board_txt}\n{footer}"

        # unknown payload type -> fallback
        return str(obs)

    # yields private state, public state
    async def _step_engine(self, action: int) -> Tuple[SokobanPrivateState, SokobanPublicState]:
        self._current_action_for_reward = action  # Set context for reward

        # --- Run underlying package environment step ---
        # The raw reward from package_sokoban_env.step() will be ignored,
        # as we are now using our RewardStack for a more structured reward calculation.
        obs_raw, _, terminated_gym, info = self.package_sokoban_env.step(action)

        self.package_sokoban_env.boxes_on_target = _count_boxes_on_target(
            self.package_sokoban_env.room_state
        )
        current_pub_state = SokobanPublicState(
            dim_room=self.package_sokoban_env.dim_room,
            room_fixed=self.package_sokoban_env.room_fixed.copy(),
            room_state=self.package_sokoban_env.room_state.copy(),
            player_position=tuple(self.package_sokoban_env.player_position),
            boxes_on_target=self.package_sokoban_env.boxes_on_target,
            num_steps=self.package_sokoban_env.num_env_steps,
            max_steps=self.package_sokoban_env.max_steps,
            last_action_name=ACTION_LOOKUP.get(action, "Unknown"),
            num_boxes=self.package_sokoban_env.num_boxes,
        )

        # --- Calculate reward using RewardStack ---
        # The 'state' for reward components is current_pub_state.
        # The 'action' for reward components is the raw agent action.
        reward_from_stack = await self.reward_stack.step_reward(
            state=current_pub_state, action=self._current_action_for_reward
        )
        self._current_action_for_reward = None  # Reset context

        self._total_reward += reward_from_stack
        # Update reward_last on the package_sokoban_env if it's used by its internal logic or for direct inspection.
        # However, the authoritative reward for our framework is reward_from_stack.
        self.package_sokoban_env.reward_last = reward_from_stack

        # --- Determine terminated and truncated status based on gym env and game logic ---
        solved = self.package_sokoban_env.boxes_on_target == self.package_sokoban_env.num_boxes
        terminated = terminated_gym or solved  # terminated_gym from underlying env, or solved state
        # If underlying env says terminated due to max_steps, it is truncation for us.
        # If solved, it's termination. Otherwise, depends on max_steps.
        truncated = (
            self.package_sokoban_env.num_env_steps >= self.package_sokoban_env.max_steps
        ) and not solved
        if solved:
            terminated = True  # Ensure solved always terminates
            truncated = False  # Cannot be truncated if solved

        priv = SokobanPrivateState(
            reward_last=reward_from_stack,
            total_reward=self._total_reward,
            terminated=terminated,
            truncated=truncated,
        )
        return priv, current_pub_state

    async def _reset_engine(
        self, *, seed: int | None = None
    ) -> Tuple[SokobanPrivateState, SokobanPublicState]:
        """
        (Re)build the wrapped PackageSokobanEnv in a fresh state.

        1.  Decide whether we have an initial snapshot in the TaskInstance.
        2.  If yes → hydrate env from it; otherwise call env.reset(seed).
        3.  Zero-out cumulative reward and emit fresh state objects.
        """
        self._total_reward = 0.0
        self._current_action_for_reward = None

        init_snap: dict | None = getattr(self.task_instance, "initial_engine_snapshot")

        if init_snap:
            self.package_sokoban_env = package_sokoban_env_from_engine_snapshot(init_snap)
            # ensure counter correct even if snapshot was stale
            self.package_sokoban_env.boxes_on_target = _count_boxes_on_target(
                self.package_sokoban_env.room_state
            )
        else:
            # No initial snapshot - this should not happen with the new pre-generated puzzle system
            logger.warning(
                "No initial_engine_snapshot provided during reset - this should not happen with verified puzzles"
            )
            # Simple fallback: try to reset the existing environment
            try:
                _ = self.package_sokoban_env.reset(seed=seed)
                # Update the boxes_on_target counter
                self.package_sokoban_env.boxes_on_target = _count_boxes_on_target(
                    self.package_sokoban_env.room_state
                )
            except Exception as e:
                logger.error(f"Failed to reset environment: {e}")
                raise RuntimeError(
                    "Environment reset failed. This should not happen with verified puzzles. "
                    "Ensure task instances have initial_engine_snapshot."
                ) from e

        # build first public/private views
        priv = SokobanPrivateState(
            reward_last=0.0,
            total_reward=0.0,
            terminated=False,
            truncated=False,
            rng_state=self.package_sokoban_env.np_random.bit_generator.state,
        )
        pub = SokobanPublicState(
            dim_room=self.package_sokoban_env.dim_room,
            room_fixed=self.package_sokoban_env.room_fixed.copy(),
            room_state=self.package_sokoban_env.room_state.copy(),
            player_position=tuple(self.package_sokoban_env.player_position),
            boxes_on_target=self.package_sokoban_env.boxes_on_target,
            num_steps=self.package_sokoban_env.num_env_steps,
            max_steps=self.package_sokoban_env.max_steps,
            last_action_name="Initial",
            num_boxes=self.package_sokoban_env.num_boxes,
        )
        return priv, pub

    async def _serialize_engine(self) -> SokobanEngineSnapshot:
        """Dump wrapped env + task_instance into a JSON-ready snapshot."""
        env = self.package_sokoban_env

        # helper – numpy RNG → dict
        def _rng_state(e):
            state = e.np_random.bit_generator.state
            state["state"] = state["state"].tolist()
            return state

        snap: Dict[str, Any] = {
            "dim_room": list(env.dim_room),
            "max_steps": env.max_steps,
            "num_boxes": env.num_boxes,
            "room_fixed": env.room_fixed.tolist(),
            "room_state": env.room_state.tolist(),
            "box_mapping": [
                {"original": list(k), "current": list(v)} for k, v in env.box_mapping.items()
            ],
            "player_position": env.player_position.tolist(),
            "num_env_steps": env.num_env_steps,
            "boxes_on_target": env.boxes_on_target,
            "reward_last": env.reward_last,
            "total_reward": getattr(self, "_total_reward", 0.0),
            # "np_random_state": _rng_state(env), # Assuming _rng_state is defined if needed
        }

        # Serialize the TaskInstance using its own serialize method
        task_instance_dict = await self.task_instance.serialize()

        return SokobanEngineSnapshot(
            task_instance_dict=task_instance_dict,  # Store serialized TaskInstance
            engine_snapshot=snap,
        )

    @classmethod
    async def _deserialize_engine(
        cls, sokoban_engine_snapshot: "SokobanEngineSnapshot"
    ) -> "SokobanEngine":
        """
        Recreate a SokobanEngine (including wrapped env and TaskInstance) from a snapshot blob.
        """
        # --- 1. rebuild TaskInstance ----------------------------------- #
        # Use the concrete SokobanTaskInstance.deserialize method
        instance = await SokobanTaskInstance.deserialize(sokoban_engine_snapshot.task_instance_dict)

        # --- 2. create engine shell ------------------------------------ #
        engine = cls.__new__(cls)  # bypass __init__
        StatefulEngine.__init__(engine)  # initialise mix-in parts
        engine.task_instance = instance  # assign restored TaskInstance

        # --- 3. initialize attributes that are normally set in __init__ --- #
        engine._current_action_for_reward = None
        engine.reward_stack = RewardStack(
            components=[
                SokobanGoalAchievedComponent(),
                SokobanStepPenaltyComponent(penalty=-0.01),
            ]
        )

        # --- 4. hydrate env & counters --------------------------------- #
        engine.package_sokoban_env = package_sokoban_env_from_engine_snapshot(
            sokoban_engine_snapshot.engine_snapshot
        )
        engine._total_reward = sokoban_engine_snapshot.engine_snapshot.get("total_reward", 0.0)
        return engine

    def get_current_states_for_observation(
        self,
    ) -> Tuple[SokobanPrivateState, SokobanPublicState]:
        # Helper to get current state without advancing engine, useful for error in Environment.step
        terminated = bool(
            self.package_sokoban_env.boxes_on_target == self.package_sokoban_env.num_boxes
        )
        truncated = bool(
            self.package_sokoban_env.num_env_steps >= self.package_sokoban_env.max_steps
        )
        priv = SokobanPrivateState(
            reward_last=self.package_sokoban_env.reward_last,  # Last known reward
            total_reward=self._total_reward,
            terminated=terminated,
            truncated=truncated,
        )
        pub = SokobanPublicState(
            dim_room=self.package_sokoban_env.dim_room,
            room_fixed=self.package_sokoban_env.room_fixed.copy(),
            room_state=self.package_sokoban_env.room_state.copy(),
            player_position=tuple(self.package_sokoban_env.player_position),
            boxes_on_target=self.package_sokoban_env.boxes_on_target,
            num_steps=self.package_sokoban_env.num_env_steps,
            max_steps=self.package_sokoban_env.max_steps,
            last_action_name=ACTION_LOOKUP.get(
                getattr(self.package_sokoban_env, "last_action", -1), "Initial"
            ),
            num_boxes=self.package_sokoban_env.num_boxes,
        )
        return priv, pub


if __name__ == "__main__":
    # // 0=wall, 1=floor, 2=target
    # // 4=box-not-on-target, 5=player
    # initial_room = {
    #     "dim_room": [5, 5],
    #     "max_steps": 120,
    #     "num_boxes": 1,
    #     "seed": 42,
    #     "room_fixed": [
    #         [0, 0, 0, 0, 0],
    #         [0, 1, 1, 2, 0],
    #         [0, 1, 0, 1, 0],
    #         [0, 1, 5, 1, 0],
    #         [0, 0, 0, 0, 0]
    #     ],
    #     "room_state": [
    #         [0, 0, 0, 0, 0],
    #         [0, 1, 4, 1, 0],
    #         [0, 1, 0, 1, 0],
    #         [0, 1, 5, 1, 0],
    #         [0, 0, 0, 0, 0]
    #     ]
    # }
    task_instance_dict = {
        "initial_engine_snapshot": {
            "dim_room": [5, 5],
            "max_steps": 120,
            "num_boxes": 1,
            "room_fixed": [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 2, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            "room_state": [
                [0, 0, 0, 0, 0],
                [0, 1, 4, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 5, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            "box_mapping": [{"original": [1, 2], "current": [3, 2]}],
            "boxes_on_target": 0,
            "np_random_state": {
                "key": "MT19937",
                "state": [1804289383, 846930886, 1681692777, 1714636915],
                "pos": 0,
            },
            "reward_last": 0,
            "num_env_steps": 0,
        }
    }
    import asyncio
    import random

    async def sanity():
        task_instance = TaskInstance()
        engine = SokobanEngine(task_instance=task_instance)
        priv, pub = await engine._reset_engine()
        print(await engine._render(priv, pub))  # initial board

        for _ in range(10):  # play 10 random moves
            a = random.randint(0, 8)  # action range 0-8
            priv, pub = await engine._step_engine(a)
            print(f"\n### step {pub.num_steps} — {ACTION_LOOKUP[a]} ###")
            print("public:", pub)
            print("private:", priv)
            print(await engine._render(priv, pub))
            if priv.terminated or priv.truncated:
                break

    asyncio.run(sanity())
    # sokoban_engine = SokobanEngine.deserialize(
    #     engine_snapshot=SokobanEngineSnapshot(
    #         instance=instance_information,
    #         snapshot_dict=instance_information["initial_engine_snapshot"],
    #     )
    # )


# {
#   "dim_room": [5, 5],
#   "max_steps": 120,
#   "num_boxes": 1,

#   "room_fixed": [...],                // as above
#   "room_state": [...],                // current grid (3 = box-on-target)

#   "box_mapping": {
#     "[1,3]": [3,2]                    // origin-target → current-pos pairs
#   },
#   "player_position": [3, 2],          // row, col

#   "num_env_steps": 15,                // steps already taken
#   "boxes_on_target": 0,               // live counter

#   "np_random_state": {                // optional but makes replay bit-exact
#     "key": "MT19937",
#     "state": [1804289383, 846930886, ...],
#     "pos": 123
#   }
# }
