"""MiniGrid Engine implementation.

This module provides a wrapper around Gymnasium MiniGrid environments
with full state management and serialization capabilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, List

import gymnasium as gym
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX

from synth_ai.environments.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from synth_ai.environments.reproducibility.core import IReproducibleEngine
from synth_ai.environments.environment.rewards.core import RewardComponent, RewardStack
from synth_ai.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_ai.environments.tasks.core import TaskInstance
from synth_ai.environments.examples.minigrid.environment_mapping import (
    get_environment_from_seed,
    get_difficulty_from_seed,
    validate_environment_name,
)


@dataclass
class MiniGridPublicState:
    """Public state of the MiniGrid environment."""

    grid_array: np.ndarray  # The grid as a numpy array
    agent_pos: Tuple[int, int]  # Agent position (x, y)
    agent_dir: int  # Agent direction (0=right, 1=down, 2=left, 3=up)
    carrying: Optional[Dict[str, Any]] = None  # Object being carried
    step_count: int = 0
    max_steps: int = 1000
    mission: str = ""
    terminated: bool = False

    def diff(self, prev_state: "MiniGridPublicState") -> Dict[str, Any]:
        """Track changes between states."""
        differences = {}
        if not np.array_equal(self.grid_array, prev_state.grid_array):
            differences["grid_array"] = self.grid_array.tolist()
        if self.agent_pos != prev_state.agent_pos:
            differences["agent_pos"] = self.agent_pos
        if self.agent_dir != prev_state.agent_dir:
            differences["agent_dir"] = self.agent_dir
        if self.carrying != prev_state.carrying:
            differences["carrying"] = self.carrying
        if self.step_count != prev_state.step_count:
            differences["step_count"] = self.step_count
        if self.mission != prev_state.mission:
            differences["mission"] = self.mission
        if self.terminated != prev_state.terminated:
            differences["terminated"] = self.terminated
        return differences


@dataclass
class MiniGridPrivateState:
    """Private state of the MiniGrid environment."""

    reward_last: float = 0.0
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    # Debug information
    last_action: Optional[str] = None
    last_action_result: Optional[str] = (
        None  # "success", "blocked_by_wall", "blocked_by_boundary", etc.
    )
    position_before_action: Optional[Tuple[int, int]] = None
    position_after_action: Optional[Tuple[int, int]] = None
    debug_message: Optional[str] = None

    def diff(self, prev_state: "MiniGridPrivateState") -> Dict[str, Any]:
        """Track changes between states."""
        differences = {}
        if self.reward_last != prev_state.reward_last:
            differences["reward_last"] = self.reward_last
        if self.total_reward != prev_state.total_reward:
            differences["total_reward"] = self.total_reward
        if self.terminated != prev_state.terminated:
            differences["terminated"] = self.terminated
        if self.truncated != prev_state.truncated:
            differences["truncated"] = self.truncated
        if self.info != prev_state.info:
            differences["info"] = self.info
        return differences


@dataclass
class MiniGridEngineSnapshot(StatefulEngineSnapshot):
    """Serialization container for MiniGrid engine."""

    task_instance_dict: Dict
    engine_snapshot: Dict


class MiniGridGoalReachedComponent(RewardComponent):
    """Reward component for reaching the goal."""

    def __init__(self, reward_value: float = 1.0):
        self.reward_value = reward_value

    async def score(self, state: MiniGridPublicState, action: Any) -> float:
        """Calculate reward based on whether goal was reached."""
        # Note: We check the private state info for success in the engine
        return 0.0  # Reward is handled by the base environment


class MiniGridStepPenaltyComponent(RewardComponent):
    """Penalty for each step taken."""

    def __init__(self, penalty: float = -0.01):
        self.penalty = penalty

    async def score(self, state: MiniGridPublicState, action: Any) -> float:
        """Apply small penalty for each step."""
        return self.penalty


class MiniGridObservationCallable(GetObservationCallable):
    """Default observation callable for MiniGrid."""

    async def get_observation(
        self, pub: MiniGridPublicState, priv: MiniGridPrivateState
    ) -> InternalObservation:
        """Generate text-based observation of the MiniGrid state."""
        # Create text representation of the grid
        grid_lines = []
        grid_array = pub.grid_array
        height, width = grid_array.shape[:2]

        # Object type mapping - use actual MiniGrid constants
        # Note: OBJECT_TO_IDX gives us the correct mapping
        # We need to create the reverse mapping: idx -> symbol

        # Direction symbols
        dir_symbols = ["→", "↓", "←", "↑"]

        # Build grid visualization
        for y in range(height):
            line = []
            for x in range(width):
                obj_type = grid_array[y, x, 0]

                if (x, y) == pub.agent_pos:
                    # Show agent with direction
                    line.append(dir_symbols[pub.agent_dir])
                elif obj_type == OBJECT_TO_IDX["empty"]:  # empty (1)
                    line.append(".")
                elif obj_type == OBJECT_TO_IDX["wall"]:  # wall (2)
                    line.append("#")
                elif obj_type == OBJECT_TO_IDX["goal"]:  # goal (8)
                    line.append("G")
                elif obj_type == OBJECT_TO_IDX["key"]:  # key (5)
                    line.append("K")
                elif obj_type == OBJECT_TO_IDX["door"]:  # door (4)
                    line.append("D")
                elif obj_type == OBJECT_TO_IDX["ball"]:  # ball (6)
                    line.append("B")
                elif obj_type == OBJECT_TO_IDX["lava"]:  # lava (9)
                    line.append("L")
                elif obj_type == OBJECT_TO_IDX["unseen"]:  # unseen (0)
                    line.append("?")
                else:
                    line.append("?")
            grid_lines.append(" ".join(line))

        # Build complete observation
        observation_parts = [
            f"Mission: {pub.mission}",
            f"Steps: {pub.step_count}/{pub.max_steps}",
            f"Agent Position: ({pub.agent_pos[0]}, {pub.agent_pos[1]})",
            f"Agent Direction: {dir_symbols[pub.agent_dir]}",
        ]

        if pub.carrying:
            observation_parts.append(f"Carrying: {pub.carrying['type']} ({pub.carrying['color']})")

        observation_parts.append("\nGrid:")
        observation_parts.extend(grid_lines)

        observation_parts.append(
            "\nLegend: # = wall, . = empty, G = goal, K = key, D = door, B = ball, L = lava"
        )
        observation_parts.append("Agent directions: → = right, ↓ = down, ← = left, ↑ = up")

        # Add debug information if available - make it more prominent
        if priv.debug_message or (priv.last_action_result and priv.last_action_result != "success"):
            observation_parts.append("\n" + "=" * 50)
            observation_parts.append("🚨 CRITICAL FEEDBACK FROM LAST ACTION:")
            if priv.debug_message:
                observation_parts.append(f"   {priv.debug_message}")
            if priv.last_action_result and priv.last_action_result != "success":
                observation_parts.append(f"   Result: {priv.last_action_result}")
            observation_parts.append(
                "   ⚠️  IMPORTANT: If blocked, you MUST turn or try different action!"
            )
            observation_parts.append("=" * 50)

        text_obs = "\n".join(observation_parts)

        observation: InternalObservation = {
            "observation": text_obs,
            "terminated": pub.terminated,
            "truncated": priv.truncated,
            "reward_last": priv.reward_last,
            "total_reward": priv.total_reward,
            # Include debug info in observation dict too
            "last_action": priv.last_action,
            "last_action_result": priv.last_action_result,
            "debug_message": priv.debug_message,
        }

        return observation


class MiniGridCheckpointObservationCallable(GetObservationCallable):
    """Checkpoint observation callable for MiniGrid."""

    async def get_observation(
        self, pub: MiniGridPublicState, priv: MiniGridPrivateState
    ) -> InternalObservation:
        """Generate checkpoint observation."""
        observation: InternalObservation = {
            "mission": pub.mission,
            "final_position": pub.agent_pos,
            "total_steps": pub.step_count,
            "total_reward": priv.total_reward,
            "terminated": pub.terminated,
            "truncated": priv.truncated,
            "success": priv.info.get("success", False),
        }
        return observation


class MiniGridEngine(StatefulEngine, IReproducibleEngine):
    """Engine for MiniGrid environments."""

    def __init__(
        self,
        task_instance: TaskInstance,
        render_mode: Optional[str] = None,
    ):
        """Initialize MiniGrid engine.

        Args:
            task_instance: Task instance containing configuration
            render_mode: Rendering mode for the environment
        """
        self.task_instance = task_instance
        self.render_mode = render_mode

        # Get environment configuration from task instance
        env_name = None
        seed = None
        difficulty = None

        # First try to get explicit configuration from metadata
        if hasattr(task_instance, "metadata"):
            if hasattr(task_instance.metadata, "env_name"):
                env_name = task_instance.metadata.env_name
            if hasattr(task_instance.metadata, "seed"):
                seed = task_instance.metadata.seed
            if hasattr(task_instance.metadata, "difficulty"):
                difficulty = task_instance.metadata.difficulty

        # If no explicit env_name but we have a seed, use seed mapping
        if env_name is None and seed is not None:
            env_name = get_environment_from_seed(seed)
            if difficulty is None:
                difficulty = get_difficulty_from_seed(seed)

        # If still no environment name, check if we can extract seed from config
        if env_name is None and hasattr(task_instance, "initial_engine_snapshot"):
            snapshot = task_instance.initial_engine_snapshot
            if snapshot and isinstance(snapshot, dict):
                config_seed = snapshot.get("seed")
                if config_seed is not None:
                    seed = config_seed
                    env_name = get_environment_from_seed(seed)
                    if difficulty is None:
                        difficulty = get_difficulty_from_seed(seed)

        # Final fallback to default environment
        if env_name is None:
            env_name = "MiniGrid-Empty-5x5-v0"
            seed = 0  # Ensure we have a seed for reproducibility

        # Validate the environment name
        if not validate_environment_name(env_name):
            print(f"Warning: Unknown environment '{env_name}', falling back to default")
            env_name = "MiniGrid-Empty-5x5-v0"
            seed = 0

        self.env_name = env_name
        self.seed = seed
        self.difficulty = difficulty

        # Create the environment
        self.env = gym.make(self.env_name, render_mode=self.render_mode)

        # Initialize reward stack
        self.reward_stack = RewardStack(
            [
                MiniGridStepPenaltyComponent(),
            ]
        )

        # Initialize state tracking
        self.total_reward = 0.0
        self._initialized = False

    def _grid_to_array(self) -> np.ndarray:
        """Convert MiniGrid grid to numpy array."""
        # Access the unwrapped environment
        unwrapped = self.env.unwrapped

        width, height = unwrapped.grid.width, unwrapped.grid.height
        grid_array = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                cell = unwrapped.grid.get(j, i)
                if cell is None:
                    grid_array[i, j] = [OBJECT_TO_IDX["empty"], 0, 0]
                else:
                    grid_array[i, j] = [
                        OBJECT_TO_IDX.get(cell.type, 0),
                        COLOR_TO_IDX.get(cell.color, 0),
                        STATE_TO_IDX.get(getattr(cell, "state", 0), 0)
                        if hasattr(cell, "state")
                        else 0,
                    ]

        # Add agent to grid
        if unwrapped.agent_pos is not None:
            ax, ay = unwrapped.agent_pos
            grid_array[ay, ax] = [
                OBJECT_TO_IDX["agent"],
                COLOR_TO_IDX["red"],
                unwrapped.agent_dir,
            ]

        return grid_array

    def _extract_public_state(self, terminated: bool = False) -> MiniGridPublicState:
        """Extract public state from the current environment."""
        # Access the unwrapped environment
        unwrapped = self.env.unwrapped

        # Get grid array representation
        grid_array = self._grid_to_array()

        # Get carrying object info
        carrying = None
        if unwrapped.carrying:
            carrying = {
                "type": unwrapped.carrying.type,
                "color": unwrapped.carrying.color,
            }

        return MiniGridPublicState(
            grid_array=grid_array,
            agent_pos=tuple(unwrapped.agent_pos),
            agent_dir=unwrapped.agent_dir,
            carrying=carrying,
            step_count=unwrapped.step_count,
            max_steps=unwrapped.max_steps,
            mission=unwrapped.mission,
            terminated=terminated,
        )

    async def _reset_engine(
        self, *, seed: int | None = None
    ) -> Tuple[MiniGridPrivateState, MiniGridPublicState]:
        """Reset to initial state."""
        # Reset environment
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        elif self.seed is not None:
            obs, info = self.env.reset(seed=self.seed)
        else:
            obs, info = self.env.reset()

        # Reset tracking
        self.total_reward = 0.0
        self._initialized = True

        # Create states
        public_state = self._extract_public_state(terminated=False)
        private_state = MiniGridPrivateState(
            reward_last=0.0,
            total_reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
        )

        return private_state, public_state

    async def _step_engine(self, action: int) -> Tuple[MiniGridPrivateState, MiniGridPublicState]:
        """Execute one step/action."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call _reset_engine first.")

        # Validate action
        if not isinstance(action, int) or action < 0 or action > 6:
            raise ValueError(f"Invalid action: {action}. Must be integer 0-6.")

        # Get position before action
        unwrapped = self.env.unwrapped
        pos_before = unwrapped.agent_pos
        dir_before = unwrapped.agent_dir

        # Map action to name
        action_names = {
            0: "left",
            1: "right",
            2: "forward",
            3: "pickup",
            4: "drop",
            5: "toggle",
            6: "done",
        }
        action_name = action_names.get(action, f"unknown({action})")

        # Execute action in environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Get position after action
        pos_after = unwrapped.agent_pos
        dir_after = unwrapped.agent_dir

        # Determine action result
        action_result = "success"
        debug_message = f"Action: {action_name}"

        if action in [0, 1]:  # Turn actions
            if dir_before != dir_after:
                action_result = "turned"
                debug_message = f"Turned {action_name}: direction {dir_before} -> {dir_after}"
            else:
                action_result = "turn_failed"
                debug_message = f"Turn {action_name} failed"
        elif action == 2:  # Forward action
            if pos_before == pos_after:
                # Check what blocked movement
                fwd_pos = unwrapped.front_pos
                if (
                    fwd_pos[0] < 0
                    or fwd_pos[0] >= unwrapped.width
                    or fwd_pos[1] < 0
                    or fwd_pos[1] >= unwrapped.height
                ):
                    action_result = "blocked_by_boundary"
                    debug_message = f"Forward blocked by grid boundary at {fwd_pos}"
                else:
                    cell = unwrapped.grid.get(*fwd_pos)
                    if cell is not None and cell.type == "wall":
                        action_result = "blocked_by_wall"
                        debug_message = f"Forward blocked by wall at {fwd_pos}"
                    elif cell is not None and cell.type == "lava":
                        action_result = "blocked_by_lava"
                        debug_message = f"Forward blocked by lava at {fwd_pos}"
                    else:
                        action_result = "blocked_by_object"
                        debug_message = (
                            f"Forward blocked by {cell.type if cell else 'unknown'} at {fwd_pos}"
                        )
            else:
                action_result = "moved"
                debug_message = f"Moved forward: {pos_before} -> {pos_after}"

        # Calculate custom rewards
        public_state = self._extract_public_state(terminated=terminated)
        custom_reward = await self.reward_stack.step_reward(public_state, action)

        # Use environment reward as base, add custom rewards
        total_step_reward = reward + custom_reward
        self.total_reward += total_step_reward

        # Create states with debug info
        private_state = MiniGridPrivateState(
            reward_last=total_step_reward,
            total_reward=self.total_reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            last_action=action_name,
            last_action_result=action_result,
            position_before_action=pos_before,
            position_after_action=pos_after,
            debug_message=debug_message,
        )

        return private_state, public_state

    async def _serialize_engine(self) -> MiniGridEngineSnapshot:
        """Serialize current state."""
        engine_snapshot = {
            "env_name": self.env_name,
            "seed": self.seed,
            "total_reward": self.total_reward,
            "initialized": self._initialized,
            # Note: Full environment state serialization would require
            # MiniGrid to support it, which it doesn't by default
        }

        task_dict = {}
        if hasattr(self.task_instance, "serialize"):
            task_dict = await self.task_instance.serialize()

        return MiniGridEngineSnapshot(
            task_instance_dict=task_dict,
            engine_snapshot=engine_snapshot,
        )

    @classmethod
    async def _deserialize_engine(cls, snapshot: MiniGridEngineSnapshot) -> "MiniGridEngine":
        """Restore from serialized state."""
        # Recreate task instance
        task_instance = None
        if snapshot.task_instance_dict:
            # This would need proper task instance deserialization
            task_instance = TaskInstance(**snapshot.task_instance_dict)

        # Create engine
        engine = cls(task_instance)

        # Restore state
        engine_data = snapshot.engine_snapshot
        engine.total_reward = engine_data.get("total_reward", 0.0)
        engine._initialized = engine_data.get("initialized", False)

        return engine

    def get_current_states_for_observation(
        self,
    ) -> Tuple[MiniGridPrivateState, MiniGridPublicState]:
        """Get current states without advancing."""
        if not self._initialized:
            # Return empty states
            return (
                MiniGridPrivateState(),
                MiniGridPublicState(
                    grid_array=np.zeros((5, 5, 3)),
                    agent_pos=(0, 0),
                    agent_dir=0,
                ),
            )

        # Access the unwrapped environment
        unwrapped = self.env.unwrapped

        # Get current state
        terminated = unwrapped.step_count >= unwrapped.max_steps
        public_state = self._extract_public_state(terminated=terminated)

        private_state = MiniGridPrivateState(
            reward_last=0.0,
            total_reward=self.total_reward,
            terminated=terminated,
            truncated=terminated,
            info={},
        )

        return private_state, public_state

    def get_available_actions(self) -> List[Tuple[int, str]]:
        """Get list of available actions with descriptions."""
        return [
            (0, "turn left"),  # Action 0 is counter-clockwise (left)
            (1, "turn right"),  # Action 1 is clockwise (right)
            (2, "move forward"),
            (3, "pickup"),
            (4, "drop"),
            (5, "toggle/activate"),
            (6, "done (complete mission)"),
        ]
