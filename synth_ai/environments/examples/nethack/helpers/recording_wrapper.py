"""Environment wrapper that adds trajectory recording capabilities."""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from src.synth_env.examples.nethack.environment import NetHackEnvironment
from src.synth_env.examples.nethack.helpers.trajectory_recorder import (
    TrajectoryRecorder,
)
from src.synth_env.environment.tools import EnvToolCall


logger = logging.getLogger(__name__)


class RecordingNetHackEnvironment(NetHackEnvironment):
    """NetHack environment with automatic trajectory recording."""

    def __init__(
        self,
        save_dir: str = "temp/nethack_trajectories",
        auto_record: bool = True,
        **kwargs,
    ):
        """Initialize recording wrapper.

        Args:
            save_dir: Directory to save trajectories
            auto_record: Whether to automatically record all episodes
            **kwargs: Arguments passed to NetHackEnvironment
        """
        super().__init__(**kwargs)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.auto_record = auto_record
        self.recorder = TrajectoryRecorder(save_dir)
        self.is_recording = False
        self.trajectory_id = None

    async def start(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Start environment and optionally begin recording."""
        public_state, private_state = await super().start(**kwargs)

        if self.auto_record:
            # Extract character role from task metadata
            character_role = "adventurer"  # default
            if hasattr(self.task_instance, "metadata") and hasattr(
                self.task_instance.metadata, "character_role"
            ):
                character_role = self.task_instance.metadata.character_role

            task_id = self.task_instance.task_id if self.task_instance else None
            self.trajectory_id = self.recorder.start_recording(character_role, task_id)
            self.is_recording = True

            logger.info(f"Started recording trajectory: {self.trajectory_id}")

            # Record initial state
            obs = self._extract_observation(public_state)
            self.recorder.record_step("reset", obs, 0.0, False, {})

        return public_state, private_state

    async def process_action(
        self, tool_calls: list[EnvToolCall]
    ) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Process action and record if enabled."""
        # Execute action
        observation, reward, terminated, info = await super().process_action(tool_calls)

        # Record step if recording
        if self.is_recording and self.recorder.is_recording:
            # Extract action from tool calls
            action = "unknown"
            if tool_calls and len(tool_calls) > 0:
                if hasattr(tool_calls[0], "args") and "action" in tool_calls[0].args:
                    action = tool_calls[0].args["action"]

            # Extract observation data
            obs_data = self._extract_observation(observation)

            # Record step
            self.recorder.record_step(action, obs_data, reward, terminated, info)

        # If episode ended, stop recording
        if terminated and self.is_recording:
            await self._finalize_recording(observation)

        return observation, reward, terminated, info

    def _extract_observation(self, state: Any) -> Dict[str, Any]:
        """Extract observation data from state object."""
        if hasattr(state, "__dict__"):
            # Convert state object to dict
            obs = {}

            # Extract relevant fields
            if hasattr(state, "ascii_map"):
                obs["ascii_map"] = state.ascii_map
            if hasattr(state, "message"):
                obs["message"] = state.message
            if hasattr(state, "character_stats"):
                obs["player_stats"] = state.character_stats
            if hasattr(state, "inventory"):
                obs["inventory"] = state.inventory
            if hasattr(state, "position"):
                obs["player_stats"] = obs.get("player_stats", {})
                obs["player_stats"]["x"] = state.position[0]
                obs["player_stats"]["y"] = state.position[1]
            if hasattr(state, "dungeon_level"):
                obs["player_stats"] = obs.get("player_stats", {})
                obs["player_stats"]["depth"] = state.dungeon_level
            if hasattr(state, "in_menu"):
                obs["in_menu"] = state.in_menu
            if hasattr(state, "menu_items"):
                obs["menu_items"] = state.menu_items

            return obs

        # If it's already a dict, return as is
        return state if isinstance(state, dict) else {}

    async def _finalize_recording(self, final_state: Any):
        """Finalize and save recording."""
        if not self.is_recording:
            return

        # Determine final status
        final_status = "completed"
        if hasattr(final_state, "message"):
            msg = final_state.message.lower()
            if "die" in msg or "killed" in msg:
                final_status = "died"
            elif "quit" in msg:
                final_status = "quit"
            elif "time limit" in msg or "truncated" in msg:
                final_status = "truncated"

        # Extract achievements if available
        achievements = {}
        # TODO: Extract achievements from game state

        # Stop recording
        self.recorder.stop_recording(final_status, achievements)

        # Save trajectory
        filepath = self.recorder.save_trajectory()
        logger.info(f"Saved trajectory to: {filepath}")

        # Get and log summary
        summary = self.recorder.get_summary()
        logger.info(f"Trajectory summary: {summary}")

        self.is_recording = False

    def start_recording(
        self, character_role: Optional[str] = None, task_id: Optional[str] = None
    ) -> str:
        """Manually start recording."""
        if self.is_recording:
            logger.warning("Recording already in progress")
            return self.trajectory_id

        if character_role is None:
            character_role = "adventurer"
            if hasattr(self, "engine") and hasattr(self.engine, "character_role"):
                character_role = self.engine.character_role

        self.trajectory_id = self.recorder.start_recording(character_role, task_id)
        self.is_recording = True
        logger.info(f"Started manual recording: {self.trajectory_id}")

        return self.trajectory_id

    def stop_recording(self, save: bool = True) -> Optional[str]:
        """Manually stop recording."""
        if not self.is_recording:
            logger.warning("No recording in progress")
            return None

        self.recorder.stop_recording()

        filepath = None
        if save:
            filepath = self.recorder.save_trajectory()
            logger.info(f"Saved trajectory to: {filepath}")

        self.is_recording = False
        return filepath

    def get_recording_status(self) -> Dict[str, Any]:
        """Get current recording status."""
        return {
            "is_recording": self.is_recording,
            "trajectory_id": self.trajectory_id,
            "current_step": self.recorder.current_step if self.is_recording else 0,
            "total_reward": self.recorder.total_reward if self.is_recording else 0.0,
        }
