"""Trajectory recording and replay functionality for NetHack."""

import json
import pickle
import gzip
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass, asdict
import base64


@dataclass
class TrajectoryFrame:
    """Single frame in a trajectory."""

    step: int
    action: str
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        d = asdict(self)
        # Handle numpy arrays in observation
        if "observation" in d and d["observation"]:
            d["observation"] = self._serialize_observation(d["observation"])
        return d

    def _serialize_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize observation, converting numpy arrays to lists."""
        serialized = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                serialized[key] = {
                    "type": "ndarray",
                    "data": value.tolist(),
                    "dtype": str(value.dtype),
                    "shape": value.shape,
                }
            elif isinstance(value, dict):
                serialized[key] = self._serialize_observation(value)
            elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], dict):
                serialized[key] = [
                    self._serialize_observation(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                serialized[key] = value
        return serialized

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryFrame":
        """Reconstruct from dict."""
        if "observation" in d and d["observation"]:
            d["observation"] = cls._deserialize_observation(d["observation"])
        return cls(**d)

    @staticmethod
    def _deserialize_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize observation, converting lists back to numpy arrays."""
        deserialized = {}
        for key, value in obs.items():
            if isinstance(value, dict) and value.get("type") == "ndarray":
                deserialized[key] = np.array(value["data"], dtype=value["dtype"]).reshape(
                    value["shape"]
                )
            elif isinstance(value, dict):
                deserialized[key] = TrajectoryFrame._deserialize_observation(value)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                deserialized[key] = [
                    TrajectoryFrame._deserialize_observation(item)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                deserialized[key] = value
        return deserialized


@dataclass
class TrajectoryMetadata:
    """Metadata for a trajectory."""

    trajectory_id: str
    character_role: str
    task_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    total_steps: int
    total_reward: float
    final_status: str  # 'completed', 'died', 'quit', 'truncated'
    max_depth_reached: int
    achievements: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        d = asdict(self)
        d["start_time"] = self.start_time.isoformat()
        if self.end_time:
            d["end_time"] = self.end_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryMetadata":
        """Reconstruct from dict."""
        d["start_time"] = datetime.fromisoformat(d["start_time"])
        if d.get("end_time"):
            d["end_time"] = datetime.fromisoformat(d["end_time"])
        return cls(**d)


class TrajectoryRecorder:
    """Records and saves NetHack game trajectories."""

    def __init__(self, save_dir: str = "temp/nethack_trajectories"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.frames: List[TrajectoryFrame] = []
        self.metadata: Optional[TrajectoryMetadata] = None
        self.is_recording = False
        self.current_step = 0
        self.total_reward = 0.0
        self.max_depth = 1

    def start_recording(self, character_role: str, task_id: Optional[str] = None) -> str:
        """Start recording a new trajectory."""
        trajectory_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + character_role

        self.metadata = TrajectoryMetadata(
            trajectory_id=trajectory_id,
            character_role=character_role,
            task_id=task_id,
            start_time=datetime.now(),
            end_time=None,
            total_steps=0,
            total_reward=0.0,
            final_status="in_progress",
            max_depth_reached=1,
            achievements={},
        )

        self.frames = []
        self.is_recording = True
        self.current_step = 0
        self.total_reward = 0.0
        self.max_depth = 1

        return trajectory_id

    def record_step(
        self,
        action: str,
        observation: Dict[str, Any],
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ):
        """Record a single step."""
        if not self.is_recording:
            raise ValueError("Recording not started. Call start_recording first.")

        frame = TrajectoryFrame(
            step=self.current_step,
            action=action,
            observation=observation.copy(),
            reward=reward,
            done=done,
            info=info.copy(),
            timestamp=datetime.now().timestamp(),
        )

        self.frames.append(frame)
        self.current_step += 1
        self.total_reward += reward

        # Update max depth if available
        if "player_stats" in observation:
            depth = observation["player_stats"].get("depth", 1)
            self.max_depth = max(self.max_depth, depth)

    def stop_recording(
        self,
        final_status: str = "completed",
        achievements: Optional[Dict[str, bool]] = None,
    ):
        """Stop recording and finalize metadata."""
        if not self.is_recording:
            return

        self.is_recording = False

        if self.metadata:
            self.metadata.end_time = datetime.now()
            self.metadata.total_steps = self.current_step
            self.metadata.total_reward = self.total_reward
            self.metadata.final_status = final_status
            self.metadata.max_depth_reached = self.max_depth
            if achievements:
                self.metadata.achievements = achievements

    def save_trajectory(self, filename: Optional[str] = None) -> str:
        """Save trajectory to disk."""
        if not self.metadata:
            raise ValueError("No trajectory to save")

        if filename is None:
            filename = f"{self.metadata.trajectory_id}.trajectory.gz"

        filepath = self.save_dir / filename

        trajectory_data = {
            "metadata": self.metadata.to_dict(),
            "frames": [frame.to_dict() for frame in self.frames],
        }

        # Save as compressed JSON
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            json.dump(trajectory_data, f, indent=2)

        # Also save a quick info file
        info_file = self.save_dir / f"{self.metadata.trajectory_id}.info.json"
        with open(info_file, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

        return str(filepath)

    @classmethod
    def load_trajectory(
        cls, filepath: str
    ) -> Tuple["TrajectoryRecorder", TrajectoryMetadata, List[TrajectoryFrame]]:
        """Load a trajectory from disk."""
        recorder = cls()

        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            data = json.load(f)

        metadata = TrajectoryMetadata.from_dict(data["metadata"])
        frames = [TrajectoryFrame.from_dict(frame) for frame in data["frames"]]

        recorder.metadata = metadata
        recorder.frames = frames

        return recorder, metadata, frames

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the current trajectory."""
        if not self.frames:
            return {}

        actions_taken = {}
        for frame in self.frames:
            actions_taken[frame.action] = actions_taken.get(frame.action, 0) + 1

        return {
            "total_steps": len(self.frames),
            "total_reward": self.total_reward,
            "max_depth": self.max_depth,
            "actions_distribution": actions_taken,
            "unique_actions": len(actions_taken),
            "average_reward_per_step": self.total_reward / len(self.frames) if self.frames else 0,
        }
