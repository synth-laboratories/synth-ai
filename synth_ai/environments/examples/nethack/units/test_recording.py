"""Unit tests for NetHack trajectory recording and visualization."""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from datetime import datetime

from src.synth_env.examples.nethack.helpers.trajectory_recorder import (
    TrajectoryRecorder,
    TrajectoryFrame,
    TrajectoryMetadata,
)
from src.synth_env.examples.nethack.helpers.visualization.visualizer import (
    NetHackVisualizer,
)


class TestTrajectoryRecorder(unittest.TestCase):
    """Test trajectory recording functionality."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.test_dir = tempfile.mkdtemp()
        self.recorder = TrajectoryRecorder(self.test_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_start_recording(self):
        """Test starting a recording."""
        trajectory_id = self.recorder.start_recording("wizard", "test_task")

        self.assertIsNotNone(trajectory_id)
        self.assertTrue(self.recorder.is_recording)
        self.assertEqual(self.recorder.metadata.character_role, "wizard")
        self.assertEqual(self.recorder.metadata.task_id, "test_task")
        self.assertEqual(self.recorder.current_step, 0)

    def test_record_step(self):
        """Test recording individual steps."""
        self.recorder.start_recording("knight")

        # Create test observation
        obs = {
            "player_stats": {
                "x": 10,
                "y": 20,
                "hp": 16,
                "max_hp": 16,
                "depth": 1,
                "gold": 0,
                "experience_level": 1,
            },
            "ascii_map": "###\n#@#\n###",
            "message": "Welcome to NetHack!",
            "inventory": [],
        }

        # Record steps
        self.recorder.record_step("north", obs, 0.0, False, {})
        self.recorder.record_step("east", obs, 1.0, False, {"extra": "info"})

        self.assertEqual(len(self.recorder.frames), 2)
        self.assertEqual(self.recorder.current_step, 2)
        self.assertEqual(self.recorder.total_reward, 1.0)

        # Check frame contents
        frame1 = self.recorder.frames[0]
        self.assertEqual(frame1.action, "north")
        self.assertEqual(frame1.reward, 0.0)
        self.assertFalse(frame1.done)

        frame2 = self.recorder.frames[1]
        self.assertEqual(frame2.action, "east")
        self.assertEqual(frame2.reward, 1.0)
        self.assertEqual(frame2.info["extra"], "info")

    def test_stop_recording(self):
        """Test stopping a recording."""
        self.recorder.start_recording("monk")

        obs = {"player_stats": {"depth": 3}}
        self.recorder.record_step("down", obs, 5.0, False, {})

        self.recorder.stop_recording("completed", {"achievement1": True})

        self.assertFalse(self.recorder.is_recording)
        self.assertEqual(self.recorder.metadata.final_status, "completed")
        self.assertEqual(self.recorder.metadata.max_depth_reached, 3)
        self.assertEqual(self.recorder.metadata.achievements, {"achievement1": True})

    def test_save_and_load_trajectory(self):
        """Test saving and loading trajectories."""
        # Create and save trajectory
        self.recorder.start_recording("rogue", "test_save")

        obs = {
            "player_stats": {"x": 5, "y": 10, "depth": 2},
            "ascii_map": "test_map",
            "message": "test message",
        }

        self.recorder.record_step("wait", obs, 0.5, False, {})
        self.recorder.record_step("search", obs, 1.5, True, {"final": True})
        self.recorder.stop_recording("died")

        filepath = self.recorder.save_trajectory()
        self.assertTrue(Path(filepath).exists())

        # Load trajectory
        loaded_recorder, metadata, frames = TrajectoryRecorder.load_trajectory(filepath)

        # Check metadata
        self.assertEqual(metadata.character_role, "rogue")
        self.assertEqual(metadata.task_id, "test_save")
        self.assertEqual(metadata.total_steps, 2)
        self.assertEqual(metadata.total_reward, 2.0)
        self.assertEqual(metadata.final_status, "died")

        # Check frames
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].action, "wait")
        self.assertEqual(frames[1].action, "search")
        self.assertTrue(frames[1].done)

    def test_numpy_serialization(self):
        """Test serialization of numpy arrays in observations."""
        self.recorder.start_recording("wizard")

        # Create observation with numpy arrays
        obs = {
            "map_array": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
            "position": np.array([10, 20]),
            "nested": {"data": np.zeros((2, 2), dtype=np.float32)},
        }

        self.recorder.record_step("test", obs, 0.0, False, {})
        self.recorder.stop_recording()

        # Save and load
        filepath = self.recorder.save_trajectory()
        _, _, frames = TrajectoryRecorder.load_trajectory(filepath)

        # Check numpy arrays are restored correctly
        loaded_obs = frames[0].observation
        np.testing.assert_array_equal(loaded_obs["map_array"], obs["map_array"])
        np.testing.assert_array_equal(loaded_obs["position"], obs["position"])
        np.testing.assert_array_equal(loaded_obs["nested"]["data"], obs["nested"]["data"])

    def test_get_summary(self):
        """Test trajectory summary generation."""
        self.recorder.start_recording("barbarian")

        obs = {"player_stats": {"depth": 1}}

        # Record various actions
        self.recorder.record_step("north", obs, 1.0, False, {})
        self.recorder.record_step("north", obs, 0.0, False, {})
        self.recorder.record_step("east", obs, 2.0, False, {})
        self.recorder.record_step("search", obs, 0.0, False, {})
        self.recorder.record_step("north", obs, 1.0, True, {})

        summary = self.recorder.get_summary()

        self.assertEqual(summary["total_steps"], 5)
        self.assertEqual(summary["total_reward"], 4.0)
        self.assertEqual(summary["unique_actions"], 3)
        self.assertEqual(summary["actions_distribution"]["north"], 3)
        self.assertEqual(summary["average_reward_per_step"], 0.8)


class TestNetHackVisualizer(unittest.TestCase):
    """Test visualization functionality."""

    def setUp(self):
        """Create visualizer instance."""
        self.viz = NetHackVisualizer(cell_size=10, font_size=8)
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)

    def test_ascii_to_image(self):
        """Test ASCII map to image conversion."""
        ascii_map = """######
#....#
#.@..#
#....#
######"""

        # Test without highlight
        img = self.viz.ascii_to_image(ascii_map)
        self.assertEqual(img.shape[0], 5 * self.viz.cell_size)  # 5 rows
        self.assertEqual(img.shape[1], 6 * self.viz.cell_size)  # 6 cols
        self.assertEqual(img.shape[2], 3)  # RGB

        # Test with highlight
        img_highlight = self.viz.ascii_to_image(ascii_map, highlight_pos=(2, 2))
        self.assertIsNotNone(img_highlight)

    def test_create_frame_image(self):
        """Test frame image creation."""
        obs = {
            "ascii_map": "#####\n#@..#\n#####",
            "player_stats": {
                "x": 1,
                "y": 1,
                "hp": 10,
                "max_hp": 10,
                "experience_level": 1,
                "depth": 1,
                "gold": 0,
            },
            "message": "Test message",
        }

        # Test without stats
        img = self.viz.create_frame_image(obs, include_stats=False)
        self.assertEqual(len(img.shape), 3)

        # Test with stats (if PIL available)
        img_stats = self.viz.create_frame_image(obs, include_stats=True)
        self.assertEqual(len(img_stats.shape), 3)

    def test_plot_trajectory_stats(self):
        """Test trajectory statistics plotting."""
        frames = []
        for i in range(10):
            frames.append(
                {
                    "action": "north" if i % 2 == 0 else "south",
                    "observation": {
                        "player_stats": {
                            "x": i,
                            "y": 10 - i,
                            "depth": 1 + i // 5,
                            "hp": 10 - i // 2,
                        }
                    },
                    "reward": 1.0 if i % 3 == 0 else 0.0,
                }
            )

        output_path = Path(self.test_dir) / "test_stats.png"
        self.viz.plot_trajectory_stats(frames, str(output_path))

        self.assertTrue(output_path.exists())

    def test_plot_action_distribution(self):
        """Test action distribution plotting."""
        frames = []
        actions = ["north", "south", "east", "west", "wait", "search"]

        for i in range(20):
            frames.append({"action": actions[i % len(actions)], "observation": {}, "reward": 0.0})

        output_path = Path(self.test_dir) / "test_actions.png"
        self.viz.plot_action_distribution(frames, str(output_path))

        self.assertTrue(output_path.exists())


class TestTrajectoryFrame(unittest.TestCase):
    """Test TrajectoryFrame serialization."""

    def test_frame_serialization(self):
        """Test frame to_dict and from_dict."""
        frame = TrajectoryFrame(
            step=5,
            action="test_action",
            observation={
                "array_data": np.array([1, 2, 3]),
                "nested": {"matrix": np.eye(2, dtype=np.float32)},
                "list_data": [1, 2, 3],
                "string": "test",
            },
            reward=2.5,
            done=False,
            info={"extra": "data"},
            timestamp=datetime.now().timestamp(),
        )

        # Serialize
        frame_dict = frame.to_dict()

        # Check numpy arrays are converted
        self.assertEqual(frame_dict["observation"]["array_data"]["type"], "ndarray")
        self.assertEqual(frame_dict["observation"]["nested"]["matrix"]["type"], "ndarray")

        # Deserialize
        restored_frame = TrajectoryFrame.from_dict(frame_dict)

        # Check restoration
        self.assertEqual(restored_frame.step, 5)
        self.assertEqual(restored_frame.action, "test_action")
        np.testing.assert_array_equal(restored_frame.observation["array_data"], np.array([1, 2, 3]))
        np.testing.assert_array_equal(
            restored_frame.observation["nested"]["matrix"], np.eye(2, dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
