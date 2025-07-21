"""Interactive replay viewer for NetHack trajectories."""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import gzip
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent.parent))

from src.synth_env.examples.nethack.helpers.trajectory_recorder import (
    TrajectoryRecorder,
    TrajectoryFrame,
)
from src.synth_env.examples.nethack.helpers.visualization.visualizer import (
    NetHackVisualizer,
)


class ReplayViewer:
    """Interactive viewer for NetHack trajectory replays."""

    def __init__(self, trajectory_path: str):
        """Initialize replay viewer with a trajectory file."""
        self.trajectory_path = Path(trajectory_path)

        # Load trajectory
        self.recorder, self.metadata, self.frames = TrajectoryRecorder.load_trajectory(
            str(trajectory_path)
        )

        # Initialize visualizer
        self.visualizer = NetHackVisualizer()

        # Playback state
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 1.0  # Frames per second

        print(f"Loaded trajectory: {self.metadata.trajectory_id}")
        print(f"Character: {self.metadata.character_role}")
        print(f"Total steps: {self.metadata.total_steps}")
        print(f"Total reward: {self.metadata.total_reward:.2f}")
        print(f"Final status: {self.metadata.final_status}")
        print(f"Max depth reached: {self.metadata.max_depth_reached}")

    def show_frame(self, frame_idx: int):
        """Display a specific frame."""
        if 0 <= frame_idx < len(self.frames):
            frame = self.frames[frame_idx]
            obs = frame.observation

            print(f"\n{'=' * 80}")
            print(f"Frame {frame_idx}/{len(self.frames) - 1} | Step: {frame.step}")
            print(f"Action: {frame.action}")
            print(f"Reward: {frame.reward:+.2f}")

            # Show message
            message = obs.get("message", "").strip()
            if message:
                print(f"Message: {message}")

            # Show stats
            stats = obs.get("player_stats", {})
            print(
                f"Position: ({stats.get('x', 0)}, {stats.get('y', 0)}) | "
                f"HP: {stats.get('hp', 0)}/{stats.get('max_hp', 0)} | "
                f"Level: {stats.get('experience_level', 1)} | "
                f"Depth: {stats.get('depth', 1)} | "
                f"Gold: {stats.get('gold', 0)}"
            )

            # Show map
            ascii_map = obs.get("ascii_map", "")
            if ascii_map:
                lines = ascii_map.split("\n")
                px, py = stats.get("x", 0), stats.get("y", 0)

                # Show area around player
                print("\nMap view:")
                for y in range(max(0, py - 10), min(len(lines), py + 11)):
                    if 0 <= y < len(lines):
                        line = lines[y]
                        start = max(0, px - 20)
                        end = min(len(line), px + 21)
                        if y == py:
                            print(f">>> {line[start:end]} <<<")
                        else:
                            print(f"    {line[start:end]}")

    def interactive_replay(self):
        """Run interactive replay session."""
        print("\n=== Interactive Replay Viewer ===")
        print("Commands:")
        print("  n/next - Next frame")
        print("  p/prev - Previous frame")
        print("  g <num> - Go to frame number")
        print("  f/first - Go to first frame")
        print("  l/last - Go to last frame")
        print("  i/info - Show trajectory info")
        print("  s/search <action> - Find frames with action")
        print("  export <type> - Export (video/stats/actions)")
        print("  q/quit - Exit viewer")

        # Show first frame
        self.show_frame(0)

        while True:
            try:
                cmd = input(f"\nFrame {self.current_frame}> ").strip().lower()

                if cmd in ["q", "quit"]:
                    break

                elif cmd in ["n", "next"]:
                    if self.current_frame < len(self.frames) - 1:
                        self.current_frame += 1
                        self.show_frame(self.current_frame)
                    else:
                        print("Already at last frame")

                elif cmd in ["p", "prev"]:
                    if self.current_frame > 0:
                        self.current_frame -= 1
                        self.show_frame(self.current_frame)
                    else:
                        print("Already at first frame")

                elif cmd.startswith("g "):
                    try:
                        frame_num = int(cmd.split()[1])
                        if 0 <= frame_num < len(self.frames):
                            self.current_frame = frame_num
                            self.show_frame(self.current_frame)
                        else:
                            print(f"Frame number must be between 0 and {len(self.frames) - 1}")
                    except (ValueError, IndexError):
                        print("Usage: g <frame_number>")

                elif cmd in ["f", "first"]:
                    self.current_frame = 0
                    self.show_frame(self.current_frame)

                elif cmd in ["l", "last"]:
                    self.current_frame = len(self.frames) - 1
                    self.show_frame(self.current_frame)

                elif cmd in ["i", "info"]:
                    self.show_trajectory_info()

                elif cmd.startswith("s ") or cmd.startswith("search "):
                    action = " ".join(cmd.split()[1:])
                    self.search_action(action)

                elif cmd.startswith("export"):
                    parts = cmd.split()
                    if len(parts) > 1:
                        self.export_trajectory(parts[1])
                    else:
                        print("Usage: export <video|stats|actions>")

                else:
                    print("Unknown command. Type 'q' to quit.")

            except KeyboardInterrupt:
                print("\nUse 'q' to quit")
            except Exception as e:
                print(f"Error: {e}")

    def show_trajectory_info(self):
        """Display trajectory information and statistics."""
        print(f"\n=== Trajectory Information ===")
        print(f"ID: {self.metadata.trajectory_id}")
        print(f"Character: {self.metadata.character_role}")
        print(f"Task ID: {self.metadata.task_id or 'N/A'}")
        print(f"Start time: {self.metadata.start_time}")
        print(f"End time: {self.metadata.end_time}")
        print(
            f"Duration: {(self.metadata.end_time - self.metadata.start_time).total_seconds():.1f} seconds"
        )
        print(f"Total steps: {self.metadata.total_steps}")
        print(f"Total reward: {self.metadata.total_reward:.2f}")
        print(f"Average reward/step: {self.metadata.total_reward / self.metadata.total_steps:.4f}")
        print(f"Final status: {self.metadata.final_status}")
        print(f"Max depth: {self.metadata.max_depth_reached}")

        # Action distribution
        action_counts = {}
        for frame in self.frames:
            action_counts[frame.action] = action_counts.get(frame.action, 0) + 1

        print(f"\nTop 10 actions:")
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {action}: {count} ({count / len(self.frames) * 100:.1f}%)")

    def search_action(self, action: str):
        """Search for frames containing specific action."""
        matches = []
        for i, frame in enumerate(self.frames):
            if action.lower() in frame.action.lower():
                matches.append(i)

        if matches:
            print(f"Found {len(matches)} frames with action '{action}':")
            for i, frame_idx in enumerate(matches[:10]):  # Show first 10
                frame = self.frames[frame_idx]
                print(f"  Frame {frame_idx}: {frame.action} (reward: {frame.reward:+.2f})")

            if len(matches) > 10:
                print(f"  ... and {len(matches) - 10} more")
        else:
            print(f"No frames found with action '{action}'")

    def export_trajectory(self, export_type: str):
        """Export trajectory in various formats."""
        output_dir = self.trajectory_path.parent / "exports"
        output_dir.mkdir(exist_ok=True)

        if export_type == "video":
            print("Creating video...")
            output_path = output_dir / f"{self.metadata.trajectory_id}.mp4"

            # Convert frames to format expected by visualizer
            vis_frames = []
            for frame in self.frames:
                vis_frames.append({"action": frame.action, "observation": frame.observation})

            try:
                video_path = self.visualizer.create_trajectory_video(
                    vis_frames, str(output_path), fps=4, include_stats=True
                )
                print(f"Video saved to: {video_path}")
            except Exception as e:
                print(f"Error creating video: {e}")
                print("Make sure ffmpeg is installed for video export")

        elif export_type == "stats":
            print("Creating statistics plots...")
            stats_path = output_dir / f"{self.metadata.trajectory_id}_stats.png"
            action_path = output_dir / f"{self.metadata.trajectory_id}_actions.png"

            # Convert frames for visualizer
            vis_frames = []
            for frame in self.frames:
                vis_frames.append(
                    {
                        "action": frame.action,
                        "observation": frame.observation,
                        "reward": frame.reward,
                    }
                )

            self.visualizer.plot_trajectory_stats(vis_frames, str(stats_path))
            self.visualizer.plot_action_distribution(vis_frames, str(action_path))
            print(f"Stats saved to: {stats_path}")
            print(f"Action distribution saved to: {action_path}")

        elif export_type == "actions":
            print("Exporting action sequence...")
            actions_path = output_dir / f"{self.metadata.trajectory_id}_actions.txt"

            with open(actions_path, "w") as f:
                f.write(f"# Trajectory: {self.metadata.trajectory_id}\n")
                f.write(f"# Character: {self.metadata.character_role}\n")
                f.write(f"# Total steps: {self.metadata.total_steps}\n")
                f.write(f"# Total reward: {self.metadata.total_reward}\n\n")

                for frame in self.frames:
                    f.write(f"{frame.step:4d}: {frame.action:15s} (reward: {frame.reward:+.2f})\n")

            print(f"Action sequence saved to: {actions_path}")

        else:
            print(f"Unknown export type: {export_type}")
            print("Available types: video, stats, actions")


def main():
    """Main entry point for replay viewer."""
    import argparse

    parser = argparse.ArgumentParser(description="NetHack Trajectory Replay Viewer")
    parser.add_argument("trajectory", help="Path to trajectory file (.trajectory.gz)")
    parser.add_argument(
        "--export",
        choices=["video", "stats", "actions"],
        help="Export trajectory without interactive mode",
    )

    args = parser.parse_args()

    if not Path(args.trajectory).exists():
        print(f"Error: Trajectory file not found: {args.trajectory}")
        sys.exit(1)

    viewer = ReplayViewer(args.trajectory)

    if args.export:
        viewer.export_trajectory(args.export)
    else:
        viewer.interactive_replay()


if __name__ == "__main__":
    main()
