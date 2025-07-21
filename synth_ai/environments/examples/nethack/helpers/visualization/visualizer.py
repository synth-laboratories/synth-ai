"""Visualization tools for NetHack trajectories."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime
import json

try:
    from PIL import Image, ImageDraw, ImageFont

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available. Some visualization features will be limited.")


class NetHackVisualizer:
    """Visualize NetHack game states and trajectories."""

    # Character to color mapping for ASCII visualization
    CHAR_COLORS = {
        "@": "#FF0000",  # Player - red
        "d": "#8B4513",  # Dog/pet - brown
        "f": "#FFA500",  # Cat - orange
        ">": "#0000FF",  # Stairs down - blue
        "<": "#00FF00",  # Stairs up - green
        ".": "#D3D3D3",  # Floor - light gray
        "#": "#696969",  # Wall - dark gray
        "+": "#8B4513",  # Door - brown
        "-": "#8B4513",  # Door - brown
        "|": "#8B4513",  # Door - brown
        "{": "#00CED1",  # Fountain - dark turquoise
        "}": "#4682B4",  # Pool - steel blue
        "^": "#FF1493",  # Trap - deep pink
        "%": "#FFD700",  # Food - gold
        "!": "#FF69B4",  # Potion - hot pink
        "?": "#DDA0DD",  # Scroll - plum
        "/": "#9370DB",  # Wand - medium purple
        "=": "#FFD700",  # Ring - gold
        '"': "#FF4500",  # Amulet - orange red
        "[": "#C0C0C0",  # Armor - silver
        ")": "#A9A9A9",  # Weapon - dark gray
        "*": "#FFFF00",  # Gold/gem - yellow
        "$": "#FFD700",  # Gold - gold
        "`": "#8B4513",  # Boulder/statue - brown
    }

    # Default color for unknown characters
    DEFAULT_COLOR = "#FFFFFF"

    def __init__(self, cell_size: int = 10, font_size: int = 8):
        self.cell_size = cell_size
        self.font_size = font_size

    def ascii_to_image(
        self, ascii_map: str, highlight_pos: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Convert ASCII map to colored image."""
        if not HAS_PIL:
            return self._simple_ascii_to_image(ascii_map, highlight_pos)

        lines = ascii_map.strip().split("\n")
        height = len(lines)
        width = max(len(line) for line in lines) if lines else 0

        # Create image
        img_width = width * self.cell_size
        img_height = height * self.cell_size
        image = Image.new("RGB", (img_width, img_height), color="black")
        draw = ImageDraw.Draw(image)

        # Try to load a monospace font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", self.font_size)
        except:
            font = ImageFont.load_default()

        # Draw each character
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char == " ":
                    continue

                # Get color for character
                color = self.CHAR_COLORS.get(char, self.DEFAULT_COLOR)

                # Highlight player position
                if highlight_pos and (x, y) == highlight_pos:
                    # Draw background
                    draw.rectangle(
                        [
                            x * self.cell_size,
                            y * self.cell_size,
                            (x + 1) * self.cell_size,
                            (y + 1) * self.cell_size,
                        ],
                        fill="yellow",
                    )

                # Draw character
                draw.text(
                    (x * self.cell_size + 2, y * self.cell_size),
                    char,
                    fill=color,
                    font=font,
                )

        return np.array(image)

    def _simple_ascii_to_image(
        self, ascii_map: str, highlight_pos: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Simple ASCII to image conversion without PIL."""
        lines = ascii_map.strip().split("\n")
        height = len(lines)
        width = max(len(line) for line in lines) if lines else 0

        # Create RGB image
        image = np.zeros((height * self.cell_size, width * self.cell_size, 3), dtype=np.uint8)

        # Simple character to grayscale mapping
        char_values = {
            "@": 255,  # Player - white
            "#": 64,  # Wall - dark gray
            ".": 128,  # Floor - gray
            ">": 200,  # Stairs - light
            "<": 200,  # Stairs - light
            " ": 0,  # Empty - black
        }

        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                value = char_values.get(char, 100)
                y_start = y * self.cell_size
                y_end = (y + 1) * self.cell_size
                x_start = x * self.cell_size
                x_end = (x + 1) * self.cell_size

                if highlight_pos and (x, y) == highlight_pos:
                    # Highlight player in red
                    image[y_start:y_end, x_start:x_end] = [255, 0, 0]
                else:
                    image[y_start:y_end, x_start:x_end] = [value, value, value]

        return image

    def create_frame_image(
        self, observation: Dict[str, Any], include_stats: bool = True
    ) -> np.ndarray:
        """Create a single frame image from observation."""
        # Standard NetHack terminal size
        TERM_WIDTH = 80
        TERM_HEIGHT = 24

        if HAS_PIL:
            return self._create_terminal_view(observation, TERM_WIDTH, TERM_HEIGHT)
        else:
            # Fallback to simple view
            ascii_map = observation.get("ascii_map", "")
            player_stats = observation.get("player_stats", {})
            player_pos = (player_stats.get("x", 0), player_stats.get("y", 0))
            return self.ascii_to_image(ascii_map, player_pos)

    def _create_terminal_view(
        self, observation: Dict[str, Any], width: int = 80, height: int = 24
    ) -> np.ndarray:
        """Create a full terminal-style NetHack view."""
        # Create black background
        img_width = width * self.cell_size
        img_height = height * self.cell_size
        image = Image.new("RGB", (img_width, img_height), color="black")
        draw = ImageDraw.Draw(image)

        # Try to load a monospace font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", self.font_size)
        except:
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                    self.font_size,
                )
            except:
                font = ImageFont.load_default()

        # Get game data
        ascii_map = observation.get("ascii_map", "")
        player_stats = observation.get("player_stats", {})
        message = observation.get("message", "").strip()

        # Draw the map (first 21 lines)
        lines = ascii_map.split("\n")
        for y, line in enumerate(lines[:21]):
            for x, char in enumerate(line[:width]):
                if char == " ":
                    continue

                # Get color for character
                color = self.CHAR_COLORS.get(char, self.DEFAULT_COLOR)

                # Highlight player position
                if (x, y) == (player_stats.get("x", -1), player_stats.get("y", -1)):
                    # Draw yellow background for player
                    draw.rectangle(
                        [
                            x * self.cell_size,
                            y * self.cell_size,
                            (x + 1) * self.cell_size,
                            (y + 1) * self.cell_size,
                        ],
                        fill="#333300",
                    )

                # Draw character
                draw.text(
                    (x * self.cell_size + 1, y * self.cell_size),
                    char,
                    fill=color,
                    font=font,
                )

        # Draw separator line (line 21)
        draw.line(
            [(0, 21 * self.cell_size), (img_width, 21 * self.cell_size)],
            fill="#666666",
            width=1,
        )

        # Draw status line (line 22)
        status_y = 22 * self.cell_size
        character_name = observation.get("character_name", "Agent")
        character_role = observation.get("character_role", "Adventurer")

        # Format status line like real NetHack
        status_parts = [
            f"{character_name} the {character_role}",
            f"St:{player_stats.get('strength', 10)}",
            f"Dx:{player_stats.get('dexterity', 10)}",
            f"Co:{player_stats.get('constitution', 10)}",
            f"In:{player_stats.get('intelligence', 10)}",
            f"Wi:{player_stats.get('wisdom', 10)}",
            f"Ch:{player_stats.get('charisma', 10)}",
            "Neutral",  # Alignment
        ]

        status_line = "  ".join(status_parts)
        draw.text((5, status_y), status_line, fill="white", font=font)

        # Draw second status line (line 23)
        status2_y = 23 * self.cell_size
        dlvl = player_stats.get("depth", 1)
        gold = player_stats.get("gold", 0)
        hp = player_stats.get("hp", 10)
        max_hp = player_stats.get("max_hp", 10)
        pw = player_stats.get("energy", 0)
        max_pw = player_stats.get("max_energy", 0)
        ac = player_stats.get("ac", 10)
        xp = player_stats.get("experience_level", 1)

        status2_parts = [
            f"Dlvl:{dlvl}",
            f"$:{gold}",
            f"HP:{hp}({max_hp})",
            f"Pw:{pw}({max_pw})",
            f"AC:{ac}",
            f"Xp:{xp}",
        ]

        # Add turn count if available
        if "turn_count" in observation:
            status2_parts.append(f"T:{observation['turn_count']}")

        status2_line = "  ".join(status2_parts)
        draw.text((5, status2_y), status2_line, fill="white", font=font)

        # Draw message at top if present
        if message:
            # Clear message area (line 0)
            draw.rectangle([0, 0, img_width, self.cell_size], fill="black")
            draw.text((5, 0), message[: width - 1], fill="white", font=font)

        return np.array(image)

    def create_trajectory_video(
        self,
        frames: List[Dict[str, Any]],
        output_path: str,
        fps: int = 4,
        include_stats: bool = True,
    ) -> str:
        """Create a video from trajectory frames."""
        if not frames:
            raise ValueError("No frames to create video from")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.axis("off")

        # Create first frame
        first_img = self.create_frame_image(frames[0]["observation"], include_stats)
        im = ax.imshow(first_img)

        def animate(i):
            if i < len(frames):
                img = self.create_frame_image(frames[i]["observation"], include_stats)
                im.set_array(img)
                ax.set_title(f"Step {i}: {frames[i]['action']}")
            return [im]

        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(frames), interval=1000 / fps, blit=True
        )

        # Save as video
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".gif":
            anim.save(str(output_path), writer="pillow", fps=fps)
        else:
            anim.save(str(output_path), writer="ffmpeg", fps=fps)

        plt.close()

        return str(output_path)

    def plot_trajectory_stats(
        self, frames: List[Dict[str, Any]], output_path: Optional[str] = None
    ):
        """Plot statistics from a trajectory."""
        if not frames:
            return

        # Extract data
        steps = []
        rewards = []
        depths = []
        hps = []
        positions_x = []
        positions_y = []

        cumulative_reward = 0
        for i, frame in enumerate(frames):
            steps.append(i)
            cumulative_reward += frame["reward"]
            rewards.append(cumulative_reward)

            stats = frame["observation"].get("player_stats", {})
            depths.append(stats.get("depth", 1))
            hps.append(stats.get("hp", 0))
            positions_x.append(stats.get("x", 0))
            positions_y.append(stats.get("y", 0))

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Cumulative reward
        axes[0, 0].plot(steps, rewards, "b-")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Cumulative Reward")
        axes[0, 0].set_title("Reward Progress")
        axes[0, 0].grid(True)

        # Depth progression
        axes[0, 1].plot(steps, depths, "g-")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Dungeon Depth")
        axes[0, 1].set_title("Depth Exploration")
        axes[0, 1].grid(True)

        # HP over time
        axes[1, 0].plot(steps, hps, "r-")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Hit Points")
        axes[1, 0].set_title("Health Over Time")
        axes[1, 0].grid(True)

        # Position heatmap
        axes[1, 1].scatter(positions_x, positions_y, c=steps, cmap="viridis", s=1)
        axes[1, 1].set_xlabel("X Position")
        axes[1, 1].set_ylabel("Y Position")
        axes[1, 1].set_title("Movement Pattern")
        axes[1, 1].invert_yaxis()  # Invert Y axis for proper orientation

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            plt.close()
        else:
            plt.show()

    def plot_action_distribution(
        self, frames: List[Dict[str, Any]], output_path: Optional[str] = None
    ):
        """Plot distribution of actions taken."""
        if not frames:
            return

        # Count actions
        action_counts = {}
        for frame in frames:
            action = frame["action"]
            action_counts[action] = action_counts.get(action, 0) + 1

        # Sort by frequency
        actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)

        # Create bar plot
        plt.figure(figsize=(12, 6))
        actions_list, counts = zip(*actions[:20])  # Top 20 actions

        plt.bar(range(len(actions_list)), counts)
        plt.xticks(range(len(actions_list)), actions_list, rotation=45, ha="right")
        plt.xlabel("Action")
        plt.ylabel("Count")
        plt.title("Action Distribution (Top 20)")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            plt.close()
        else:
            plt.show()
