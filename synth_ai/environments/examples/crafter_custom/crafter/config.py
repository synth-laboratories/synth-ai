"""
Configuration system for customizable Crafter environment.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class WorldGenConfig:
    """Configuration for world generation parameters."""

    # Material generation thresholds and probabilities
    coal_threshold: float = 0.0
    coal_probability: float = 0.15
    iron_threshold: float = 0.4
    iron_probability: float = 0.25
    diamond_threshold: float = 0.18
    diamond_probability: float = 0.006
    tree_threshold: float = 0.0
    tree_probability: float = 0.2
    lava_threshold: float = 0.35

    # Cave generation
    cave_threshold: float = 0.6

    # Initial spawn configuration
    cow_spawn_probability: float = 0.015
    cow_min_distance: int = 3
    zombie_spawn_probability: float = 0.007
    zombie_min_distance: int = 10
    skeleton_spawn_probability: float = 0.05

    # Dynamic spawn configuration for zombies
    zombie_spawn_rate: float = 0.3
    zombie_despawn_rate: float = 0.4
    zombie_min_spawn_distance: int = 6
    zombie_max_count: float = 3.5
    zombie_min_count: float = 0.0

    # Dynamic spawn configuration for skeletons
    skeleton_spawn_rate: float = 0.1
    skeleton_despawn_rate: float = 0.1
    skeleton_min_spawn_distance: int = 7
    skeleton_max_count: float = 2.0
    skeleton_min_count: float = 0.0

    # Dynamic spawn configuration for cows
    cow_spawn_rate: float = 0.01
    cow_despawn_rate: float = 0.1
    cow_min_spawn_distance: int = 5
    cow_max_count: float = 2.5
    cow_min_count: float = 0.0

    # World generation areas
    spawn_clear_radius: int = 4
    water_threshold: float = -0.3
    sand_threshold: float = -0.2
    mountain_threshold: float = 0.15

    @classmethod
    def from_json(cls, path: str) -> "WorldGenConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_preset(cls, preset: str) -> "WorldGenConfig":
        """Load configuration from preset name."""
        config_dir = os.path.dirname(__file__)
        preset_path = os.path.join(config_dir, "config", f"{preset}.json")
        if os.path.exists(preset_path):
            return cls.from_json(preset_path)
        else:
            print(f"Warning: preset '{preset}' not found, using default config")
            return cls()

    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# Preset configurations
PRESETS = {
    "easy": WorldGenConfig(
        # More resources
        coal_threshold=-0.2,
        coal_probability=0.25,
        iron_threshold=0.2,
        iron_probability=0.35,
        diamond_threshold=0.15,
        diamond_probability=0.02,
        tree_threshold=-0.2,
        tree_probability=0.35,
        lava_threshold=0.5,
        # More friendly spawns
        cow_spawn_probability=0.03,
        cow_min_distance=2,
        zombie_spawn_probability=0.003,
        zombie_min_distance=15,
        skeleton_spawn_probability=0.01,
        # Fewer dynamic enemies
        zombie_spawn_rate=0.1,
        zombie_despawn_rate=0.6,
        zombie_min_spawn_distance=10,
        zombie_max_count=1.5,
        skeleton_spawn_rate=0.02,
        skeleton_despawn_rate=0.3,
        skeleton_min_spawn_distance=12,
        skeleton_max_count=0.5,
        cow_spawn_rate=0.03,
        cow_despawn_rate=0.05,
        cow_min_spawn_distance=3,
        cow_max_count=4.0,
    ),
    "hard": WorldGenConfig(
        # Fewer resources
        coal_threshold=0.2,
        coal_probability=0.08,
        iron_threshold=0.5,
        iron_probability=0.15,
        diamond_threshold=0.25,
        diamond_probability=0.002,
        tree_threshold=0.2,
        tree_probability=0.1,
        lava_threshold=0.25,
        # More dangerous spawns
        cow_spawn_probability=0.005,
        cow_min_distance=5,
        zombie_spawn_probability=0.02,
        zombie_min_distance=6,
        skeleton_spawn_probability=0.15,
        # Many dynamic enemies
        zombie_spawn_rate=0.5,
        zombie_despawn_rate=0.2,
        zombie_min_spawn_distance=4,
        zombie_max_count=6.0,
        skeleton_spawn_rate=0.3,
        skeleton_despawn_rate=0.05,
        skeleton_min_spawn_distance=5,
        skeleton_max_count=4.0,
        cow_spawn_rate=0.002,
        cow_despawn_rate=0.2,
        cow_min_spawn_distance=8,
        cow_max_count=1.0,
    ),
    "peaceful": WorldGenConfig(
        # Abundant resources
        coal_threshold=-0.3,
        coal_probability=0.3,
        iron_threshold=0.1,
        iron_probability=0.4,
        diamond_threshold=0.1,
        diamond_probability=0.03,
        tree_threshold=-0.3,
        tree_probability=0.4,
        lava_threshold=0.6,
        # No enemies, many cows
        cow_spawn_probability=0.05,
        cow_min_distance=2,
        zombie_spawn_probability=0.0,
        zombie_min_distance=100,
        skeleton_spawn_probability=0.0,
        # No dynamic enemies
        zombie_spawn_rate=0.0,
        zombie_despawn_rate=1.0,
        zombie_min_spawn_distance=100,
        zombie_max_count=0.0,
        skeleton_spawn_rate=0.0,
        skeleton_despawn_rate=1.0,
        skeleton_min_spawn_distance=100,
        skeleton_max_count=0.0,
        cow_spawn_rate=0.05,
        cow_despawn_rate=0.02,
        cow_min_spawn_distance=2,
        cow_max_count=5.0,
    ),
}
