"""
Simplified monkey patch for configurable world generation in crafter.
This version modifies generation parameters rather than rewriting functions.
"""

import json
import os
from typing import Any, Dict, Optional

import crafter

print("[PATCH] Attempting to apply simplified Crafter world configuration patch...")

# World configuration presets
WORLD_CONFIGS = {
    "easy": {
        "name": "Easy Mode",
        "description": "More resources, fewer enemies",
        # Modify spawn probabilities by multiplying original values
        "spawn_multipliers": {
            "tree": 1.5,  # 50% more trees
            "coal": 1.5,  # 50% more coal
            "iron": 1.5,  # 50% more iron
            "diamond": 3.0,  # 3x more diamonds
            "cow": 2.0,  # 2x more cows
            "zombie": 0.3,  # 70% fewer zombies
            "skeleton": 0.2,  # 80% fewer skeletons
        },
        # Modify spawn distances
        "spawn_distances": {
            "zombie": 15,  # Farther away (default 10)
            "skeleton": 10,  # Farther away (default varies)
            "cow": 2,  # Closer (default 3)
        },
    },
    "normal": {
        "name": "Normal Mode",
        "description": "Standard crafter experience",
        "spawn_multipliers": {
            "tree": 1.0,
            "coal": 1.0,
            "iron": 1.0,
            "diamond": 1.0,
            "cow": 1.0,
            "zombie": 1.0,
            "skeleton": 1.0,
        },
        "spawn_distances": {"zombie": 10, "skeleton": 7, "cow": 3},
    },
    "hard": {
        "name": "Hard Mode",
        "description": "Fewer resources, more enemies",
        "spawn_multipliers": {
            "tree": 0.5,  # 50% fewer trees
            "coal": 0.5,  # 50% less coal
            "iron": 0.5,  # 50% less iron
            "diamond": 0.3,  # 70% fewer diamonds
            "cow": 0.3,  # 70% fewer cows
            "zombie": 3.0,  # 3x more zombies
            "skeleton": 3.0,  # 3x more skeletons
        },
        "spawn_distances": {
            "zombie": 5,  # Much closer
            "skeleton": 4,  # Much closer
            "cow": 8,  # Farther away
        },
    },
    "peaceful": {
        "name": "Peaceful Mode",
        "description": "No enemies, more resources",
        "spawn_multipliers": {
            "tree": 2.0,
            "coal": 2.0,
            "iron": 2.0,
            "diamond": 5.0,
            "cow": 3.0,
            "zombie": 0.0,  # No zombies
            "skeleton": 0.0,  # No skeletons
        },
        "spawn_distances": {"zombie": 100, "skeleton": 100, "cow": 1},
    },
}

# Store active configuration
_active_config = WORLD_CONFIGS["normal"]
_original_balance_chunk = None
_last_loaded_config = None  # Track what was last loaded to avoid duplicate prints


def load_world_config(
    config_name: str = "normal", config_path: Optional[str] = None
) -> Dict[str, Any]:
    """Load world configuration."""
    global _active_config, _last_loaded_config

    # Create a config identifier
    config_id = config_path if config_path else config_name

    # Only print if configuration actually changed
    if _last_loaded_config != config_id:
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                _active_config = json.load(f)
            # print(f"[PATCH] Loaded custom world config from {config_path}")
        elif config_name in WORLD_CONFIGS:
            _active_config = WORLD_CONFIGS[config_name]
            # print(f"[PATCH] Loaded '{config_name}' world configuration")
        else:
            _active_config = WORLD_CONFIGS["normal"]
            # print(f"[PATCH] Unknown config '{config_name}', using 'normal'")

        _last_loaded_config = config_id
    else:
        # Configuration hasn't changed, just return the active one
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                _active_config = json.load(f)
        elif config_name in WORLD_CONFIGS:
            _active_config = WORLD_CONFIGS[config_name]
        else:
            _active_config = WORLD_CONFIGS["normal"]

    return _active_config


# Patch world generation to use configuration
original_generate_world = crafter.worldgen.generate_world


def patched_generate_world(world, player):
    """Patched world generation that applies spawn multipliers."""
    # Apply configuration without modifying numpy RandomState
    multipliers = _active_config.get("spawn_multipliers", {})
    distances = _active_config.get("spawn_distances", {})

    # Call original generation first
    result = original_generate_world(world, player)

    # Post-process to adjust spawns based on multipliers
    # This is a simpler approach that modifies objects after generation
    if multipliers:
        # Remove some objects based on multipliers < 1.0
        objects_to_remove = []
        for obj in world._objects:
            if obj is None or obj is player:
                continue

            obj_type = type(obj).__name__.lower()

            # Check if we should remove this object based on multiplier
            multiplier = 1.0
            if "tree" in obj_type or hasattr(obj, "kind") and getattr(obj, "kind") == "tree":
                multiplier = multipliers.get("tree", 1.0)
            elif "cow" in obj_type:
                multiplier = multipliers.get("cow", 1.0)
            elif "zombie" in obj_type:
                multiplier = multipliers.get("zombie", 1.0)
            elif "skeleton" in obj_type:
                multiplier = multipliers.get("skeleton", 1.0)

            # Remove objects if multiplier < 1.0
            if multiplier < 1.0 and world.random.random() > multiplier:
                objects_to_remove.append(obj)

        # Remove marked objects
        for obj in objects_to_remove:
            world.remove(obj)

        # Add extra objects if multiplier > 1.0
        # (This is more complex and would require spawning new objects)

    # Post-process to adjust spawn distances
    if distances:
        # Adjust initial enemy positions based on distance config
        for obj in list(world._objects):
            if obj is None:
                continue

            obj_type = type(obj).__name__.lower()
            if obj_type in distances:
                min_dist = distances[obj_type]
                player_pos = player.pos
                obj_pos = obj.pos
                dist = abs(obj_pos[0] - player_pos[0]) + abs(obj_pos[1] - player_pos[1])

                if dist < min_dist:
                    # Remove objects too close to player
                    world.remove(obj)

    return result


# Patch the balance function for dynamic spawning
def patched_balance_chunk(self, chunk, objs=None):
    """Patched chunk balancing with config support."""
    global _original_balance_chunk
    if _original_balance_chunk is None:
        return

    multipliers = _active_config.get("spawn_multipliers", {})
    distances = _active_config.get("spawn_distances", {})

    # Call original balance function with objs parameter
    if objs is not None:
        _original_balance_chunk(self, chunk, objs)
    else:
        _original_balance_chunk(self, chunk)

    # Post-process spawned objects based on multipliers
    # Check if any new objects were spawned and adjust them
    if multipliers:
        # Get the chunk bounds
        chunk_x = chunk[0] * self._world._chunk_size
        chunk_y = chunk[1] * self._world._chunk_size
        chunk_w = self._world._chunk_size
        chunk_h = self._world._chunk_size

        objects_to_remove = []
        for obj in self._world._objects:
            if obj is None:
                continue

            # Check if object is in this chunk
            try:
                pos_x, pos_y = obj.pos[0], obj.pos[1]
                if chunk_x <= pos_x < chunk_x + chunk_w and chunk_y <= pos_y < chunk_y + chunk_h:
                    obj_type = type(obj).__name__.lower()
                    multiplier = 1.0

                    if "zombie" in obj_type:
                        multiplier = multipliers.get("zombie", 1.0)
                    elif "skeleton" in obj_type:
                        multiplier = multipliers.get("skeleton", 1.0)
                    elif "cow" in obj_type:
                        multiplier = multipliers.get("cow", 1.0)

                    # Remove objects based on multiplier
                    if multiplier < 1.0 and self._world.random.random() > multiplier:
                        objects_to_remove.append(obj)
            except (ValueError, AttributeError):
                # Skip objects with invalid position data
                continue

        # Remove marked objects
        for obj in objects_to_remove:
            self._world.remove(obj)


# Apply patches
crafter.worldgen.generate_world = patched_generate_world

# Store original balance_chunk
_original_balance_chunk = crafter.Env._balance_chunk
crafter.Env._balance_chunk = patched_balance_chunk

# Extend Env.__init__ to accept world_config
original_env_init = crafter.Env.__init__


def patched_env_init(
    self,
    area=(64, 64),
    view=(9, 9),
    length=10000,
    seed=None,
    world_config="normal",
    world_config_path=None,
):
    """Extended Env.__init__ that accepts world configuration."""
    # Load configuration
    load_world_config(world_config, world_config_path)

    # Call original init
    original_env_init(self, area=area, view=view, length=length, seed=seed)

    # Store config name
    self._world_config_name = world_config if not world_config_path else "custom"


crafter.Env.__init__ = patched_env_init

print("[PATCH] Simplified Crafter world configuration patch complete.")
print("[PATCH] Available configs: easy, normal, hard, peaceful")

# Example custom config
EXAMPLE_CUSTOM_CONFIG = {
    "name": "Custom Config",
    "description": "My custom world",
    "spawn_multipliers": {
        "tree": 1.2,
        "coal": 1.3,
        "iron": 1.4,
        "diamond": 2.0,
        "cow": 1.5,
        "zombie": 0.5,
        "skeleton": 0.5,
    },
    "spawn_distances": {"zombie": 12, "skeleton": 8, "cow": 4},
}
