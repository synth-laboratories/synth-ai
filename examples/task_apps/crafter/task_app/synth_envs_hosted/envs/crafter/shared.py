"""Shared utilities for Crafter environment and policy.

This module formats Crafter observations for the LLM and parses actions.
It now mirrors the ludic_private implementation for semantic map rendering
by dynamically deriving the id->name mapping from the actual Crafter env
when available, with a sensible fallback. This fixes the issue where the
rendered surroundings appeared only as iron/stone due to a mismatched
hardcoded mapping.
"""

import itertools
import re
from typing import Any

import numpy as np

VIEW_SIZE = 5  # Default view size for the map (match eval_rollout_table)

# Action mappings from the game
CRAFTER_ACTIONS = {
    "noop": 0,
    "move_left": 1,
    "move_right": 2,
    "move_up": 3,
    "move_down": 4,
    "do": 5,
    "sleep": 6,
    "place_stone": 7,
    "place_table": 8,
    "place_furnace": 9,
    "place_plant": 10,
    "make_wood_pickaxe": 11,
    "make_stone_pickaxe": 12,
    "make_iron_pickaxe": 13,
    "make_wood_sword": 14,
    "make_stone_sword": 15,
    "make_iron_sword": 16,
}

# Common action aliases
ACTION_ALIASES = {
    # Movement aliases
    "left": "move_left",
    "right": "move_right",
    "up": "move_up",
    "down": "move_down",
    # Interaction aliases
    "interact": "do",
    "use": "do",
    "action": "do",
    # Sleep
    "rest": "sleep",
    # Crafting
    "craft_wood_pickaxe": "make_wood_pickaxe",
    "craft_stone_pickaxe": "make_stone_pickaxe",
    "craft_iron_pickaxe": "make_iron_pickaxe",
    "craft_wood_sword": "make_wood_sword",
    "craft_stone_sword": "make_stone_sword",
    "craft_iron_sword": "make_iron_sword",
}

VALID_PRIMARY_ACTIONS: set[str] = set(CRAFTER_ACTIONS.keys())
VALID_ACTION_ALIASES: set[str] = set(ACTION_ALIASES.keys())
ALL_VALID_ACTION_STRINGS: set[str] = VALID_PRIMARY_ACTIONS | VALID_ACTION_ALIASES


def validate_action(action: str) -> bool:
    """Check if an action string is valid."""
    normalized = action.strip().lower().replace(" ", "_")
    return normalized in ALL_VALID_ACTION_STRINGS


def parse_actions(action_text: str) -> list[str]:
    """Extract actions from response text.

    Tries multiple parsing strategies:
    1. <action>...</action> tags (original format)
    2. [action]...[/action] or [action]... format
    3. ACTION: prefix format
    4. Plain action names if they match valid actions
    5. Newline-separated actions
    """

    # First try the original <action> tag format
    matches = re.findall(r"<action>(.*?)</action>", action_text, re.IGNORECASE)
    if matches:
        return [m.strip() for m in matches if validate_action(m.strip())]

    # Try [action] format
    matches = re.findall(r"\[action\](.*?)(?:\[/action\]|\n|$)", action_text, re.IGNORECASE)
    if matches:
        return [m.strip() for m in matches if validate_action(m.strip())]

    # If no tags found, try to parse plain text
    text = action_text.strip()

    # Check if the entire text is a valid action
    if validate_action(text):
        return [text]

    # Try splitting by newlines and checking each line
    lines = text.split("\n")
    actions = []
    for line in lines:
        line = line.strip()

        # Remove various prefixes
        for prefix in ["ACTION:", "Action:", "action:", "ACTION", "-", "*", "•", "**ACTION:**"]:
            if line.startswith(prefix):
                line = line[len(prefix) :].strip()
                break

        # Also handle numbered lists
        if re.match(r"^\d+\.\s*", line):
            line = re.sub(r"^\d+\.\s*", "", line)

        # Split by common separators to handle multiple actions on one line
        parts = re.split(r"[,;]|\s+and\s+|\s+then\s+", line)

        for part in parts:
            part = part.strip()
            # Remove quotes if present
            if part.startswith('"') and part.endswith('"'):
                part = part[1:-1]
            if part.startswith("'") and part.endswith("'"):
                part = part[1:-1]

            # Check if it's a valid action
            if part and validate_action(part):
                actions.append(part)

    return actions


def format_observation(obs_data: dict[str, Any], step_count: int = 0, max_steps: int = 100) -> str:
    """Format a Crafter observation dictionary into a human-readable string.

    This is critical for preventing massive token counts when observations
    contain large numpy arrays or deeply nested structures.
    """
    if not obs_data:
        return ""

    # Extract key information
    health = obs_data.get("health") or obs_data.get("inventory", {}).get("health", 0)
    inventory_dict = obs_data.get("inventory", {})
    pos = obs_data.get("player_position", [0, 0])
    direction = obs_data.get("player_direction", [0, 1])
    achievements = obs_data.get("achievements_status", {})

    # Prefer step/max from observation if provided by the env
    step_from_obs = (
        obs_data.get("steps")
        if obs_data.get("steps") is not None
        else obs_data.get("num_steps_taken")
    )
    if isinstance(step_from_obs, int | float) and step_from_obs >= 0:
        step_count = int(step_from_obs)

    max_steps_from_obs = obs_data.get("max_steps_episode") or obs_data.get("max_steps")
    if isinstance(max_steps_from_obs, int | float) and max_steps_from_obs > 0:
        max_steps = int(max_steps_from_obs)

    # Format inventory (skip health as it's shown separately)
    inv_items = [f"{k}:{v}" for k, v in inventory_dict.items() if v > 0 and k != "health"]
    inventory_str = ", ".join(inv_items) if inv_items else "empty"

    # Format achievements
    achieved_list = [k for k, v in achievements.items() if v]
    achievements_str = ", ".join(achieved_list) if achieved_list else "none"

    # Format semantic map view (simplified version)
    map_view = _format_semantic_map_view(obs_data, VIEW_SIZE)

    return (
        f"=== CRAFTER GAME STATE ===\n"
        f"Step: {step_count}/{max_steps}\n"
        f"Health: {health}\n"
        f"Position: {pos}\n"
        f"Facing: {direction}\n"
        f"Inventory: {inventory_str}\n"
        f"Achievements: {achievements_str}\n"
        f"{map_view}\n\n"
        f"Choose your next actions.\n"
    )


def _try_build_dynamic_mapping():
    """Attempt to build id->name mapping from a real Crafter env.

    Returns a list where index is semantic ID and value is the lowercase name.
    On failure (crafter not installed or internal API changed), returns None.
    """
    try:
        import crafter  # type: ignore
    except Exception:
        return None

    dummyenv = None
    try:
        dummyenv = crafter.Env()
        # Combine material IDs and semantic view object IDs
        world_ids = getattr(dummyenv, "_world", None)
        sem_view = getattr(dummyenv, "_sem_view", None)
        if world_ids is None or sem_view is None:
            return None
        mat_ids = getattr(world_ids, "_mat_ids", None)
        obj_ids = getattr(sem_view, "_obj_ids", None)
        if not isinstance(mat_ids, dict) or not isinstance(obj_ids, dict):
            return None
        max_id = max(max(mat_ids.values()), max(obj_ids.values())) + 1
        id_to_item = ["void"] * max_id
        for name, idx in itertools.chain(mat_ids.items(), obj_ids.items()):
            if name is None:
                clean = "none"
            elif hasattr(name, "__name__"):
                clean = name.__name__.lower()
            else:
                clean = str(name).lower()
            if 0 <= idx < len(id_to_item):
                id_to_item[idx] = clean
        return id_to_item
    except Exception:
        return None
    finally:
        try:
            if dummyenv is not None:
                dummyenv.close()
        except Exception:
            pass


# Build dynamic mapping if possible; otherwise fall back to a basic map
_ID_TO_NAME = _try_build_dynamic_mapping()
_FALLBACK_ID_TO_NAME = {
    0: "none",  # None from materials
    1: "water",
    2: "grass",
    3: "stone",
    4: "path",
    5: "sand",
    6: "tree",
    7: "lava",
    8: "coal",
    9: "iron",
    10: "diamond",
    11: "table",
    12: "furnace",
    13: "player",
    14: "cow",
    15: "zombie",
    16: "skeleton",
    17: "arrow",
    18: "plant",
}


def _format_semantic_map_view(obs_data: dict[str, Any], view_size: int = VIEW_SIZE) -> str:
    """Format the semantic map into a text representation using dynamic IDs.

    Shows a local view around the player with nearby objects.
    """
    semantic_map = obs_data.get("semantic_map")
    player_position = obs_data.get("player_position", [0, 0])

    if semantic_map is None:
        return "Map view unavailable"

    # Convert to numpy array if needed
    sem_arr = np.asarray(semantic_map)
    if sem_arr.ndim == 1:
        # Reshape flat array to 2D
        side = int(len(sem_arr) ** 0.5)
        sem_arr = sem_arr.reshape(side, side)

    px, py = map(int, player_position)
    half = view_size // 2

    # Choose mapping source
    use_list = isinstance(_ID_TO_NAME, list) and len(_ID_TO_NAME) > 0

    # Build matrix centered at player, then transpose for human-friendly view
    matrix: list[list[str]] = []
    for dy in range(-half, half + 1):
        row_tokens: list[str] = []
        for dx in range(-half, half + 1):
            x, y = px + dx, py + dy
            if not (0 <= x < sem_arr.shape[0] and 0 <= y < sem_arr.shape[1]):
                row_tokens.append("void")
            elif dx == 0 and dy == 0:
                row_tokens.append("player")
            else:
                obj_id = int(sem_arr[x, y])
                if use_list and 0 <= obj_id < len(_ID_TO_NAME):
                    name = _ID_TO_NAME[obj_id]  # type: ignore[index]
                else:
                    name = _FALLBACK_ID_TO_NAME.get(obj_id, str(obj_id))
                row_tokens.append(name)
        matrix.append(row_tokens)

    transposed = list(zip(*matrix, strict=False))
    grid_rows: list[str] = [" ".join(row) for row in transposed]
    return (
        "\nLocal Map View (" + str(view_size) + "x" + str(view_size) + "):\n" + "\n".join(grid_rows)
    )
