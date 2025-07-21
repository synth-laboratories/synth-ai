"""Observation processing utilities for NetHack."""

from typing import Dict, Any, List, Optional, Tuple
import re


def format_observation_for_llm(observation: Dict[str, Any]) -> str:
    """
    Format NetHack observation for LLM consumption.

    Args:
        observation: Raw observation dictionary

    Returns:
        Formatted string suitable for LLM input
    """
    lines = []

    # Header with turn count and location
    lines.append(f"=== NetHack - Turn {observation.get('turn_count', 0)} ===")
    lines.append(f"Dungeon Level: {observation.get('dungeon_level', 1)}")

    # ASCII map
    if "ascii_map" in observation:
        lines.append("\n--- Dungeon Map ---")
        lines.append(observation["ascii_map"])

    # Game message
    if "message" in observation and observation["message"]:
        # Clean null bytes and trailing whitespace
        message = observation["message"].rstrip("\x00").strip()
        if message:
            lines.append(f"\nMessage: {message}")

    # Character stats
    if "character_stats" in observation:
        stats = observation["character_stats"]
        lines.append("\n--- Character Status ---")
        lines.append(f"HP: {stats.get('hp', 0)}/{stats.get('max_hp', 0)}")
        lines.append(f"Level: {stats.get('level', 1)} (Exp: {stats.get('experience', 0)})")
        lines.append(f"AC: {stats.get('ac', 10)}, Gold: {stats.get('gold', 0)}")

        # Attributes
        attrs = []
        for attr in [
            "strength",
            "dexterity",
            "constitution",
            "intelligence",
            "wisdom",
            "charisma",
        ]:
            if attr in stats:
                attrs.append(f"{attr[:3].upper()}:{stats[attr]}")
        if attrs:
            lines.append(f"Attributes: {' '.join(attrs)}")

    # Inventory summary
    if "inventory_summary" in observation:
        lines.append("\n--- Inventory ---")
        lines.append(observation["inventory_summary"])

    # Menu items if in menu
    if observation.get("in_menu", False) and "menu_items" in observation:
        lines.append("\n--- Menu Options ---")
        for i, item in enumerate(observation["menu_items"]):
            if i < 26:
                lines.append(f"{chr(ord('a') + i)}) {item}")
            else:
                lines.append(f"{i - 26}) {item}")

    # Score and rewards
    lines.append("\n--- Progress ---")
    lines.append(f"Score: {observation.get('score', 0)}")
    lines.append(f"Total Reward: {observation.get('total_reward', 0.0):.2f}")
    lines.append(f"Last Reward: {observation.get('reward_last', 0.0):.2f}")

    # Termination status
    if observation.get("terminated", False):
        lines.append("\n*** GAME OVER ***")

    return "\n".join(lines)


def parse_ascii_map(ascii_map: str) -> Dict[str, Any]:
    """
    Parse ASCII map to extract key information.

    Args:
        ascii_map: Raw ASCII map string

    Returns:
        Dictionary with extracted map information
    """
    lines = ascii_map.strip().split("\n")

    map_info = {
        "width": 0,
        "height": len(lines),
        "player_position": None,
        "stairs_positions": [],
        "door_positions": [],
        "item_positions": [],
        "monster_positions": [],
        "wall_positions": [],
        "floor_positions": [],
    }

    if lines:
        map_info["width"] = max(len(line) for line in lines)

    # Common NetHack ASCII symbols
    symbols = {
        "@": "player",
        "<": "stairs_up",
        ">": "stairs_down",
        "+": "closed_door",
        "-": "open_door_horizontal",
        "|": "open_door_vertical",
        ".": "floor",
        "#": "corridor",
        " ": "wall",
        "$": "gold",
        "*": "gem",
        "!": "potion",
        "?": "scroll",
        "/": "wand",
        "=": "ring",
        '"': "amulet",
        "[": "armor",
        ")": "weapon",
        "(": "tool",
        "%": "food",
        "^": "trap",
    }

    # Monster symbols (letters)
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            pos = (x, y)

            if char == "@":
                map_info["player_position"] = pos
            elif char == "<":
                map_info["stairs_positions"].append(("up", pos))
            elif char == ">":
                map_info["stairs_positions"].append(("down", pos))
            elif char in ["+", "-", "|"]:
                map_info["door_positions"].append((char, pos))
            elif char in ["$", "*", "!", "?", "/", "=", '"', "[", ")", "(", "%"]:
                map_info["item_positions"].append((char, pos))
            elif char.isalpha() and char != "@":
                # Store both the character and position for monster identification
                map_info["monster_positions"].append((char, pos))
            elif char == ".":
                map_info["floor_positions"].append(pos)
            elif char in ["#", " "]:
                map_info["wall_positions"].append(pos)

    return map_info


def extract_game_context(observation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract high-level game context from observation.

    Args:
        observation: Raw observation dictionary

    Returns:
        Dictionary with game context information
    """
    context = {
        "in_combat": False,
        "in_shop": False,
        "at_stairs": False,
        "items_nearby": False,
        "doors_nearby": False,
        "low_health": False,
        "hungry": False,
        "encumbered": False,
        "in_menu": observation.get("in_menu", False),
        "game_over": observation.get("terminated", False),
    }

    # Parse map for context
    if "ascii_map" in observation:
        map_info = parse_ascii_map(observation["ascii_map"])

        # Check if player is near important features
        if map_info["player_position"]:
            px, py = map_info["player_position"]

            # Check for nearby monsters (within 2 squares)
            # Exclude pets from combat detection
            pet_symbols = ["f", "d"]  # f = kitten, d = dog
            for monster_char, (mx, my) in map_info["monster_positions"]:
                if abs(mx - px) <= 2 and abs(my - py) <= 2:
                    # Only trigger combat for non-pet monsters
                    if monster_char not in pet_symbols:
                        context["in_combat"] = True
                        break

            # Check for stairs
            for stair_type, (sx, sy) in map_info["stairs_positions"]:
                if sx == px and sy == py:
                    context["at_stairs"] = True
                    context["stairs_type"] = stair_type
                    break

            # Check for nearby items
            for _, (ix, iy) in map_info["item_positions"]:
                if abs(ix - px) <= 1 and abs(iy - py) <= 1:
                    context["items_nearby"] = True
                    break

            # Check for nearby doors
            for _, (dx, dy) in map_info["door_positions"]:
                if abs(dx - px) <= 1 and abs(dy - py) <= 1:
                    context["doors_nearby"] = True
                    break

    # Check health status
    if "character_stats" in observation:
        stats = observation["character_stats"]
        hp = stats.get("hp", 0)
        max_hp = stats.get("max_hp", 1)
        if hp < max_hp * 0.3:
            context["low_health"] = True

    # Check for specific messages
    message = observation.get("message", "").lower()
    if "hungry" in message or "weak" in message:
        context["hungry"] = True
    if "burdened" in message or "stressed" in message:
        context["encumbered"] = True
    if "shop" in message or "shopkeeper" in message:
        context["in_shop"] = True

    return context


def simplify_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a simplified observation for agents that need less detail.

    Args:
        observation: Full observation dictionary

    Returns:
        Simplified observation dictionary
    """
    simplified = {
        "turn": observation.get("turn_count", 0),
        "level": observation.get("dungeon_level", 1),
        "hp": 0,
        "max_hp": 0,
        "message": observation.get("message", ""),
        "terminated": observation.get("terminated", False),
        "reward": observation.get("reward_last", 0.0),
    }

    # Extract HP
    if "character_stats" in observation:
        stats = observation["character_stats"]
        simplified["hp"] = stats.get("hp", 0)
        simplified["max_hp"] = stats.get("max_hp", 0)

    # Extract key map features
    if "ascii_map" in observation:
        map_info = parse_ascii_map(observation["ascii_map"])
        simplified["player_pos"] = map_info["player_position"]
        simplified["monsters_nearby"] = len(map_info["monster_positions"])
        simplified["items_nearby"] = len(map_info["item_positions"])
        simplified["at_stairs"] = any(
            pos == map_info["player_position"] for _, pos in map_info["stairs_positions"]
        )

    return simplified


def extract_inventory_from_message(message: str) -> List[Dict[str, Any]]:
    """
    Extract inventory information from NetHack inventory messages.

    Args:
        message: Inventory message string

    Returns:
        List of inventory items
    """
    items = []

    # Common inventory line patterns
    # Example: "a - a blessed +1 long sword (weapon in hand)"
    # Example: "b - an uncursed food ration"
    pattern = r"^([a-zA-Z])\s*-\s*(.+?)(?:\s*\(([^)]+)\))?$"

    for line in message.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            letter, description, status = match.groups()

            item = {
                "letter": letter,
                "description": description.strip(),
                "status": status.strip() if status else None,
            }

            # Parse quantity
            qty_match = re.match(r"^(\d+)\s+(.+)", description)
            if qty_match:
                item["quantity"] = int(qty_match.group(1))
                item["name"] = qty_match.group(2)
            else:
                item["quantity"] = 1
                item["name"] = description

            # Identify item type
            item["type"] = identify_item_type(description)

            items.append(item)

    return items


def identify_item_type(description: str) -> str:
    """
    Identify the type of an item from its description.

    Args:
        description: Item description string

    Returns:
        Item type string
    """
    desc_lower = description.lower()

    # Weapons
    if any(
        word in desc_lower
        for word in [
            "sword",
            "dagger",
            "spear",
            "axe",
            "mace",
            "bow",
            "arrow",
            "dart",
            "knife",
        ]
    ):
        return "weapon"

    # Armor
    if any(
        word in desc_lower
        for word in [
            "armor",
            "mail",
            "helmet",
            "boots",
            "gloves",
            "shield",
            "cloak",
            "robe",
        ]
    ):
        return "armor"

    # Food
    if any(
        word in desc_lower
        for word in [
            "food",
            "ration",
            "corpse",
            "egg",
            "fruit",
            "meat",
            "candy",
            "cookie",
        ]
    ):
        return "food"

    # Potions
    if "potion" in desc_lower:
        return "potion"

    # Scrolls
    if "scroll" in desc_lower:
        return "scroll"

    # Wands
    if "wand" in desc_lower:
        return "wand"

    # Rings
    if "ring" in desc_lower:
        return "ring"

    # Amulets
    if "amulet" in desc_lower:
        return "amulet"

    # Tools
    if any(
        word in desc_lower
        for word in [
            "pick",
            "key",
            "lamp",
            "candle",
            "bag",
            "sack",
            "horn",
            "whistle",
            "mirror",
        ]
    ):
        return "tool"

    # Gems/stones
    if any(word in desc_lower for word in ["gem", "stone", "rock", "crystal"]):
        return "gem"

    # Gold
    if "gold" in desc_lower:
        return "gold"

    return "unknown"
