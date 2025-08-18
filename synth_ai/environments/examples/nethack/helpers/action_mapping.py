"""Action mapping and validation for NetHack."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ActionCategory:
    """Category of actions with description."""

    name: str
    description: str
    actions: List[str]


# Comprehensive NetHack action mapping
NETHACK_ACTIONS: Dict[str, str] = {
    # Movement actions (8 directions + wait)
    "north": "move north",
    "south": "move south",
    "east": "move east",
    "west": "move west",
    "northeast": "move northeast",
    "northwest": "move northwest",
    "southeast": "move southeast",
    "southwest": "move southwest",
    "wait": "wait/rest for one turn",
    # Movement modifiers
    "run_north": "run north until something interesting",
    "run_south": "run south until something interesting",
    "run_east": "run east until something interesting",
    "run_west": "run west until something interesting",
    "go_up": "go up stairs/ladder",
    "go_down": "go down stairs/ladder",
    # Basic interactions
    "search": "search for secret doors/traps",
    "open": "open a door",
    "close": "close a door",
    "kick": "kick something",
    "force": "force a lock",
    "untrap": "untrap something",
    # Inventory and items
    "inventory": "check inventory",
    "pickup": "pick up items",
    "drop": "drop items",
    "dropall": "drop all items",
    "wear": "wear armor/accessories",
    "take_off": "take off armor/accessories",
    "wield": "wield a weapon",
    "unwield": "unwield current weapon",
    "quiver": "ready ammunition",
    "put_on": "put on accessories",
    "remove": "remove accessories",
    # Using items
    "eat": "eat food",
    "drink": "drink a potion",
    "read": "read a scroll/spellbook",
    "zap": "zap a wand",
    "apply": "apply/use a tool",
    "invoke": "invoke an artifact",
    "rub": "rub a lamp/stone",
    "throw": "throw an item",
    "fire": "fire from quiver",
    # Magic
    "cast": "cast a spell",
    "pray": "pray to your deity",
    "offer": "offer sacrifice",
    "turn_undead": "turn undead (priest ability)",
    # Information (NOTE: These don't consume turns!)
    "look": "look around (FREE ACTION - doesn't advance time)",
    "farlook": "look at specific location (FREE ACTION)",
    "whatis": "identify map symbol (FREE ACTION)",
    "identify": "identify inventory item (FREE ACTION)",
    "discoveries": "list discoveries (FREE ACTION)",
    "conduct": "check conduct (FREE ACTION)",
    "attributes": "check attributes (FREE ACTION)",
    # Character actions
    "enhance": "enhance skills",
    "sit": "sit down",
    "pay": "pay shopkeeper",
    "chat": "talk to someone",
    "loot": "loot a container",
    "engrave": "write on the ground",
    "monster_ability": "use monster ability",
    # Game commands
    "save": "save the game",
    "quit": "quit the game",
    "help": "show help",
    "version": "show version",
    "history": "show message history",
    "name": "name an item/monster",
    "call": "call item type",
    "adjust": "adjust inventory letters",
    # Special responses for prompts/menus
    "yes": "answer yes",
    "no": "answer no",
    "all": "select all",
    "none": "select none",
    "menu_next": "next menu page",
    "menu_previous": "previous menu page",
    "escape": "cancel/escape",
}

# Single character responses for menu selections
MENU_ACTIONS: Dict[str, str] = {
    chr(i): f"select option {chr(i)}" for i in range(ord("a"), ord("z") + 1)
}
MENU_ACTIONS.update({chr(i): f"select option {chr(i)}" for i in range(ord("A"), ord("Z") + 1)})
MENU_ACTIONS.update({str(i): f"select option {i}" for i in range(10)})

# Combine all actions
ALL_ACTIONS = {**NETHACK_ACTIONS, **MENU_ACTIONS}

# Action categories for organization
ACTION_CATEGORIES = [
    ActionCategory(
        name="Movement",
        description="Basic movement and navigation",
        actions=[
            "north",
            "south",
            "east",
            "west",
            "northeast",
            "northwest",
            "southeast",
            "southwest",
            "wait",
            "go_up",
            "go_down",
        ],
    ),
    ActionCategory(
        name="Inventory",
        description="Managing items and equipment",
        actions=[
            "inventory",
            "pickup",
            "drop",
            "wear",
            "wield",
            "eat",
            "drink",
            "read",
            "apply",
            "throw",
        ],
    ),
    ActionCategory(
        name="Combat",
        description="Fighting and defense (attack by moving into monsters!)",
        actions=["fire", "zap", "throw", "kick"],
    ),
    ActionCategory(
        name="Exploration",
        description="Discovering the dungeon",
        actions=["search", "open", "close", "look", "farlook"],
    ),
    ActionCategory(
        name="Magic",
        description="Spells and divine intervention",
        actions=["cast", "pray", "offer", "invoke"],
    ),
    ActionCategory(
        name="Game",
        description="Meta game commands",
        actions=["save", "quit", "help", "inventory"],
    ),
]


def validate_action(action: str, game_state: Optional[Dict] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate if an action is valid given the current game state.

    Args:
        action: The action string to validate
        game_state: Optional game state for context-aware validation

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if action exists
    if action not in ALL_ACTIONS:
        return False, f"Unknown action: {action}. Use 'help' to see available actions."

    # Context-aware validation if game state provided
    if game_state:
        # Check if in menu
        if game_state.get("in_menu", False):
            if action not in MENU_ACTIONS and action not in [
                "escape",
                "menu_next",
                "menu_previous",
            ]:
                return False, "Currently in a menu. Use letter/number keys or 'escape'."

        # Check if game is over
        if game_state.get("terminated", False):
            if action not in ["quit", "save"]:
                return False, "Game is over. You can only 'save' or 'quit'."

        # Check stairs availability
        if action in ["go_up", "go_down"]:
            if not game_state.get("stairs_here", False):
                return False, f"No stairs here to {action.replace('go_', '')}."

    return True, None


def get_action_description(action: str) -> str:
    """Get the description of an action."""
    return ALL_ACTIONS.get(action, "Unknown action")


def get_actions_for_context(game_state: Dict) -> List[str]:
    """Get relevant actions for the current game context."""
    if game_state.get("in_menu", False):
        # In menu - return menu navigation actions
        menu_items = game_state.get("menu_items", [])
        actions = ["escape"]

        # Add letter options based on menu items
        for i, item in enumerate(menu_items):
            if i < 26:
                actions.append(chr(ord("a") + i))

        return actions

    if game_state.get("terminated", False):
        return ["quit", "save"]

    # Normal gameplay - return common actions
    common_actions = [
        "north",
        "south",
        "east",
        "west",
        "search",
        "inventory",
        "pickup",
        "look",
        "wait",
        "open",
        "close",
    ]

    # Add context-specific actions
    if game_state.get("stairs_here", False):
        if game_state.get("stairs_down", False):
            common_actions.append("go_down")
        if game_state.get("stairs_up", False):
            common_actions.append("go_up")

    if game_state.get("items_here", False):
        common_actions.append("pickup")

    if game_state.get("door_here", False):
        if game_state.get("door_open", False):
            common_actions.append("close")
        else:
            common_actions.append("open")

    return common_actions


def convert_action_to_nle(action: str, action_map: Dict[str, int]) -> Optional[int]:
    """
    Convert string action to NLE integer action.

    Args:
        action: String action name
        action_map: Dictionary mapping action names to NLE indices

    Returns:
        NLE action index or None if not found
    """
    # Direct lookup in action map
    if action in action_map:
        return action_map[action]

    # Handle special cases
    if action == "terminate":
        # This is handled at a higher level
        return None

    # Single character actions (menu selections)
    if len(action) == 1 and (action.isalpha() or action.isdigit()):
        if action in action_map:
            return action_map[action]

    return None


def parse_compound_action(action: str) -> List[str]:
    """
    Parse compound actions into individual steps.
    E.g., "go to stairs and descend" -> ["navigate_to_stairs", "go_down"]
    """
    # This could be extended to handle more complex action parsing
    return [action]  # For now, just return the action as-is
