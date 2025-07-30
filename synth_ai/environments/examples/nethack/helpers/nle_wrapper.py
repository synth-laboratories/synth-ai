"""NLE integration wrapper for NetHack environment."""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np

try:
    import nle
    from nle import nethack
except ImportError as e:
    raise ImportError(
        "NLE (NetHack Learning Environment) is required but not installed. "
        "Please install it with: pip install nle"
    ) from e


class NLEWrapper:
    """Wrapper around NLE (NetHack Learning Environment) for synth-env integration."""

    def __init__(self, character_role: str = "mon", observation_keys: Optional[List[str]] = None):
        """Initialize NLE wrapper.

        Args:
            character_role: Character class (mon, val, wiz, etc.)
            observation_keys: Which observations to include
        """
        self.character_role = self._convert_role_name(character_role)

        # Default observation keys
        if observation_keys is None:
            observation_keys = [
                "glyphs",
                "chars",
                "colors",
                "specials",
                "blstats",
                "message",
                "inv_glyphs",
                "inv_letters",
                "inv_oclasses",
                "inv_strs",
                "tty_chars",
                "tty_colors",
                "tty_cursor",
            ]

        # Create NLE environment
        self.env = nle.env.NLE(character=self.character_role, observation_keys=observation_keys)

        # Build action mapping
        self._build_action_mapping()

        # Track last observation
        self.last_obs = None
        self.last_reward = 0.0
        self.last_done = False
        self.last_info = {}

    def _convert_role_name(self, role: str) -> str:
        """Convert full role names to NLE abbreviations."""
        role_map = {
            "tourist": "tou",
            "knight": "kni",
            "wizard": "wiz",
            "barbarian": "bar",
            "ranger": "ran",
            "priest": "pri",
            "monk": "mon",
            "rogue": "rog",
            "valkyrie": "val",
            "samurai": "sam",
            "archeologist": "arc",
            "healer": "hea",
            "caveman": "cav",
        }
        return role_map.get(role.lower(), "mon")  # Default to monk

    def _build_action_mapping(self):
        """Build mapping from action names to NLE action indices."""
        self.action_map = {}
        self.index_to_action = {}

        # Map each action in env.actions to a name
        for i, action in enumerate(self.env.actions):
            # Compass directions
            if action == nethack.CompassDirection.N:
                name = "north"
            elif action == nethack.CompassDirection.S:
                name = "south"
            elif action == nethack.CompassDirection.E:
                name = "east"
            elif action == nethack.CompassDirection.W:
                name = "west"
            elif action == nethack.CompassDirection.NE:
                name = "northeast"
            elif action == nethack.CompassDirection.NW:
                name = "northwest"
            elif action == nethack.CompassDirection.SE:
                name = "southeast"
            elif action == nethack.CompassDirection.SW:
                name = "southwest"

            # Misc directions
            elif action == nethack.MiscDirection.UP:
                name = "up"
            elif action == nethack.MiscDirection.DOWN:
                name = "down"
            elif action == nethack.MiscDirection.WAIT:
                name = "wait"

            # Commands
            elif action == nethack.Command.SEARCH:
                name = "search"
            elif action == nethack.Command.INVENTORY:
                name = "inventory"
            elif action == nethack.Command.LOOK:
                name = "look"
            elif action == nethack.Command.OPEN:
                name = "open"
            elif action == nethack.Command.CLOSE:
                name = "close"
            elif action == nethack.Command.KICK:
                name = "kick"
            elif action == nethack.Command.PICKUP:
                name = "pickup"
            elif action == nethack.Command.DROP:
                name = "drop"
            elif action == nethack.Command.EAT:
                name = "eat"
            elif action == nethack.Command.WIELD:
                name = "wield"
            elif action == nethack.Command.WEAR:
                name = "wear"
            elif action == nethack.Command.TAKEOFF:
                name = "takeoff"
            elif action == nethack.Command.PUTON:
                name = "puton"
            elif action == nethack.Command.REMOVE:
                name = "remove"
            elif action == nethack.Command.READ:
                name = "read"
            elif action == nethack.Command.QUAFF:
                name = "quaff"
            elif action == nethack.Command.ZAP:
                name = "zap"
            elif action == nethack.Command.THROW:
                name = "throw"
            elif action == nethack.Command.FIRE:
                name = "fire"
            elif action == nethack.Command.APPLY:
                name = "apply"
            elif action == nethack.Command.PRAY:
                name = "pray"
            elif action == nethack.Command.SAVE:
                name = "save"
            elif action == nethack.Command.QUIT:
                name = "quit"
            elif action == nethack.Command.ESC:
                name = "escape"
            elif action == nethack.Command.PAY:
                name = "pay"
            elif action == nethack.Command.LOOT:
                name = "loot"
            elif action == nethack.Command.ENHANCE:
                name = "enhance"
            elif action == nethack.Command.FORCE:
                name = "force"
            elif action == nethack.Command.INVOKE:
                name = "invoke"
            elif action == nethack.Command.OFFER:
                name = "offer"
            elif action == nethack.Command.RUB:
                name = "rub"
            elif action == nethack.Command.SIT:
                name = "sit"
            elif action == nethack.Command.TURN:
                name = "turn"
            elif action == nethack.Command.UNTRAP:
                name = "untrap"
            elif action == nethack.Command.WIPE:
                name = "wipe"
            elif action == nethack.Command.ENGRAVE:
                name = "engrave"
            elif action == nethack.Command.JUMP:
                name = "jump"
            elif action == nethack.Command.CHAT:
                name = "chat"
            elif action == nethack.Command.DIP:
                name = "dip"
            elif action == nethack.Command.RIDE:
                name = "ride"
            elif action == nethack.Command.TIP:
                name = "tip"

            # Special/Misc
            elif action == nethack.MiscAction.MORE:
                name = "more"

            # Text characters (for menu selection)
            elif hasattr(action, "value") and 97 <= action.value <= 122:  # a-z
                name = chr(action.value)
            elif hasattr(action, "value") and 65 <= action.value <= 90:  # A-Z
                name = chr(action.value)
            elif hasattr(action, "value") and 48 <= action.value <= 57:  # 0-9
                name = chr(action.value)
            elif action == nethack.TextCharacters.SPACE:
                name = "space"
            elif action == nethack.TextCharacters.APOS:
                name = "'"
            elif action == nethack.TextCharacters.QUOTE:
                name = '"'
            else:
                # Skip unmapped actions
                continue

            self.action_map[name] = i
            self.index_to_action[i] = name

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset the NLE environment."""
        if seed is not None:
            self.env.seed(seed)

        self.last_obs = self.env.reset()
        self.last_reward = 0.0
        self.last_done = False
        self.last_info = {}

        return self._process_observation(self.last_obs)

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Action name (e.g., "north", "pickup", "a" for menu)

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Handle compound actions like "open west"
        if " " in action:
            # Split compound action
            parts = action.split(" ", 1)
            if len(parts) == 2 and parts[0] in self.action_map and parts[1] in self.action_map:
                # Execute both actions in sequence
                print(f"Splitting compound action '{action}' into '{parts[0]}' then '{parts[1]}'")
                # First action
                action_idx = self.action_map[parts[0]]
                self.env.step(action_idx)
                # Second action (the direction)
                action = parts[1]
            else:
                raise ValueError(f"Invalid compound action: {action}. Use separate actions.")

        # Get action index
        if action not in self.action_map:
            # Special handling for menu letters that might not be in action_map
            if len(action) == 1 and action.isalpha():
                # Try to find ASCII value for single letter
                ascii_val = ord(action)
                # Look for an action with this ASCII value
                for i, act in enumerate(self.env.actions):
                    if hasattr(act, "value") and act.value == ascii_val:
                        print(f"Found menu letter '{action}' at action index {i}")
                        action_idx = i
                        break
                else:
                    raise ValueError(
                        f"Unknown action: {action}. Valid actions: {list(self.action_map.keys())}"
                    )
            else:
                raise ValueError(
                    f"Unknown action: {action}. Valid actions: {list(self.action_map.keys())}"
                )
        else:
            action_idx = self.action_map[action]

        # Take step
        self.last_obs, self.last_reward, self.last_done, self.last_info = self.env.step(action_idx)

        # Process observation
        processed_obs = self._process_observation(self.last_obs)

        return processed_obs, self.last_reward, self.last_done, self.last_info

    def _process_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Process NLE observation into a more usable format."""
        processed = {}

        # Decode message
        if "message" in obs:
            processed["message"] = obs["message"].tobytes().decode("ascii").strip()

        # Get ASCII map
        if "chars" in obs:
            processed["ascii_chars"] = obs["chars"]
            # Convert to string map
            lines = []
            for row in obs["chars"]:
                line = "".join(chr(c) for c in row)
                lines.append(line)
            processed["ascii_map"] = "\n".join(lines)

        # Get player stats from blstats
        if "blstats" in obs:
            blstats = obs["blstats"]
            processed["player_stats"] = {
                "x": int(blstats[0]),
                "y": int(blstats[1]),
                "strength": int(blstats[2]),
                "strength_pct": int(blstats[3]),
                "dexterity": int(blstats[4]),
                "constitution": int(blstats[5]),
                "intelligence": int(blstats[6]),
                "wisdom": int(blstats[7]),
                "charisma": int(blstats[8]),
                "score": int(blstats[9]),
                "hp": int(blstats[10]),
                "max_hp": int(blstats[11]),
                "depth": int(blstats[12]),
                "gold": int(blstats[13]),
                "energy": int(blstats[14]),
                "max_energy": int(blstats[15]),
                "ac": int(blstats[16]),
                "monster_level": int(blstats[17]),
                "experience_level": int(blstats[18]),
                "experience_points": int(blstats[19]),
                "time": int(blstats[20]),
                "hunger_state": int(blstats[21]),
                "carrying_capacity": int(blstats[22]),
                "dungeon_number": int(blstats[23]),
                "level_number": int(blstats[24]),
            }

        # Get inventory
        if "inv_strs" in obs:
            inv_items = []
            for i, inv_str in enumerate(obs["inv_strs"]):
                if inv_str[0] != 0:  # Non-empty slot
                    item_str = inv_str.tobytes().decode("ascii").strip("\x00")
                    if item_str:
                        letter = chr(obs["inv_letters"][i]) if "inv_letters" in obs else "?"
                        inv_items.append({"letter": letter, "description": item_str})
            processed["inventory"] = inv_items

        # Check if in menu
        if "tty_chars" in obs:
            tty_text = []
            for row in obs["tty_chars"][:5]:  # Check first 5 rows
                line = "".join(chr(c) if 32 <= c <= 126 else " " for c in row).strip()
                if line:
                    tty_text.append(line)

            # Simple menu detection
            processed["in_menu"] = any(
                keyword in " ".join(tty_text).lower()
                for keyword in [
                    "pick up",
                    "drop",
                    "wear",
                    "take off",
                    "what do you want",
                ]
            )
            processed["menu_text"] = tty_text

        # TTY cursor position
        if "tty_cursor" in obs:
            processed["cursor"] = (int(obs["tty_cursor"][1]), int(obs["tty_cursor"][0]))

        # Raw observations for advanced use
        processed["_raw"] = obs

        return processed

    def get_valid_actions(self) -> List[str]:
        """Get list of valid action names."""
        return list(self.action_map.keys())

    def get_state(self) -> bytes:
        """Get the current NLE state for serialization."""
        # NLE environments have a clone_state method that returns the full state
        if hasattr(self.env, "clone_state"):
            return self.env.clone_state()
        elif hasattr(self.env, "clone_full_state"):
            return self.env.clone_full_state()
        else:
            raise RuntimeError("NLE environment does not support state cloning")

    def set_state(self, state: bytes):
        """Restore NLE state from serialized data."""
        # NLE environments have a restore_state method
        if hasattr(self.env, "restore_state"):
            self.env.restore_state(state)
        elif hasattr(self.env, "restore_full_state"):
            self.env.restore_full_state(state)
        else:
            raise RuntimeError("NLE environment does not support state restoration")

    def close(self):
        """Close the NLE environment."""
        self.env.close()
