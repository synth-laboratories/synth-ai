from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from synth_ai.environments.environment.rewards.core import RewardStack
from synth_ai.environments.reproducibility.core import IReproducibleEngine

# Import logging configuration first to suppress JAX debug messages
from synth_ai.environments.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from synth_ai.environments.tasks.core import TaskInstance

from .engine_helpers.reward_components import (
    BadgeRewardComponent,
    BattleVictoryComponent,
    LevelUpComponent,
    MapTransitionComponent,
    StepPenaltyComponent,
    XPGainComponent,
)
from .engine_helpers.state_extraction import extract_game_state

try:
    from pyboy import PyBoy
    from pyboy.pyboy import WindowEvent

    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    PyBoy = None
    WindowEvent = None

if not PYBOY_AVAILABLE:

    class WindowEvent:
        PRESS_BUTTON_A = 0
        PRESS_BUTTON_B = 1
        PRESS_ARROW_UP = 2
        PRESS_ARROW_DOWN = 3
        PRESS_ARROW_LEFT = 4
        PRESS_ARROW_RIGHT = 5
        PRESS_BUTTON_START = 6
        PRESS_BUTTON_SELECT = 7
        RELEASE_BUTTON_A = 8
        RELEASE_BUTTON_B = 9
        RELEASE_ARROW_UP = 10
        RELEASE_ARROW_DOWN = 11
        RELEASE_ARROW_LEFT = 12
        RELEASE_ARROW_RIGHT = 13
        RELEASE_BUTTON_START = 14
        RELEASE_BUTTON_SELECT = 15


# Game Boy button mappings - PyBoy uses string names
BUTTON_MAP = {
    "A": "a",
    "B": "b",
    "UP": "up",
    "DOWN": "down",
    "LEFT": "left",
    "RIGHT": "right",
    "START": "start",
    "SELECT": "select",
}


@dataclass
class PokemonData:
    """Detailed Pokemon information"""

    species_id: int
    level: int
    hp_current: int
    hp_max: int
    xp: int
    hp_percentage: float
    # TODO: Add when memory addresses are available
    # attack: int = 0
    # defense: int = 0
    # speed: int = 0
    # special: int = 0
    # status_conditions: List[str] = None
    # moves: List[str] = None
    # nickname: str = ""


@dataclass
class InventoryItem:
    """Inventory item information"""

    item_id: int
    quantity: int
    # TODO: Add when we have item name mapping
    # name: str = ""
    # category: str = ""


@dataclass
class GameWorldState:
    """Current world/map state information"""

    map_id: int
    player_x: int
    player_y: int
    # TODO: Add when available
    # map_name: str = ""
    # map_type: str = ""  # town, route, building, dungeon
    # available_services: List[str] = None  # Pokemon Center, Pokemart, Gym, etc.
    # npcs_nearby: List[str] = None
    # items_on_ground: List[str] = None
    # wild_encounters_available: bool = False


@dataclass
class GameSystemState:
    """Current game system state (menus, battles, etc.)"""

    in_battle: bool
    battle_outcome: int
    menu_state: int
    text_box_active: bool
    warp_flag: int
    # TODO: Add when available
    # current_menu_type: str = ""
    # dialogue_speaker: str = ""
    # available_actions: List[str] = None


@dataclass
class PlayerProgressState:
    """Player progression and achievements"""

    badges: int
    badge_count: int
    money: int
    step_count: int
    # TODO: Add when available
    # pokedex_seen: int = 0
    # pokedex_caught: int = 0
    # story_flags: List[str] = None
    # time_played: str = "00:00"


@dataclass
class PokemonRedPublicState:
    """Comprehensive Pokemon Red game state for text-based AI interaction

    This structure provides rich, semantic game information to eliminate
    the need for visual processing and enable strategic decision making.
    Based on requirements from text_port.txt.
    """

    # Core game world state
    world: GameWorldState

    # Player progress and achievements
    progress: PlayerProgressState

    # Pokemon party information (up to 6 Pokemon)
    party: List[PokemonData]

    # Inventory and items
    inventory: List[InventoryItem]

    # Current game system state
    system: GameSystemState

    # Error information
    error_info: Optional[str] = None

    # Legacy compatibility fields (for existing code)
    @property
    def map_id(self) -> int:
        return self.world.map_id

    @property
    def player_x(self) -> int:
        return self.world.player_x

    @property
    def player_y(self) -> int:
        return self.world.player_y

    @property
    def badges(self) -> int:
        return self.progress.badges

    @property
    def in_battle(self) -> bool:
        return self.system.in_battle

    @property
    def party_level(self) -> int:
        return self.party[0].level if self.party else 0

    @property
    def party_hp_current(self) -> int:
        return self.party[0].hp_current if self.party else 0

    @property
    def party_hp_max(self) -> int:
        return self.party[0].hp_max if self.party else 0

    @property
    def party_xp(self) -> int:
        return self.party[0].xp if self.party else 0

    @property
    def step_count(self) -> int:
        return self.progress.step_count


@dataclass
class PokemonRedPrivateState:
    reward_last_step: float
    total_reward: float
    terminated: bool
    truncated: bool
    step_count: int


class PokemonRedEngineSnapshot(StatefulEngineSnapshot):
    def __init__(self, state_data: Dict[str, Any], total_reward: float, step_count: int):
        self.state_data = state_data
        self.total_reward = total_reward
        self.step_count = step_count

    def model_dump(self) -> Dict[str, Any]:
        return {
            "state_data": self.state_data,
            "total_reward": self.total_reward,
            "step_count": self.step_count,
        }


class PokemonRedEngine(StatefulEngine, IReproducibleEngine):
    """Pokemon Red game engine with dense reward tracking"""

    def __init__(self, task_instance: TaskInstance, skip_rom_check: bool = False):
        self.task_instance = task_instance

        # Initialize PyBoy emulator
        if not skip_rom_check:
            if not PYBOY_AVAILABLE:
                raise ImportError("PyBoy is required but not installed. Run: uv add pyboy")

            rom_path = self._get_rom_path()
            if not rom_path.exists():
                raise FileNotFoundError(
                    f"Pokemon Red ROM not found at {rom_path}. Please see README.md for setup instructions."
                )

            self.emulator = PyBoy(str(rom_path), window="null")

            # Load the working init state to get the game into a playable state
            self._load_init_state()
        else:
            # For testing purposes, use None emulator
            self.emulator = None

        # Initialize reward stack with dense components
        self.reward_stack = RewardStack(
            components=[
                BadgeRewardComponent(),
                MapTransitionComponent(),
                BattleVictoryComponent(),
                LevelUpComponent(),
                XPGainComponent(),
                StepPenaltyComponent(),
            ]
        )

        self._total_reward = 0.0
        self._step_count = 0
        self._previous_state: Optional[Dict[str, Any]] = None

    def _get_rom_path(self) -> Path:
        """Get path to Pokemon Red ROM file"""
        # Check several possible locations
        possible_paths = [
            Path(__file__).parent / "roms" / "pokemon_red.gb",
            Path(__file__).parent / "roms" / "PokemonRed.gb",
            Path(__file__).parent / "vendor" / "pokemon_red.gb",
            Path.home() / "Games" / "pokemon_red.gb",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # Return default expected location
        return Path(__file__).parent / "roms" / "pokemon_red.gb"

    def _load_init_state(self) -> None:
        """Load the initial save state to get the game into a playable state"""
        init_state_paths = [
            Path(__file__).parent / "roms" / "working_init.state",
            Path(__file__).parent / "roms" / "init.state",
        ]

        for state_path in init_state_paths:
            if state_path.exists():
                try:
                    with open(state_path, "rb") as f:
                        self.emulator.load_state(f)
                    logging.info(f"Loaded init state from: {state_path}")
                    return
                except Exception as e:
                    logging.warning(f"Failed to load init state from {state_path}: {e}")
                    continue

        # If no init state found, try to use PyBoy's game wrapper
        logging.warning("No init state found, trying PyBoy game wrapper...")
        try:
            if hasattr(self.emulator.game_wrapper, "start_game"):
                self.emulator.game_wrapper.start_game()
                logging.info("Used PyBoy game wrapper start_game()")
            else:
                logging.warning("PyBoy game wrapper doesn't have start_game method")
        except Exception as e:
            logging.warning(f"PyBoy game wrapper start_game failed: {e}")

    def _extract_current_state(self) -> Dict[str, Any]:
        """Extract current game state from emulator memory"""
        if self.emulator is None:
            # Return mock state for testing
            return {
                "map_id": 1,
                "player_x": 10,
                "player_y": 10,
                "badges": 0,
                "in_battle": False,
                "party_level": 5,
                "party_hp_current": 25,
                "party_hp_max": 25,
                "party_xp": 100,
            }

        # Get memory from PyBoy
        memory = self.emulator.memory
        return extract_game_state(memory)

    def _press_button(self, button: str, frames: int = 1):
        """Press a Game Boy button for specified frames"""
        if button not in BUTTON_MAP:
            raise ValueError(f"Invalid button: {button}. Valid buttons: {list(BUTTON_MAP.keys())}")

        button_name = BUTTON_MAP[button]

        if self.emulator is None:
            return  # Skip for testing

        # Press button
        self.emulator.button_press(button_name)

        # Hold for specified frames
        for _ in range(frames):
            self.emulator.tick()

        # Release button
        self.emulator.button_release(button_name)

        # Let release take effect
        self.emulator.tick()

    def _press_button_with_retry(
        self, button: str, frames: int = 1, max_attempts: int = 10
    ) -> bool:
        """
        Press a button with automatic retry for movement commands.

        For movement buttons (UP, DOWN, LEFT, RIGHT), this will automatically
        repeat the button press until movement occurs or max_attempts is reached.

        For other buttons (A, B, START, SELECT), this behaves like _press_button.

        Note: Previous menu-closing logic for 'B' button was removed because
        investigation showed that menu_state memory address represents
        "selected menu item index" not "menu is open", leading to false positives.

        Returns True if the expected state change occurred or always True for non-retryable buttons.
        """
        movement_buttons = {"UP", "DOWN", "LEFT", "RIGHT"}

        # Handle movement buttons with retry until position changes
        if button in movement_buttons:
            if self.emulator is None:
                return True  # Skip for testing

            # Get initial position
            try:
                initial_state = self._extract_current_state()
                initial_position = (
                    initial_state.get("player_x", 0),
                    initial_state.get("player_y", 0),
                )
                initial_map = initial_state.get("map_id", 0)
            except Exception as e:
                logging.warning(f"Could not extract initial state for movement retry: {e}")
                # Fall back to single press
                self._press_button(button, frames)
                return True

            for attempt in range(max_attempts):
                # Press the button
                self._press_button(button, frames)

                # Check if position changed
                try:
                    new_state = self._extract_current_state()
                    new_position = (
                        new_state.get("player_x", 0),
                        new_state.get("player_y", 0),
                    )
                    new_map = new_state.get("map_id", 0)

                    # Movement successful if position or map changed
                    if new_position != initial_position or new_map != initial_map:
                        logging.debug(
                            f"Movement successful after {attempt + 1} attempts: {initial_position} -> {new_position}"
                        )
                        return True

                except Exception as e:
                    logging.warning(
                        f"Could not extract state during movement retry attempt {attempt + 1}: {e}"
                    )
                    continue

            # If we get here, movement didn't occur after max_attempts
            logging.warning(
                f"Movement button {button} pressed {max_attempts} times but no position change detected"
            )
            return False

        else:
            # For all other buttons (A, B, START, SELECT), just press once
            # No retry logic needed - let the game handle the response naturally
            self._press_button(button, frames)
            return True

    def _create_states(
        self, reward: float, terminated: bool = False
    ) -> tuple[PokemonRedPrivateState, PokemonRedPublicState]:
        """Create private and public state objects"""
        try:
            current_state = self._extract_current_state()
        except Exception as e:
            logging.error(f"Error extracting game state: {e}")
            # Provide default state values
            current_state = {
                "map_id": 0,
                "player_x": 0,
                "player_y": 0,
                "badges": 0,
                "in_battle": False,
                "party_pokemon": [],
                "inventory_items": [],
                "money": 0,
                "battle_outcome": 0,
                "menu_state": 0,
                "text_box_active": False,
                "warp_flag": 0,
            }

        try:
            private_state = PokemonRedPrivateState(
                reward_last_step=reward,
                total_reward=self._total_reward,
                terminated=terminated,
                truncated=False,
                step_count=self._step_count,
            )

            # Extract comprehensive game state data
            map_id = int(current_state.get("map_id", 0))
            player_x = int(current_state.get("player_x", 0))
            player_y = int(current_state.get("player_y", 0))
            badges = int(current_state.get("badges", 0))
            money = int(current_state.get("money", 0))

            # Count badges for badge_count field
            badge_count = bin(badges).count("1")

            # Create Pokemon party from detailed party data
            party_pokemon_data = current_state.get("party_pokemon", [])
            party = []
            for pokemon_data in party_pokemon_data:
                try:
                    pokemon = PokemonData(
                        species_id=int(pokemon_data.get("species_id", 0)),
                        level=int(pokemon_data.get("level", 1)),
                        hp_current=int(pokemon_data.get("hp_current", 1)),
                        hp_max=int(pokemon_data.get("hp_max", 1)),
                        xp=int(pokemon_data.get("xp", 0)),
                        hp_percentage=float(pokemon_data.get("hp_percentage", 100.0)),
                    )
                    party.append(pokemon)
                except (TypeError, ValueError) as e:
                    logging.warning(f"Error creating Pokemon data: {e}")
                    continue

            # Create inventory from detailed inventory data
            inventory_data = current_state.get("inventory_items", [])
            inventory = []
            for item_data in inventory_data:
                try:
                    item = InventoryItem(
                        item_id=int(item_data.get("item_id", 0)),
                        quantity=int(item_data.get("quantity", 0)),
                    )
                    inventory.append(item)
                except (TypeError, ValueError) as e:
                    logging.warning(f"Error creating inventory item: {e}")
                    continue

            # Create comprehensive public state
            public_state = PokemonRedPublicState(
                world=GameWorldState(map_id=map_id, player_x=player_x, player_y=player_y),
                progress=PlayerProgressState(
                    badges=badges,
                    badge_count=badge_count,
                    money=money,
                    step_count=self._step_count,
                ),
                party=party,
                inventory=inventory,
                system=GameSystemState(
                    in_battle=bool(current_state.get("in_battle", False)),
                    battle_outcome=int(current_state.get("battle_outcome", 0)),
                    menu_state=int(current_state.get("menu_state", 0)),
                    text_box_active=bool(current_state.get("text_box_active", False)),
                    warp_flag=int(current_state.get("warp_flag", 0)),
                ),
            )

        except (TypeError, ValueError) as e:
            logging.error(f"Error creating states with data {current_state}: {e}")
            # Create minimal safe states
            private_state = PokemonRedPrivateState(
                reward_last_step=0.0,
                total_reward=0.0,
                terminated=True,
                truncated=False,
                step_count=self._step_count,
            )
            public_state = PokemonRedPublicState(
                world=GameWorldState(map_id=0, player_x=0, player_y=0),
                progress=PlayerProgressState(
                    badges=0, badge_count=0, money=0, step_count=self._step_count
                ),
                party=[],
                inventory=[],
                system=GameSystemState(
                    in_battle=False,
                    battle_outcome=0,
                    menu_state=0,
                    text_box_active=False,
                    warp_flag=0,
                ),
                error_info=f"State creation error: {e}",
            )

        return private_state, public_state

    async def _reset_engine(
        self, *, seed: Optional[int] = None
    ) -> tuple[PokemonRedPrivateState, PokemonRedPublicState]:
        """Reset the Pokemon Red engine to initial state"""
        # Load initial save state if provided
        if (
            hasattr(self.task_instance, "initial_engine_snapshot")
            and self.task_instance.initial_engine_snapshot
        ):
            snapshot_path = self.task_instance.initial_engine_snapshot
            if isinstance(snapshot_path, Path) and snapshot_path.exists():
                self.emulator.load_state(str(snapshot_path))

        self._total_reward = 0.0
        self._step_count = 0
        self._previous_state = self._extract_current_state()

        return self._create_states(reward=0.0)

    async def _step_engine(
        self, action: Dict[str, Any]
    ) -> tuple[PokemonRedPrivateState, PokemonRedPublicState]:
        """Execute one step in the Pokemon Red environment"""
        try:
            # Extract previous state for reward calculation
            prev_state = self._previous_state or self._extract_current_state()

            # Execute action (button press)
            button = action.get("button", "A")
            frames = action.get("frames", 1)

            self._press_button_with_retry(button, frames)

            self._step_count += 1

            # Extract new state
            current_state = self._extract_current_state()

            # Calculate reward using reward stack
            try:
                reward = await self.reward_stack.step_reward(
                    state=current_state,
                    action={
                        "prev_badges": int(prev_state.get("badges", 0)),
                        "prev_map_id": int(prev_state.get("map_id", 0)),
                        "prev_in_battle": bool(prev_state.get("in_battle", False)),
                        "prev_party_level": int(prev_state.get("party_level", 0)),
                        "prev_party_xp": int(prev_state.get("party_xp", 0)),
                    },
                )
            except Exception as e:
                logging.error(f"Error calculating reward: {e}")
                reward = -0.01  # Small penalty for error

            self._total_reward += reward
            self._previous_state = current_state

            # Check termination condition (example: got Boulder Badge)
            try:
                badges = current_state.get("badges", 0)
                badges = int(badges) if badges is not None else 0
                terminated = (badges & 0x01) != 0
            except (TypeError, ValueError) as e:
                logging.error(
                    f"Error checking termination condition with badges={current_state.get('badges')}: {e}"
                )
                terminated = False

            return self._create_states(reward=reward, terminated=terminated)

        except Exception as e:
            logging.error(f"Error in step engine: {e}")
            # Still increment step count even on error
            self._step_count += 1
            # Return safe default states
            return self._create_states(reward=-1.0, terminated=True)

    async def _serialize_engine(self) -> PokemonRedEngineSnapshot:
        """Serialize engine state for checkpointing"""
        # Save state to temporary file
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".state")
        temp_file.close()

        if self.emulator is not None:
            with open(temp_file.name, "wb") as f:
                self.emulator.save_state(f)

            # Read state file as bytes for storage
            with open(temp_file.name, "rb") as f:
                state_bytes = f.read()
        else:
            # For testing without emulator
            state_bytes = b"mock_state_data"

        current_state = self._extract_current_state()
        current_state["_save_state_bytes"] = state_bytes

        return PokemonRedEngineSnapshot(
            state_data=current_state,
            total_reward=self._total_reward,
            step_count=self._step_count,
        )

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: PokemonRedEngineSnapshot, task_instance: TaskInstance
    ) -> "PokemonRedEngine":
        """Deserialize engine from checkpoint"""
        engine = cls(task_instance)

        # Restore save state if available
        if "_save_state_bytes" in snapshot.state_data and engine.emulator is not None:
            import io

            state_bytes = snapshot.state_data["_save_state_bytes"]
            state_io = io.BytesIO(state_bytes)
            engine.emulator.load_state(state_io)

        engine._total_reward = snapshot.total_reward
        engine._step_count = snapshot.step_count
        engine._previous_state = {
            k: v for k, v in snapshot.state_data.items() if k != "_save_state_bytes"
        }

        return engine
