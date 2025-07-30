import asyncio
import uuid
import pytest
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Deque, Literal
from pydantic import BaseModel, Field, validator
from collections import deque
from synth_ai.zyk import LM
from synth_ai.zyk.lms.tools.base import BaseTool
from synth_sdk.tracing.decorators import trace_event_async
from synth_sdk.tracing.abstractions import RewardSignal, Dataset, TrainingQuestion
from synth_sdk.tracing.utils import get_system_id

# Monkey patch the zyk cache handler to allow mixed content types (for images)
try:
    from synth_ai.zyk.lms.caching.handler import CacheHandler

    original_validate_messages = CacheHandler._validate_messages

    def patched_validate_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Validate that messages are in the correct format - PATCHED to allow mixed content for images."""
        # Allow mixed content types when images are involved - just check that messages exist
        assert all(isinstance(msg, dict) and "content" in msg for msg in messages), (
            "All messages must be dicts with content"
        )

    CacheHandler._validate_messages = patched_validate_messages
    print("[DEBUG] Successfully monkey patched zyk cache validation to support images")
except Exception as e:
    print(f"[DEBUG] Failed to monkey patch zyk cache validation: {e}")
    # Continue anyway - the assertion might not be hit in all cases

# Pokemon Red specific imports
from synth_ai.environments.examples.red.environment import (
    PokemonRedEnvironment,
    PokemonRedPublicState,
    PokemonRedPrivateState,
)

# Import early game reward components
from synth_ai.environments.examples.red.engine_helpers.reward_library.pallet_town_rewards import (
    LeaveStartingRoomReward,
    TalkToMomReward,
    InteractWithTVReward,
    CheckComputerReward,
    ExitHouseReward,
    ExploreTownReward,
    TalkToNPCsReward,
    OakLabDiscoveryReward,
    AttemptRoute1Reward,
    ChooseStarterPokemonReward,
    DoorInteractionReward,
    ObjectInteractionReward,
    TryAllDirectionsReward,
)
from synth_ai.environments.examples.red.engine_helpers.reward_library.exploration_rewards import (
    NewAreaDiscoveryReward,
    BuildingEntryReward,
)
from synth_ai.environments.examples.red.engine_helpers.reward_library.novelty_rewards import (
    FirstBattleReward,
    FirstPokemonCenterVisitReward,
)

from synth_ai.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_ai.environments.examples.red.taskset import PokemonRedTaskInstance
from synth_ai.environments.tasks.core import Impetus, Intent, TaskInstanceMetadata
from synth_ai.environments.environment.tools import EnvToolCall

# Import screen analysis functions
from synth_ai.environments.examples.red.engine_helpers.screen_analysis import (
    analyze_screen_buffer,
    create_detailed_screen_description,
)

import logging

logging.disable(logging.CRITICAL)


# --- Early Game Reward Manager ---
class EarlyGameRewardManager:
    """Manages early game rewards for Pokemon Red to encourage exploration and progress"""

    def __init__(self):
        # Initialize early game reward components
        self.rewards = [
            # Pallet Town house exploration
            LeaveStartingRoomReward(),
            TalkToMomReward(),
            InteractWithTVReward(),
            CheckComputerReward(),
            ExitHouseReward(),
            # Town and building exploration
            ExploreTownReward(),
            TalkToNPCsReward(),
            NewAreaDiscoveryReward(),
            BuildingEntryReward(),
            # Story progression
            OakLabDiscoveryReward(),
            AttemptRoute1Reward(),
            ChooseStarterPokemonReward(),
            # Basic interactions
            DoorInteractionReward(),
            ObjectInteractionReward(),
            TryAllDirectionsReward(),
            # First time experiences
            FirstBattleReward(),
            FirstPokemonCenterVisitReward(),
        ]

        self.total_reward_earned = 0.0
        self.reward_history = []

    async def calculate_rewards(
        self,
        current_state: Dict[str, Any],
        prev_state: Dict[str, Any],
        action_info: Dict[str, Any],
    ) -> float:
        """Calculate rewards for the current state transition"""
        total_reward = 0.0
        step_rewards = []

        # Create action context with previous state info
        action_context = {
            "prev_map_id": prev_state.get("map_id", -1),
            "prev_player_x": prev_state.get("player_x", -1),
            "prev_player_y": prev_state.get("player_y", -1),
            "prev_text_box_active": prev_state.get("text_box_active", False),
            "prev_in_battle": prev_state.get("in_battle", False),
            "prev_party": prev_state.get("party", []),
            "prev_inventory": prev_state.get("inventory", []),
            "prev_money": prev_state.get("money", 0),
            **action_info,  # Include any additional action info
        }

        # Calculate rewards from each component
        for reward_component in self.rewards:
            try:
                reward = await reward_component.score(current_state, action_context)
                if reward > 0:
                    total_reward += reward
                    step_rewards.append(
                        {
                            "component": reward_component.__class__.__name__,
                            "reward": reward,
                        }
                    )
                    print(f"[REWARD] {reward_component.__class__.__name__}: +{reward:.1f}")
            except Exception as e:
                print(f"[REWARD_ERROR] {reward_component.__class__.__name__}: {e}")
                continue

        if total_reward > 0:
            self.total_reward_earned += total_reward
            self.reward_history.append(
                {
                    "step": current_state.get("step_count", 0),
                    "total_reward": total_reward,
                    "components": step_rewards,
                }
            )
            print(
                f"[REWARD_TOTAL] Step {current_state.get('step_count', 0)}: +{total_reward:.1f} (Total: {self.total_reward_earned:.1f})"
            )

        return total_reward


# --- Helper function to format observation for LLM ---
def format_obs_for_llm_from_states(
    pub: PokemonRedPublicState,
    priv: PokemonRedPrivateState,
    screen_analysis: dict = None,
    mode: str = "state_and_screen",
) -> str:
    """Format Pokemon Red observation for LLM consumption with comprehensive text-based state information.

    This function provides rich, semantic game state information to eliminate
    the need for visual processing, as specified in text_port.txt requirements.
    """

    obs_lines = [
        "=== POKEMON RED GAME STATE ===",
        f"Step: {pub.progress.step_count}",
    ]

    # === VISUAL SCREEN INFORMATION ===
    if screen_analysis:
        obs_lines.extend(["", "=== VISUAL SCREEN ANALYSIS ==="])

        # Add detailed screen description - only include ASCII for state_and_ascii mode
        if mode == "state_and_ascii":
            screen_description = create_detailed_screen_description(screen_analysis)
        else:
            # For state_and_screen mode, show summary without ASCII
            screen_description = f"SCREEN TYPE: {screen_analysis.get('screen_type', 'UNKNOWN')}\n"

            # Add color analysis
            if "colors" in screen_analysis:
                colors_text = "DOMINANT COLORS: " + ", ".join(
                    [f"{color}({pct}%)" for color, pct in screen_analysis["colors"].items()]
                )
                screen_description += colors_text + "\n"

            # Add entity detection summary
            if "entities" in screen_analysis:
                screen_description += (
                    f"DETECTED ENTITIES: {len(screen_analysis['entities'])} sprite-like objects\n"
                )

            # Add UI elements
            if "ui_elements" in screen_analysis:
                ui_elements = screen_analysis["ui_elements"]
                if ui_elements:
                    screen_description += f"UI: {', '.join(ui_elements)} detected\n"

        obs_lines.append(screen_description)

    # === WORLD INFORMATION ===
    obs_lines.extend(
        [
            "",
            "=== WORLD LOCATION ===",
            f"Map ID: {pub.world.map_id} | Position: ({pub.world.player_x}, {pub.world.player_y})",
        ]
    )

    # === PLAYER PROGRESS ===
    obs_lines.extend(
        [
            "",
            "=== PLAYER PROGRESS ===",
            f"Badges: {pub.progress.badge_count}/8 (0x{pub.progress.badges:02X})",
            f"Money: ${pub.progress.money:,}",
        ]
    )

    # === POKEMON PARTY ===
    obs_lines.extend(["", "=== POKEMON PARTY ==="])

    if pub.party:
        for i, pokemon in enumerate(pub.party, 1):
            status_icon = "●" if pokemon.hp_current > 0 else "✗"
            obs_lines.append(
                f"{i}. Species#{pokemon.species_id:03d} L{pokemon.level} | "
                f"HP:{pokemon.hp_current}/{pokemon.hp_max} ({pokemon.hp_percentage:.1f}%) {status_icon} | "
                f"XP:{pokemon.xp:,}"
            )
    else:
        obs_lines.append("No Pokemon in party")

    # === INVENTORY ===
    obs_lines.extend(["", "=== INVENTORY ==="])

    if pub.inventory:
        # Show first 8 items with quantities
        for item in pub.inventory[:8]:
            obs_lines.append(f"Item#{item.item_id:03d} x{item.quantity}")

        if len(pub.inventory) > 8:
            obs_lines.append(f"... and {len(pub.inventory) - 8} more items")

        obs_lines.append(f"Total Items: {len(pub.inventory)}")
    else:
        obs_lines.append("No items in inventory")

    # === GAME SYSTEM STATE ===
    obs_lines.extend(["", "=== GAME SYSTEM STATE ==="])

    # Just show raw state without interpretation
    if pub.system.in_battle:
        obs_lines.append("In Battle: True")
        obs_lines.append(f"Battle Outcome: {pub.system.battle_outcome}")
    else:
        obs_lines.append("In Battle: False")

    if pub.system.text_box_active:
        obs_lines.append("Text Box Active: True")
    else:
        obs_lines.append("Text Box Active: False")

    obs_lines.append(f"Warp Flag: {pub.system.warp_flag}")

    # === TECHNICAL INFO ===
    obs_lines.extend(
        [
            "",
            "=== TECHNICAL INFO ===",
            f"Last Reward: {priv.reward_last_step:.3f}",
            f"Total Reward: {priv.total_reward:.3f}",
            f"Terminated: {priv.terminated} | Truncated: {priv.truncated}",
        ]
    )

    if pub.error_info:
        obs_lines.append(f"Error: {pub.error_info}")

    obs_lines.append("=== END GAME STATE ===")

    return "\n".join(obs_lines)


# --- Custom observation callable for Pokemon Red ---
class PokemonRedHistoryObservationCallable(GetObservationCallable):
    def __init__(
        self,
        max_history: int = 1,
        mode: Literal["state_and_ascii", "state_and_screen"] = "state_and_screen",
    ):
        self._hist_obs: Deque[str] = deque(maxlen=max_history)
        self._hist_pub_state: Deque[PokemonRedPublicState] = deque(maxlen=max_history)
        self._hist_priv_state: Deque[PokemonRedPrivateState] = deque(maxlen=max_history)
        self._last_state_hash = None
        self._stuck_count = 0
        self.screen_buffer = None  # Store screen buffer for agent access
        self.mode = mode  # Store mode for observation formatting

        # Initialize reward manager for early game rewards
        self.reward_manager = EarlyGameRewardManager()
        self._last_state_dict = None  # Store previous state for reward calculation

    async def get_observation(
        self, pub: PokemonRedPublicState, priv: PokemonRedPrivateState
    ) -> InternalObservation:
        if pub is None or priv is None:
            raise RuntimeError("Missing public or private state in get_observation - HARD FAIL")

        # Create current state dict for reward calculation
        current_state_dict = {
            "map_id": pub.map_id,
            "player_x": pub.player_x,
            "player_y": pub.player_y,
            "step_count": pub.step_count,
            "text_box_active": pub.system.text_box_active,
            "in_battle": pub.system.in_battle,
            "party": [
                {
                    "species_id": p.species_id,
                    "level": p.level,
                    "hp_current": p.hp_current,
                    "hp_max": p.hp_max,
                }
                for p in pub.party
            ],
            "inventory": [
                {"item_id": item.item_id, "quantity": item.quantity} for item in pub.inventory
            ],
            "money": pub.progress.money,
            "badges": pub.progress.badges,
        }

        # Calculate rewards if we have a previous state
        additional_reward = 0.0
        if self._last_state_dict is not None:
            try:
                additional_reward = await self.reward_manager.calculate_rewards(
                    current_state_dict,
                    self._last_state_dict,
                    {"buttons_pressed": []},  # Could track actual buttons if needed
                )
            except Exception as e:
                print(f"[REWARD_ERROR] Failed to calculate rewards: {e}")

        # Store current state for next iteration
        self._last_state_dict = current_state_dict.copy()

        # Check if we're stuck (same position and menu state for multiple steps)
        # Use property accessors that handle the new state structure
        current_state_hash = hash((pub.player_x, pub.player_y, pub.map_id, pub.step_count))
        if self._last_state_hash == current_state_hash and pub.step_count > 1:
            self._stuck_count += 1
            if self._stuck_count >= 3:
                raise RuntimeError(
                    f"Agent stuck in same state for {self._stuck_count} steps - HARD FAIL. Position: ({pub.player_x}, {pub.player_y}), Map: {pub.map_id}"
                )
        else:
            self._stuck_count = 0
            self._last_state_hash = current_state_hash

        # Extract screen buffer for agent vision - FAIL HARD if screen access doesn't work
        additional_context = ""
        screen_analysis = None

        try:
            # Look for environment in call stack to access engine/emulator
            import inspect

            frame = inspect.currentframe()
            env = None

            # Walk up the call stack to find the environment
            while frame:
                if "self" in frame.f_locals and hasattr(frame.f_locals["self"], "engine"):
                    env = frame.f_locals["self"]
                    break
                frame = frame.f_back

            if not env or not hasattr(env, "engine") or not env.engine:
                raise RuntimeError("Cannot access environment engine - HARD FAIL")

            # REQUIRE screen access to work
            if not hasattr(env.engine, "emulator") or not env.engine.emulator:
                raise RuntimeError("Emulator not available - HARD FAIL")

            if not hasattr(env.engine.emulator, "screen"):
                raise RuntimeError("Emulator screen not available - HARD FAIL")

            # Use PyBoy's documented screen.ndarray property - shape (144, 160, 4) RGBA
            screen_buffer = (
                env.engine.emulator.screen.ndarray.copy()
            )  # Copy to avoid reference issues

            if screen_buffer is None:
                raise RuntimeError("Screen ndarray is None - HARD FAIL")

            # Store screen buffer for agent to access
            self.screen_buffer = screen_buffer
            print(f"[DEBUG] Successfully extracted screen buffer with shape: {screen_buffer.shape}")

            # Perform detailed screen analysis
            screen_analysis = analyze_screen_buffer(screen_buffer)
            print(
                f"[DEBUG] Screen analysis completed - type: {screen_analysis.get('screen_type', 'UNKNOWN')}"
            )

            # Get additional game state context - REQUIRE this to work
            current_state = env.engine._extract_current_state()
            if not current_state:
                raise RuntimeError("Failed to extract game state - HARD FAIL")

            # Use the new structured state information from the public state
            additional_context += f"\nWarp Flag: {pub.system.warp_flag}"
            additional_context += f"\nBattle Outcome: {pub.system.battle_outcome}"
            additional_context += f"\nInventory Count: {len(pub.inventory)}"

        except Exception as e:
            # HARD FAIL on any screen/context extraction errors
            raise RuntimeError(f"Screen/context extraction HARD FAIL: {e}")

        # Format the base observation with screen analysis
        if self.mode == "state_and_ascii":
            # Include ASCII analysis but no screen buffer in observation
            formatted_obs = format_obs_for_llm_from_states(pub, priv, screen_analysis, self.mode)
        else:
            # Include screen analysis for screen mode
            formatted_obs = format_obs_for_llm_from_states(pub, priv, screen_analysis, self.mode)

        # Add context info
        enhanced_obs = formatted_obs.replace(
            "\n=== END GAME STATE ===", f"{additional_context}\n=== END GAME STATE ==="
        )

        # Add reward information to the observation
        if additional_reward > 0 or self.reward_manager.total_reward_earned > 0:
            reward_info = "\n\n=== REWARD PROGRESS ===\n"
            if additional_reward > 0:
                reward_info += f"Step Reward: +{additional_reward:.1f}\n"
            reward_info += f"Total Rewards Earned: {self.reward_manager.total_reward_earned:.1f}\n"

            # Show recent reward achievements (last 3)
            if self.reward_manager.reward_history:
                reward_info += "Recent Achievements:\n"
                for achievement in self.reward_manager.reward_history[-3:]:
                    for component in achievement["components"]:
                        reward_info += f"• {component['component']}: +{component['reward']:.1f}\n"

            enhanced_obs = enhanced_obs.replace(
                "\n=== END GAME STATE ===", f"{reward_info}=== END GAME STATE ==="
            )

        self._hist_obs.append(enhanced_obs)
        self._hist_pub_state.append(pub)
        self._hist_priv_state.append(priv)

        observation_dict = {
            "public": pub,
            "private": priv,
            "formatted_obs": enhanced_obs,
            "history_formatted_obs": list(self._hist_obs),
            "history_public_states": list(self._hist_pub_state),
            "history_private_states": list(self._hist_priv_state),
        }

        # Only include screen buffer for screen mode
        if self.mode == "state_and_screen":
            observation_dict["screen_buffer"] = self.screen_buffer

        return observation_dict  # type: ignore[return-value]


# --- Pydantic Models for Tool Arguments ---
class PokemonRedInteractArgs(BaseModel):
    buttons: List[str] = Field(
        description="A sequence of 1-5 buttons to press in Pokemon Red (e.g., ['A'], ['UP', 'RIGHT'], ['START', 'DOWN', 'A']). Each button should be one of: A, B, UP, DOWN, LEFT, RIGHT, START, SELECT."
    )
    reasoning: str = Field(
        description="A brief explanation of why this sequence of buttons was chosen and what you expect to accomplish."
    )

    @validator("buttons")
    def validate_buttons(cls, v):
        valid_buttons = {"A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"}
        if not v or len(v) == 0:
            raise ValueError("Must provide at least one button")
        if len(v) > 5:  # Reduced from 20 to 5
            raise ValueError("Cannot provide more than 5 buttons in sequence")
        for button in v:
            if button.upper() not in valid_buttons:
                raise ValueError(f"Invalid button: {button}. Valid buttons: {valid_buttons}")
        return [button.upper() for button in v]  # Normalize to uppercase


class TerminateArgs(BaseModel):
    reason: str = Field(
        description="Reason for termination (e.g., 'all tasks complete', 'stuck', 'max_steps_reached')."
    )


# --- Environment tool call wrapper ---
class PressButtonCall(EnvToolCall):
    """Helper class for creating button press calls"""

    def __init__(self, button: str, frames: int = 1):
        super().__init__(tool="press_button", args={"button": button, "frames": frames})


# --- ReAct agent for Pokemon Red ---
class ReActAgent:
    def __init__(self, llm, max_turns: int = 50):
        self.llm, self.max_turns = llm, max_turns
        self.history: List[Dict[str, Any]] = []
        self.system_name: str = "pokemon-red-react"
        self.system_id: Any = get_system_id(self.system_name)
        self.system_instance_id: str = str(uuid.uuid4())
        self.last_obs_dict: Optional[Dict[str, Any]] = None
        self.current_badges: int = 0

        # Valid button inputs for Pokemon Red
        self.valid_buttons = [
            "A",
            "B",
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT",
            "START",
            "SELECT",
        ]

        # Create proper BaseTool objects for zyk
        self.tools = [
            BaseTool(
                name="pokemon_red_interact",
                description="Interacts with the Pokemon Red game by pressing a button.",
                arguments=PokemonRedInteractArgs,
            ),
            BaseTool(
                name="terminate",
                description="Terminates the agent's execution if the task is considered complete or no useful progress can be made.",
                arguments=TerminateArgs,
            ),
        ]

    def _format_history_for_prompt(self) -> str:
        prompt_history = []
        for entry in self.history:
            if entry["type"] == "obs":
                prompt_history.append(f"OBSERVATION:\n{entry['content']}")
            elif entry["type"] == "tool_call":
                args_str = json.dumps(entry["tool_arguments"])
                prompt_history.append(
                    f"THOUGHT:\nI will call the tool `{entry['tool_name']}` with arguments: {args_str}\nACTION: (Tool call executed)"
                )
            elif entry["type"] == "tool_response":
                prompt_history.append(
                    "TOOL_RESPONSE:\n(Button pressed, new observation will follow if not terminal)"
                )
        return "\n".join(prompt_history)

    def _get_recent_reasoning_traces(self, k: int = 5) -> str:
        """Get the reasoning from the last k tool calls to help agent avoid repeating mistakes."""
        recent_reasoning = []
        tool_calls = [entry for entry in self.history if entry["type"] == "tool_call"]

        # Get last k tool calls
        for tool_call in tool_calls[-k:]:
            if "tool_arguments" in tool_call and "reasoning" in tool_call["tool_arguments"]:
                step_num = len(
                    [
                        e
                        for e in self.history[: self.history.index(tool_call) + 1]
                        if e["type"] == "tool_call"
                    ]
                )
                reasoning = tool_call["tool_arguments"]["reasoning"]
                buttons = tool_call["tool_arguments"].get("buttons", ["unknown"])
                recent_reasoning.append(
                    f"Step {step_num}: Pressed {buttons} - Reasoning: {reasoning}"
                )

        if recent_reasoning:
            # Add warning if same button pressed many times OR same button sequence repeated
            if len(recent_reasoning) >= 3:
                last_3_buttons = []
                last_3_sequences = []
                for trace in recent_reasoning[-3:]:
                    # Extract buttons from trace
                    if "Pressed ['" in trace:
                        start = trace.find("Pressed ['") + 10
                        end = trace.find("']", start)
                        if end > start:
                            buttons_str = trace[start:end]
                            # Handle both single buttons and sequences
                            if "', '" in buttons_str:
                                buttons = buttons_str.split("', '")
                            else:
                                buttons = [buttons_str]
                            last_3_buttons.append(buttons[0] if buttons else "unknown")
                            last_3_sequences.append(str(buttons))

                # Check for repeated single button
                if len(set(last_3_buttons)) == 1 and len(last_3_buttons) >= 3:
                    warning = f"\n⚠️ WARNING: You've pressed '{last_3_buttons[0]}' button {len(last_3_buttons)} times in a row! This button may not be working for the current situation. Try a different approach like pressing 'B' to cancel, or movement buttons to navigate away.\n"
                    return (
                        "RECENT REASONING HISTORY:\n" + "\n".join(recent_reasoning) + warning + "\n"
                    )

                # Check for repeated button sequences
                if len(set(last_3_sequences)) == 1 and len(last_3_sequences) >= 3:
                    warning = f"\n⚠️ WARNING: You've used the same button sequence {last_3_sequences[0]} {len(last_3_sequences)} times in a row! This sequence may not be working. Try a completely different approach like 'B' to cancel or different movement directions.\n"
                    return (
                        "RECENT REASONING HISTORY:\n" + "\n".join(recent_reasoning) + warning + "\n"
                    )

            return "RECENT REASONING HISTORY:\n" + "\n".join(recent_reasoning) + "\n\n"
        return ""

    @trace_event_async(event_type="react_agent_decide")
    async def decide(
        self,
        obs_str: str,
        current_raw_obs: Dict[str, Any],
        mode: Literal["state_and_ascii", "state_and_screen"] = "state_and_screen",
    ) -> List[str]:
        print(f"[AGENT_DEBUG] Starting decide with obs: {obs_str[:100]}...")
        self.history.append({"type": "obs", "content": obs_str})
        self.last_obs_dict = current_raw_obs

        # Update current badge count from the raw observation
        if current_raw_obs and isinstance(current_raw_obs.get("public"), PokemonRedPublicState):
            pub_state: PokemonRedPublicState = current_raw_obs["public"]
            self.current_badges = pub_state.badges

        print(f"[AGENT_DEBUG] History length: {len(self.history)}")

        # Extract current step count for cache busting
        current_step_count = 0
        if current_raw_obs and isinstance(current_raw_obs.get("public"), PokemonRedPublicState):
            pub_state: PokemonRedPublicState = current_raw_obs["public"]
            current_step_count = pub_state.step_count

        # Extract screen buffer for vision only in screen mode
        screen_images_bytes = []
        if mode == "state_and_screen":
            try:
                # Get screen buffer directly from the observation
                if (
                    current_raw_obs
                    and "screen_buffer" in current_raw_obs
                    and current_raw_obs["screen_buffer"] is not None
                ):
                    screen_buffer = current_raw_obs["screen_buffer"]
                    print(f"[AGENT_DEBUG] Got screen buffer with shape: {screen_buffer.shape}")

                    # Convert screen buffer to base64 image
                    import base64
                    import io
                    from PIL import Image
                    import numpy as np

                    # Ensure the array is in the right format (0-255 uint8)
                    if screen_buffer.dtype != np.uint8:
                        if screen_buffer.max() <= 1.0:
                            screen_array = (screen_buffer * 255).astype(np.uint8)
                        else:
                            screen_array = screen_buffer.astype(np.uint8)
                    else:
                        screen_array = screen_buffer

                    # PyBoy screen format is (144, 160, 4) RGBA
                    if len(screen_array.shape) == 3 and screen_array.shape[2] == 4:  # RGBA
                        # Convert RGBA to RGB by dropping alpha channel
                        image = Image.fromarray(screen_array[:, :, :3], mode="RGB")
                    else:
                        raise ValueError(f"Unsupported screen array shape: {screen_array.shape}")

                    # DEBUG: Save the image to debug directory
                    debug_dir = Path(__file__).parent / "debug"
                    debug_dir.mkdir(exist_ok=True)
                    debug_filename = (
                        f"step_{current_step_count:04d}_agent_{self.system_instance_id[-8:]}.png"
                    )
                    debug_path = debug_dir / debug_filename
                    image.save(debug_path)
                    print(f"[DEBUG] Saved screen image to: {debug_path}")

                    # Convert to base64
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    buffer.seek(0)
                    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    screen_images_bytes = [base64_image]
                    print("[AGENT_DEBUG] Successfully converted screen to base64 image")
                else:
                    print("[AGENT_DEBUG] No screen buffer available in observation")

            except Exception as e:
                print(f"[AGENT_DEBUG] Failed to extract screen buffer: {e}")
                # Continue without screen - the text observation should still work

        # Create appropriate prompt based on mode
        if mode == "state_and_ascii":
            prompt = (
                f"{self._get_recent_reasoning_traces(k=5)}"
                f"CURRENT OBSERVATION:\n{obs_str}\n\n"
                "Based on the game state text and ASCII representation above, "
                "what is your reasoning and which tool (`pokemon_red_interact` or `terminate`) should you call next? "
                "The ASCII representation shows the visual layout of the screen. "
                "Look at your recent reasoning history to avoid repeating the same ineffective actions. "
                "Focus on making progress: collect badges, heal when HP is low, explore new areas, and interact with the world.\n"
                f"[Turn: {current_step_count}]"
            )
        else:  # state_and_screen
            prompt = (
                f"{self._get_recent_reasoning_traces(k=5)}"
                f"CURRENT OBSERVATION:\n{obs_str}\n\n"
                "Based on the game state text above AND the game screen image (if provided), "
                "what is your reasoning and which tool (`pokemon_red_interact` or `terminate`) should you call next? "
                "Look at both the text information and the visual screen to understand what's happening in the game. "
                "Look at your recent reasoning history to avoid repeating the same ineffective actions. "
                "Focus on making progress: collect badges, heal when HP is low, explore new areas, and interact with the world.\n"
                f"[Turn: {current_step_count}]"
            )

        system_message = (
            "You are an agent playing Pokemon Red. You receive structured game state information "
            "and can execute button sequences to interact with the game. "
            "Your goal is to progress through the game by collecting badges, training Pokemon, and exploring.\n\n"
            "GAME STATE INFORMATION:\n"
            "You receive detailed information about:\n"
            "• World Location: Current map ID and position coordinates\n"
            "• Player Progress: Badge count and money\n"
            "• Pokemon Party: Each Pokemon's species, level, HP, and XP\n"
            "• Inventory: Items with quantities\n"
            "• Game System State: Raw system flags and states\n"
        )

        if mode == "state_and_ascii":
            system_message += (
                "• Visual Screen Analysis: ASCII representation and entity detection\n\n"
            )
        else:
            system_message += (
                "• Visual Screen Analysis: ASCII representation and actual screen images\n\n"
            )

        system_message += (
            "AVAILABLE ACTIONS:\n"
            "You can execute sequences of 1-5 buttons. Use as many button presses as are appropriate - sometimes 1 or 2, occasionally 3-5:\n"
            f"• Available buttons: {', '.join(self.valid_buttons)}\n"
            "• Examples: ['A'], ['UP', 'RIGHT'], ['START', 'DOWN', 'A']\n\n"
            "IMPORTANT GUIDANCE:\n"
            "• If 'Text Box Active: True' and A button isn't working, try B to cancel or navigate away\n"
            "• If you're repeating the same button many times without progress, try a different approach\n"
            "• When stuck, try movement buttons (UP, DOWN, LEFT, RIGHT) to explore or navigate menus\n"
            "• B button often cancels menus or text boxes when A doesn't work\n"
            "• Look at your recent reasoning history to avoid ineffective repeated actions\n"
            "• Use shorter button sequences (1-3 buttons) rather than long sequences\n"
            "• If the same action doesn't work after 2-3 tries, try something completely different\n\n"
            "TOOLS AVAILABLE:\n"
            f"• pokemon_red_interact: Execute button sequences\n"
            "• terminate: End the session\n\n"
            "Make decisions based on the game state information provided. "
            "Always provide reasoning that references the specific state information."
        )

        print("=" * 80)
        print("[AI_INPUT] SYSTEM MESSAGE:")
        print(system_message)
        print("-" * 40)
        print("[AI_INPUT] USER MESSAGE:")
        print(prompt)
        print("-" * 40)
        print("[AI_INPUT] TOOLS:")
        print(json.dumps([tool.to_openai_tool() for tool in self.tools], indent=2))
        print("-" * 40)
        print(f"[AI_INPUT] IMAGES: {len(screen_images_bytes)} image(s) provided")
        print("=" * 80)

        print(
            f"[AGENT_DEBUG] Calling LLM with prompt length: {len(prompt)}, images: {len(screen_images_bytes)}"
        )
        response_obj = await self.llm.respond_async(
            system_message=system_message,
            user_message=prompt,
            tools=self.tools,
            images_as_bytes=screen_images_bytes,
        )
        print("[AGENT_DEBUG] LLM response received")

        print("=" * 80)
        print("[AI_OUTPUT] RESPONSE OBJECT:")
        print(f"Response type: {type(response_obj)}")
        print(f"Response content: {response_obj}")
        if hasattr(response_obj, "tool_calls"):
            print(f"Tool calls: {response_obj.tool_calls}")
        if hasattr(response_obj, "content"):
            print(f"Content: {response_obj.content}")
        print("=" * 80)

        assert response_obj.tool_calls, "Response object didn't have tool call"
        tool_calls = None

        try:
            if hasattr(response_obj, "tool_calls") and response_obj.tool_calls:
                tool_calls = response_obj.tool_calls
                print(f"[AGENT_DEBUG] Found {len(tool_calls)} tool calls")

            if not tool_calls:
                print("[AGENT_DEBUG] No tool calls found, falling back to A")
                self.history.append(
                    {
                        "type": "tool_call",
                        "tool_name": "pokemon_red_interact",
                        "tool_arguments": {
                            "button": "A",
                            "reasoning": "LLM failed to provide tool_calls, fallback to A button.",
                        },
                    }
                )
                return ["A"]

            if len(tool_calls) == 0:
                print("[AGENT_DEBUG] Empty tool calls list, falling back to A")
                self.history.append(
                    {"type": "error", "content": "LLM returned empty tool_calls list."}
                )
                return ["A"]

            tool_call_data = tool_calls[0]
            tool_name = ""
            tool_args_str = ""

            if (
                hasattr(tool_call_data, "function")
                and hasattr(tool_call_data.function, "name")
                and hasattr(tool_call_data.function, "arguments")
            ):
                tool_name = tool_call_data.function.name
                tool_args_str = tool_call_data.function.arguments
            elif (
                isinstance(tool_call_data, dict)
                and "function" in tool_call_data
                and isinstance(tool_call_data["function"], dict)
            ):
                tool_name = tool_call_data["function"].get("name")
                tool_args_str = tool_call_data["function"].get("arguments")
                if not isinstance(tool_args_str, str):
                    tool_arguments_dict = tool_args_str
                    tool_args_str = json.dumps(tool_arguments_dict)
                else:
                    tool_arguments_dict = json.loads(tool_args_str)
            else:
                print("[AGENT_DEBUG] Unexpected tool_call structure, falling back to A")
                self.history.append({"type": "error", "content": "Unexpected tool_call structure."})
                return ["A"]

            print(f"[AGENT_DEBUG] Tool name: {tool_name}, Args: {tool_args_str}")

            if not tool_args_str:
                print(f"[AGENT_DEBUG] Missing arguments for tool {tool_name}, falling back to A")
                self.history.append(
                    {
                        "type": "error",
                        "content": f"Missing arguments for tool {tool_name}. Args string: '{tool_args_str}'",
                    }
                )
                return ["A"]

            tool_arguments = json.loads(tool_args_str)

            self.history.append(
                {
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "tool_arguments": tool_arguments,
                }
            )

            if tool_name == "pokemon_red_interact":
                print("[AGENT_DEBUG] Processing pokemon_red_interact tool call")
                validated_args = PokemonRedInteractArgs(**tool_arguments)
                buttons = validated_args.buttons
                print(
                    f"[AGENT_DEBUG] Buttons: {buttons}, Valid: {[button in self.valid_buttons for button in buttons]}"
                )

                invalid_buttons = [button for button in buttons if button not in self.valid_buttons]
                if invalid_buttons:
                    print(f"[AGENT_DEBUG] Invalid buttons: {invalid_buttons}, falling back to A")
                    self.history.append(
                        {
                            "type": "error",
                            "content": f"Invalid buttons: {invalid_buttons}. Falling back to A.",
                        }
                    )
                    return ["A"]
                print(f"[AGENT_DEBUG] Returning buttons: {buttons}")
                return buttons

            elif tool_name == "terminate":
                print("[AGENT_DEBUG] Processing terminate tool call")
                # Allow termination if agent decides
                print("[AGENT_DEBUG] Agent decided to terminate, returning TERMINATE")
                return ["TERMINATE"]

            else:
                print(f"[AGENT_DEBUG] Unknown tool_name: {tool_name}, falling back to A")
                self.history.append({"type": "error", "content": f"Unknown tool_name: {tool_name}"})
                return ["A"]

        except Exception as e:
            error_content = (
                f"Error processing LLM response: {str(e)}. Response: {str(response_obj)[:500]}"
            )
            print(f"[AGENT_DEBUG] Exception in decide: {error_content}")
            self.history.append({"type": "error", "content": error_content})
            return ["A"]


# --- Test for a single agent run ---
@pytest.mark.asyncio
async def test_react_agent_pokemon_red(
    tmp_path: Path,
    mode: Literal["state_and_ascii", "state_and_screen"] = "state_and_screen",
):
    # Create a simple Pokemon Red task instance for testing
    task_metadata = TaskInstanceMetadata()
    inst = PokemonRedTaskInstance(
        id=uuid.uuid4(),
        impetus=Impetus(instructions="Start your Pokemon journey and collect badges."),
        intent=Intent(
            rubric={"goal": "Collect badges and progress"},
            gold_trajectories=None,
            gold_state_diff={},
        ),
        metadata=task_metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
    )

    hist_cb = PokemonRedHistoryObservationCallable(max_history=1, mode=mode)
    env = PokemonRedEnvironment(inst, custom_step_obs=hist_cb)

    llm = LM(model_name="gpt-4.1-nano", formatting_model_name="gpt-4.1-nano", temperature=0.0)
    agent = ReActAgent(llm, max_turns=30)

    async def run_episode():
        obs_payload = await env.initialize()

        if "error" in obs_payload:
            print(f"Error during env.initialize: {obs_payload['error']}")
            return False, 0

        current_formatted_obs = obs_payload["formatted_obs"]
        raw_obs_for_agent_decision = obs_payload

        for turn in range(agent.max_turns):
            buttons = await agent.decide(current_formatted_obs, raw_obs_for_agent_decision, mode)

            if "TERMINATE" in buttons:
                obs_payload_next = obs_payload
                break

            # Execute button sequence one by one
            for i, button in enumerate(buttons):
                print(f"[DEBUG] Executing button {i + 1}/{len(buttons)}: {button}")
                obs_payload_next = await env.step([[PressButtonCall(button)]])

                if "error" in obs_payload_next:
                    raise RuntimeError(
                        f"Environment step error on button {i + 1}: {obs_payload_next['error']}"
                    )

                # Update observation after each button press
                obs_payload = obs_payload_next

                # Check if environment terminated after this button
                if obs_payload["private"].terminated or obs_payload["private"].truncated:
                    print(
                        f"[DEBUG] Environment terminated/truncated after button {i + 1}/{len(buttons)}"
                    )
                    break

        if "obs_payload_next" not in locals():
            obs_payload_next = obs_payload

        if "error" in obs_payload_next:
            return False, agent.current_badges

        final_private_state: PokemonRedPrivateState = obs_payload_next["private"]
        episode_successful = final_private_state.terminated or final_private_state.truncated
        return episode_successful, agent.current_badges

    episode_completed, badges_collected = await run_episode()

    dataset = Dataset(
        questions=[
            TrainingQuestion(
                id="pokemon_red_ep_test",
                intent="progress_in_game",
                criteria="completed_episode_or_collected_badges",
            )
        ],
        reward_signals=[
            RewardSignal(
                question_id="pokemon_red_ep_test",
                run_id=agent.system_instance_id,
                system_instance_id=agent.system_instance_id,
                reward=1 if episode_completed or badges_collected > 0 else 0,
                error_message="" if episode_completed else "Episode not completed as expected.",
                metadata={
                    "agent_history": agent.history,
                    "badges_collected": badges_collected,
                    "total_reward_earned": hist_cb.reward_manager.total_reward_earned,
                    "reward_history": hist_cb.reward_manager.reward_history,
                },
            )
        ],
    )
    # upload(dataset=dataset)  # Optional: uncomment to upload trace

    assert episode_completed or badges_collected > 0, (
        "Agent failed to complete the episode or collect any badges in the test."
    )


async def eval_react_pokemon_red(
    model_name: str = "gpt-4o-mini",
    max_turns: int = 20,
    mode: Literal["state_and_ascii", "state_and_screen"] = "state_and_screen",
) -> None:
    """
    Run ReAct agents on Pokemon Red instances of different difficulties,
    and print aggregated success rates and average badges collected.
    """
    from tabulate import tabulate

    current_model_name_for_eval = model_name

    _temp_llm_for_names = LM(
        model_name=current_model_name_for_eval,
        formatting_model_name=current_model_name_for_eval,
        temperature=0.0,
    )
    _temp_agent_for_names = ReActAgent(_temp_llm_for_names)
    actual_system_name = _temp_agent_for_names.system_name

    # ------------------------------------------------------------------ helpers
    async def run_episode_eval(
        inst: PokemonRedTaskInstance, agent_max_turns: int
    ) -> tuple[bool, int, float, list]:
        """Run a single agent/instance episode and return (success_status, badges_collected, total_rewards, reward_history)."""
        print(f"[DEBUG] Starting episode for instance {inst.id}")
        hist_cb = PokemonRedHistoryObservationCallable(max_history=1, mode=mode)
        env = PokemonRedEnvironment(inst, custom_step_obs=hist_cb)

        llm_for_episode = LM(
            model_name=current_model_name_for_eval,
            formatting_model_name=current_model_name_for_eval,
            temperature=0.0,
        )
        agent = ReActAgent(llm_for_episode, max_turns=agent_max_turns)
        print(f"[DEBUG] Created agent with max_turns={agent_max_turns}")

        print("[DEBUG] Initializing environment...")
        obs_payload = await env.initialize()
        print(
            f"[DEBUG] Environment initialized. Obs keys: {list(obs_payload.keys()) if isinstance(obs_payload, dict) else type(obs_payload)}"
        )
        if "error" in obs_payload:
            raise RuntimeError(f"Environment initialization failed: {obs_payload['error']}")

        current_formatted_obs = obs_payload["formatted_obs"]
        raw_obs_for_agent_decision = obs_payload
        print(f"[DEBUG] Initial formatted obs: {current_formatted_obs[:200]}...")

        # Track state changes to detect if agent is stuck
        last_position = None
        last_map_id = None
        stuck_count = 0
        same_button_count = 0
        last_button = None

        turn_count = 0
        for turn_idx in range(agent.max_turns):
            turn_count += 1
            print(f"[DEBUG] === Turn {turn_idx + 1}/{agent.max_turns} ===")
            print(f"[DEBUG] Agent deciding on obs: {current_formatted_obs[:100]}...")

            buttons = await agent.decide(current_formatted_obs, raw_obs_for_agent_decision, mode)
            print(f"[DEBUG] Agent decided buttons: {buttons}")

            # Check for repeated button presses
            if buttons[0] == last_button:
                same_button_count += 1
                # Increased tolerance since engine now handles retries automatically
                # and some game states may legitimately require the same button multiple times
                if same_button_count >= 8:  # Increased from 4 to 8
                    print(
                        f"[WARNING] Agent pressed same button '{buttons[0]}' {same_button_count} times in a row"
                    )
                    # Don't hard fail anymore - let the engine's retry mechanism handle it
                    # raise RuntimeError(f"Agent pressing same button '{buttons[0]}' {same_button_count} times in a row - HARD FAIL")
            else:
                same_button_count = 1
                last_button = buttons[0]

            if "TERMINATE" in buttons:
                print(f"[DEBUG] Agent decided to terminate after {turn_count} turns")
                break

            print(f"[DEBUG] Stepping environment with buttons {buttons}")

            try:
                # Execute button sequence one by one
                for i, button in enumerate(buttons):
                    print(f"[DEBUG] Executing button {i + 1}/{len(buttons)}: {button}")
                    obs_payload_next = await env.step([[PressButtonCall(button)]])

                    if "error" in obs_payload_next:
                        raise RuntimeError(
                            f"Environment step error on button {i + 1}: {obs_payload_next['error']}"
                        )

                    # Update observation after each button press
                    obs_payload = obs_payload_next

                    # Check if environment terminated after this button
                    if obs_payload["private"].terminated or obs_payload["private"].truncated:
                        print(
                            f"[DEBUG] Environment terminated/truncated after button {i + 1}/{len(buttons)}"
                        )
                        break
            except RuntimeError as e:
                if "HARD FAIL" in str(e):
                    raise  # Re-raise hard failures immediately
                raise RuntimeError(f"Environment step failed: {e}")

            print(
                f"[DEBUG] Environment step completed. Obs keys: {list(obs_payload.keys()) if isinstance(obs_payload, dict) else type(obs_payload)}"
            )

            if "error" in obs_payload:
                raise RuntimeError(f"Environment step error: {obs_payload['error']}")

            # Check if state is changing meaningfully using screen buffer hashes
            screen_changed = True
            if obs_payload.get("screen_buffer") is not None:
                import hashlib

                current_screen_hash = hashlib.md5(
                    obs_payload["screen_buffer"].tobytes()
                ).hexdigest()
                if not hasattr(run_episode_eval, "last_screen_hash"):
                    run_episode_eval.last_screen_hash = None
                    run_episode_eval.same_screen_count = 0

                if run_episode_eval.last_screen_hash == current_screen_hash:
                    run_episode_eval.same_screen_count += 1
                    screen_changed = False
                else:
                    run_episode_eval.same_screen_count = 0
                    screen_changed = True

                run_episode_eval.last_screen_hash = current_screen_hash
                print(
                    f"[DEBUG] Screen hash: {current_screen_hash[:8]}..., Same count: {run_episode_eval.same_screen_count}, Changed: {screen_changed}"
                )

                # More intelligent failure detection for Pokemon Red
                # Based on investigation: menu_state=1 is normal overworld state, not a stuck condition
                # B button doing nothing is often expected (no menu to close)
                button_tolerance = {
                    "B": 15,  # B often does nothing in overworld - very lenient
                    "A": 10,  # A for interactions/dialogue - moderately lenient
                    "START": 8,  # START for menu opening - moderate
                    "SELECT": 8,  # SELECT for menu navigation - moderate
                    "UP": 5,  # Movement buttons - less lenient
                    "DOWN": 5,
                    "LEFT": 5,
                    "RIGHT": 5,
                }

                max_same_button = button_tolerance.get(
                    buttons[0], 5
                )  # Default to 5 for unknown buttons
                min_screen_unchanged = 12  # Increased - Pokemon Red often has static screens
                min_turn_threshold = 10  # Increased - allow more exploration time

                # Only fail if BOTH conditions are met:
                # 1. Screen hasn't changed for many turns (visual stuckness)
                # 2. Agent is repeating ineffective actions beyond reasonable tolerance
                if (
                    run_episode_eval.same_screen_count >= min_screen_unchanged
                    and turn_idx > min_turn_threshold
                    and same_button_count >= max_same_button
                ):
                    # Additional check: don't fail on B button if menu_state indicates normal overworld
                    if buttons[0] == "B":
                        # B button in overworld is often ineffective but not necessarily wrong
                        # Just be more lenient with B button in general
                        if same_button_count < 20:  # Much more lenient for B button
                            print(
                                f"[DEBUG] B button often ineffective in overworld - allowing more attempts ({same_button_count}/20)"
                            )
                            # Continue without failing
                            obs_payload = obs_payload_next
                            continue

                    print(
                        f"[WARNING] Agent appears stuck - screen unchanged for {run_episode_eval.same_screen_count} turns with repeated button '{buttons[0]}' {same_button_count} times"
                    )
                    print(
                        f"[WARNING] Button tolerance for '{buttons[0]}': {max_same_button}, screen unchanged threshold: {min_screen_unchanged}"
                    )
                    raise RuntimeError(
                        f"Agent stuck - screen unchanged for {run_episode_eval.same_screen_count} turns with repeated button '{buttons[0]}' ({same_button_count} times, tolerance: {max_same_button}) - HARD FAIL"
                    )

            # Legacy position-based detection (keep as fallback but make more lenient)
            current_pub = obs_payload["public"]
            current_position = (current_pub.player_x, current_pub.player_y)
            current_map_id = current_pub.map_id

            # Only check position-based stuck if screen is also not changing
            if (
                last_position == current_position
                and last_map_id == current_map_id
                and not screen_changed
                and turn_idx > 8
            ):  # Much more lenient - allow many turns for dialogue
                stuck_count += 1
                if stuck_count >= 8:  # Require many more turns of true stuck state
                    raise RuntimeError(
                        f"Agent truly stuck - no position or screen changes for {stuck_count} turns. Position: {current_position}, Map: {current_map_id} - HARD FAIL"
                    )
            else:
                stuck_count = 0

            last_position = current_position
            last_map_id = current_map_id

            current_formatted_obs = obs_payload["formatted_obs"]
            raw_obs_for_agent_decision = obs_payload

            agent.history.append(
                {
                    "type": "tool_response",
                    "content": f"Button sequence executed: {buttons}",
                }
            )

            print(f"[DEBUG] New formatted obs: {current_formatted_obs[:100]}...")

            if obs_payload["private"].terminated or obs_payload["private"].truncated:
                print(f"[DEBUG] Environment terminated/truncated after {turn_count} turns")
                print(
                    f"[DEBUG] Terminated: {obs_payload['private'].terminated}, Truncated: {obs_payload['private'].truncated}"
                )
                break

        print(f"[DEBUG] Episode completed after {turn_count} turns")
        final_private_state: PokemonRedPrivateState = obs_payload["private"]
        run_successful = final_private_state.terminated or final_private_state.truncated
        badges_collected = agent.current_badges
        total_rewards = hist_cb.reward_manager.total_reward_earned
        print(
            f"[DEBUG] Episode result - successful: {run_successful}, badges: {badges_collected}, rewards: {total_rewards:.1f}"
        )
        print(
            f"[DEBUG] Final private state - terminated: {final_private_state.terminated}, truncated: {final_private_state.truncated}"
        )
        print(f"[DEBUG] Total reward: {final_private_state.total_reward}")
        return (
            run_successful,
            badges_collected,
            total_rewards,
            hist_cb.reward_manager.reward_history,
        )

    # ---------------------------------------------------------------- instance factory
    async def make_pokemon_red_instances(
        difficulty: str, n_instances: int = 3, start_seed: int = 0
    ) -> List[PokemonRedTaskInstance]:
        instances = []

        for i in range(n_instances):
            current_seed = start_seed + i
            metadata = TaskInstanceMetadata()
            instance = PokemonRedTaskInstance(
                id=uuid.uuid4(),
                impetus=Impetus(
                    instructions=f"Play Pokemon Red on {difficulty} difficulty and collect badges."
                ),
                intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
                metadata=metadata,
                is_reproducible=True,
                initial_engine_snapshot=None,
            )
            instances.append(instance)
        return instances

    # ---------------------------------------------------------------- evaluation
    configs = [
        (
            "easy",
            1,
            max_turns,
        ),  # (difficulty_label, num_agents/instances, max_turns_per_episode) - Use parameter
    ]
    table_rows = []
    base_seed_for_difficulty = {"easy": 1000, "hard": 2000}

    print("Starting Pokemon Red ReAct Agent Evaluation...")
    print(f"Model: {current_model_name_for_eval}, System: {actual_system_name}")

    all_generated_task_data = []
    all_reward_achievements = {}  # Track all rewards across all runs

    print("\nGenerating task instances...")
    all_tasks_for_eval: Dict[str, List[PokemonRedTaskInstance]] = {}
    for label, num_agents, _ in configs:
        insts = await make_pokemon_red_instances(
            label, n_instances=num_agents, start_seed=base_seed_for_difficulty[label]
        )
        all_tasks_for_eval[label] = insts
        for inst in insts:
            instance_dict = await inst.serialize()
            all_generated_task_data.append(instance_dict)
        print(f"Generated {len(insts)} instances for {label} difficulty.")

    # Save all generated task data to a single JSON file
    dataset_dir = Path(__file__).parent.parent / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    synthetic_mix_path = dataset_dir / "synthetic_mix.json"
    with open(synthetic_mix_path, "w") as f:
        json.dump(all_generated_task_data, f, indent=2)
    print(
        f"Saved all {len(all_generated_task_data)} generated task instances to {synthetic_mix_path}"
    )

    # Now, run the evaluations using the generated tasks
    for label, num_agents, max_episode_turns in configs:
        print(
            f"\nRunning {num_agents} agents on {label} difficulty tasks (max_turns: {max_episode_turns})..."
        )
        current_difficulty_instances = all_tasks_for_eval[label]
        print(f"[DEBUG] About to run {len(current_difficulty_instances)} instances")

        import time

        start_time = time.time()
        print(
            f"[DEBUG] Starting asyncio.gather for {len(current_difficulty_instances)} episodes at {start_time}"
        )
        results = await asyncio.gather(
            *(run_episode_eval(inst, max_episode_turns) for inst in current_difficulty_instances)
        )
        end_time = time.time()
        print(f"[DEBUG] Completed asyncio.gather in {end_time - start_time:.2f} seconds")
        print(f"[DEBUG] Results: {results}")

        num_successful_runs = sum(1 for r_success, _, _, _ in results if r_success)
        total_badges = sum(r_badges for _, r_badges, _, _ in results)
        total_rewards = sum(r_rewards for _, _, r_rewards, _ in results)
        avg_badges = total_badges / len(results) if results else 0.0
        avg_rewards = total_rewards / len(results) if results else 0.0

        # Collect reward data for summary
        reward_counts = {}
        for inst_idx, (_, _, _, reward_history) in enumerate(results):
            # Get the reward history from the corresponding hist_cb
            # We need to access this from the episode run, so let's store it
            reward_counts[inst_idx] = reward_history

        # Aggregate rewards across all instances for this difficulty
        for inst_idx, reward_history in reward_counts.items():
            for achievement in reward_history:
                for component in achievement["components"]:
                    component_name = component["component"]
                    if component_name not in all_reward_achievements:
                        all_reward_achievements[component_name] = 0
                    all_reward_achievements[component_name] += 1

        table_rows.append(
            [
                label,
                f"{num_successful_runs}/{len(current_difficulty_instances)}",
                f"{avg_badges:.2f}",
                f"{avg_rewards:.1f}",
            ]
        )
        print(
            f"Completed {label}: {num_successful_runs}/{len(current_difficulty_instances)} successful, Avg. Badges: {avg_badges:.2f}, Avg. Rewards: {avg_rewards:.1f}"
        )

    print("\n--- Evaluation Summary ---")
    print(f"Model: {current_model_name_for_eval}, System: {actual_system_name}")
    print(
        tabulate(
            table_rows,
            headers=[
                "Difficulty",
                "Successful Runs",
                "Avg Badges Collected",
                "Avg Rewards Earned",
            ],
            tablefmt="github",
        )
    )

    # Display reward achievements summary
    if all_reward_achievements:
        print("\n--- Reward Achievements Summary ---")
        reward_summary_rows = []
        for reward_name, count in sorted(
            all_reward_achievements.items(), key=lambda x: x[1], reverse=True
        ):
            reward_summary_rows.append([reward_name, count])

        print(
            tabulate(
                reward_summary_rows,
                headers=["Reward Component", "Times Achieved"],
                tablefmt="github",
            )
        )
        print(f"\nTotal Unique Rewards Achieved: {len(all_reward_achievements)}")
        print(f"Total Reward Instances: {sum(all_reward_achievements.values())}")
    else:
        print("\n--- No Rewards Achieved ---")


if __name__ == "__main__":
    # To run the test:
    # import tempfile
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     asyncio.run(test_react_agent_pokemon_red(Path(tmpdir)))

    # better state management
    # To run the evaluation:
    asyncio.run(
        eval_react_pokemon_red(model_name="gpt-4.1-mini", max_turns=10, mode="state_and_screen")
    )
