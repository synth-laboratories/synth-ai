"""NetHack engine implementation with state management and NLE integration."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING, cast
import numpy as np
import logging

from synth_ai.environments.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from synth_ai.environments.reproducibility.core import IReproducibleEngine
from synth_ai.environments.environment.rewards.core import RewardStack, RewardComponent
from synth_ai.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_ai.environments.tasks.core import TaskInstance

logger = logging.getLogger(__name__)

# NLE imports are required
try:
    from .helpers.nle_wrapper import NLEWrapper
    from .helpers.action_mapping import convert_action_to_nle
    from .achievements import NetHackAchievements, calculate_balrog_reward
except ImportError as e:
    raise ImportError(
        "NLE (NetHack Learning Environment) is required but not installed. "
        "Please install it with: pip install nle"
    ) from e

if TYPE_CHECKING:
    from .taskset import NetHackTaskInstanceMetadata


@dataclass
class NetHackPublicState:
    """State visible to the agent."""

    # Game state
    dungeon_level: int = 1
    character_stats: Dict[str, Any] = field(default_factory=dict)
    inventory: List[Dict[str, Any]] = field(default_factory=list)
    position: Tuple[int, int] = (0, 0)

    # Observation data
    ascii_map: str = ""
    message: str = ""
    cursor_position: Tuple[int, int] = (0, 0)

    # Meta information
    turn_count: int = 0
    max_turns: int = 10000
    last_action: str = ""
    terminated: bool = False

    # Game context
    in_menu: bool = False
    menu_items: List[str] = field(default_factory=list)

    # Achievements tracking
    achievements: NetHackAchievements = field(default_factory=NetHackAchievements)
    achievements_unlocked: Dict[str, bool] = field(default_factory=dict)

    def diff(self, prev_state: "NetHackPublicState") -> Dict[str, Any]:
        """Track changes between states."""
        differences = {}

        if self.dungeon_level != prev_state.dungeon_level:
            differences["dungeon_level"] = (
                prev_state.dungeon_level,
                self.dungeon_level,
            )
        if self.position != prev_state.position:
            differences["position"] = (prev_state.position, self.position)
        if self.message != prev_state.message:
            differences["message"] = (prev_state.message, self.message)
        if self.turn_count != prev_state.turn_count:
            differences["turn_count"] = (prev_state.turn_count, self.turn_count)
        if self.terminated != prev_state.terminated:
            differences["terminated"] = (prev_state.terminated, self.terminated)
        if self.last_action != prev_state.last_action:
            differences["last_action"] = (prev_state.last_action, self.last_action)

        return differences

    @property
    def map_text(self) -> str:
        """Formatted ASCII dungeon map."""
        return self.ascii_map


@dataclass
class NetHackPrivateState:
    """Internal state (rewards, termination flags)."""

    reward_last: float = 0.0
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False

    # Progress tracking
    score: int = 0
    depth_reached: int = 1
    experience_level: int = 1
    monsters_killed: int = 0
    items_collected: int = 0

    # Balrog reward tracking
    balrog_reward_last: float = 0.0
    balrog_total_reward: float = 0.0

    def diff(self, prev_state: "NetHackPrivateState") -> Dict[str, Any]:
        """Track reward/progress changes."""
        differences = {}

        if self.reward_last != prev_state.reward_last:
            differences["reward_last"] = (prev_state.reward_last, self.reward_last)
        if self.total_reward != prev_state.total_reward:
            differences["total_reward"] = (prev_state.total_reward, self.total_reward)
        if self.score != prev_state.score:
            differences["score"] = (prev_state.score, self.score)
        if self.depth_reached != prev_state.depth_reached:
            differences["depth_reached"] = (
                prev_state.depth_reached,
                self.depth_reached,
            )

        return differences


@dataclass
class NetHackEngineSnapshot(StatefulEngineSnapshot):
    """Serialization container for NetHack engine state."""

    task_instance_dict: Dict[str, Any]
    engine_snapshot: Dict[str, Any]
    nle_state: Optional[Dict[str, Any]] = None  # NLE-specific state if available


class NetHackSurvivalComponent(RewardComponent):
    """Reward component for staying alive."""

    async def score(self, state: NetHackPublicState, action: str) -> float:
        if state.terminated:
            return -1.0  # Penalty for death
        return 0.01  # Small reward for each turn survived


class NetHackProgressComponent(RewardComponent):
    """Reward component for exploration and depth."""

    def __init__(self):
        self.last_depth = 1

    async def score(self, state: NetHackPublicState, action: str) -> float:
        reward = 0.0

        # Reward for reaching new dungeon levels
        if state.dungeon_level > self.last_depth:
            reward += 1.0 * (state.dungeon_level - self.last_depth)
            self.last_depth = state.dungeon_level

        return reward


class NetHackScoreComponent(RewardComponent):
    """Reward component based on game score."""

    def __init__(self):
        self.last_score = 0

    async def score(self, state: NetHackPublicState, action: str) -> float:
        # Get score from character stats - require it exists
        current_score = state.character_stats["score"]

        # Calculate score delta
        score_delta = current_score - self.last_score
        self.last_score = current_score

        # Scale the score reward (NLE scores can be large)
        return score_delta / 100.0 if score_delta > 0 else 0.0


class NetHackAchievementComponent(RewardComponent):
    """Reward component for unlocking achievements."""

    def __init__(self):
        self.last_unlocked = set()

    async def score(self, state: NetHackPublicState, action: str) -> float:
        reward = 0.0

        # Count newly unlocked achievements
        current_unlocked = set(k for k, v in state.achievements_unlocked.items() if v)
        new_achievements = current_unlocked - self.last_unlocked

        # Give rewards for different achievement types
        for achievement in new_achievements:
            if "first_" in achievement:
                reward += 1.0  # First-time achievements
            elif "reached_dlvl_" in achievement:
                reward += 2.0  # Depth achievements
            elif "killed_" in achievement and "monsters" in achievement:
                reward += 0.5  # Kill milestones
            elif "collected_" in achievement and "gold" in achievement:
                reward += 0.5  # Gold milestones
            elif "reached_level_" in achievement:
                reward += 1.5  # Experience level milestones
            elif "minetown" in achievement or "castle" in achievement:
                reward += 5.0  # Special locations
            elif "quest" in achievement:
                reward += 10.0  # Quest achievements
            else:
                reward += 0.5  # Default reward

        self.last_unlocked = current_unlocked
        return reward


class NetHackEngine(StatefulEngine, IReproducibleEngine):
    """NetHack game engine with NLE backend."""

    def __init__(self, task_instance: TaskInstance):
        self.task_instance = task_instance

        # Require proper metadata
        from .taskset import NetHackTaskInstanceMetadata

        if not isinstance(task_instance.metadata, NetHackTaskInstanceMetadata):
            raise TypeError(
                f"Expected NetHackTaskInstanceMetadata, got {type(task_instance.metadata).__name__}"
            )

        metadata = cast(NetHackTaskInstanceMetadata, task_instance.metadata)
        self.character_role = metadata.character_role
        self.max_turns = metadata.time_limit

        # Initialize NLE wrapper
        self.nle = NLEWrapper(character_role=self.character_role)

        # Initialize reward components with proper tracking - NO SURVIVAL NOISE
        self.progress_component = NetHackProgressComponent()
        self.score_component = NetHackScoreComponent()
        self.achievement_component = NetHackAchievementComponent()

        self.reward_stack = RewardStack(
            [
                self.progress_component,  # Depth progress
                self.score_component,  # Game score changes
                self.achievement_component,  # Achievement unlocks
            ]
        )

        # State tracking
        self.public_state: Optional[NetHackPublicState] = None
        self.private_state: Optional[NetHackPrivateState] = None

        # NLE observation processing
        self.last_nle_obs = None

    async def _reset_engine(
        self, *, seed: int | None = None
    ) -> Tuple[NetHackPrivateState, NetHackPublicState]:
        """Reset to initial state using NLE."""
        # Reset NLE environment with seed
        obs = await asyncio.to_thread(self.nle.reset, seed)
        self.last_nle_obs = obs

        # Log what we actually got from NLE
        logger.info(f"NLE reset returned observation keys: {list(obs.keys())}")
        if "player_stats" in obs:
            logger.info(f"Player stats keys: {list(obs['player_stats'].keys())}")

        # Initialize private state - require all fields
        player_stats = obs["player_stats"]  # Will KeyError if missing
        self.private_state = NetHackPrivateState(
            reward_last=0.0,
            total_reward=0.0,
            terminated=False,
            truncated=False,
            score=player_stats["score"],
            depth_reached=player_stats["depth"],
            experience_level=player_stats["experience_level"],
            monsters_killed=0,
            items_collected=0,
            balrog_reward_last=0.0,
            balrog_total_reward=0.0,
        )

        # Initialize public state from NLE observation - no fallbacks
        self.public_state = NetHackPublicState(
            dungeon_level=player_stats["depth"],
            character_stats={
                "hp": player_stats["hp"],
                "max_hp": player_stats["max_hp"],
                "strength": player_stats["strength"],
                "dexterity": player_stats["dexterity"],
                "constitution": player_stats["constitution"],
                "intelligence": player_stats["intelligence"],
                "wisdom": player_stats["wisdom"],
                "charisma": player_stats["charisma"],
                "gold": player_stats["gold"],
                "experience": player_stats["experience_points"],
                "level": player_stats["experience_level"],
                "ac": player_stats["ac"],
            },
            inventory=self._process_inventory(obs["inventory"]) if "inventory" in obs else [],
            position=(player_stats["y"], player_stats["x"]),
            ascii_map=obs["ascii_map"],
            message=obs["message"],
            cursor_position=obs.get(
                "cursor", (player_stats["y"], player_stats["x"])
            ),  # Cursor might not be in processed obs
            turn_count=0,
            max_turns=self.max_turns,
            last_action="",
            terminated=False,
            in_menu=obs.get("in_menu", False),  # Menu detection is heuristic-based
            menu_items=obs.get("menu_text", []),  # Menu text only present when in menu
            achievements=NetHackAchievements(),
            achievements_unlocked={},
        )

        # Reset reward components
        self.progress_component.last_depth = self.public_state.dungeon_level
        self.score_component.last_score = self.private_state.score

        return self.private_state, self.public_state

    def _process_inventory(self, inventory_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process NLE inventory format to our format."""
        processed_items = []
        for item in inventory_items:
            processed_items.append(
                {
                    "name": item["description"],
                    "count": 1,  # NLE doesn't always provide count
                    "letter": item["letter"],
                }
            )
        return processed_items

    async def _step_engine(self, action: str) -> Tuple[NetHackPrivateState, NetHackPublicState]:
        """Execute one step/action using NLE."""
        # print(f"===== NetHack Engine _step_engine called with action: {action} =====")
        if self.public_state is None or self.private_state is None:
            raise RuntimeError("Engine not initialized. Call _reset_engine first.")

        # Validate action
        if action not in self.nle.action_map and action not in ["terminate"]:
            # Try to handle menu selections and special cases
            if len(action) == 1 and (action.isalpha() or action.isdigit()):
                # Single character actions are likely menu selections
                pass
            else:
                raise ValueError(
                    f"Invalid action: {action}. Valid actions: {list(self.nle.action_map.keys())}"
                )

        # Update turn count
        self.public_state.turn_count += 1
        self.public_state.last_action = action

        # Define non-turn-consuming actions
        non_turn_actions = [
            "look",
            "farlook",
            "whatis",
            "identify",
            "discoveries",
            "conduct",
            "attributes",
            "help",
            "version",
            "history",
        ]

        # Warn about non-advancing actions
        if action in non_turn_actions:
            logger.warning(f"Action '{action}' is a free action that doesn't advance game time!")
            # If we're repeatedly using non-advancing actions, force a wait
            if hasattr(self, "_consecutive_free_actions"):
                self._consecutive_free_actions += 1
                if self._consecutive_free_actions >= 3:
                    logger.warning(
                        f"Too many consecutive free actions ({self._consecutive_free_actions}), forcing 'wait'"
                    )
                    action = "wait"
                    self._consecutive_free_actions = 0
            else:
                self._consecutive_free_actions = 1
        else:
            self._consecutive_free_actions = 0

        # Check for manual termination
        if action == "terminate":
            self.public_state.terminated = True
            self.private_state.terminated = True
            self.public_state.message = "Game terminated by agent."
            return self.private_state, self.public_state

        # Check for timeout
        if self.public_state.turn_count >= self.public_state.max_turns:
            self.public_state.terminated = True
            self.private_state.terminated = True
            self.private_state.truncated = True
            self.public_state.message = "Time limit reached. Game over!"
            return self.private_state, self.public_state

        # Execute action in NLE
        try:
            # Save previous observation BEFORE stepping
            prev_obs = self.last_nle_obs

            obs, reward, done, info = await asyncio.to_thread(self.nle.step, action)
            logger.debug(f"NLE step returned - reward: {reward}, done: {done}, info: {info}")
        except Exception as e:
            logger.error(f"NLE step failed for action '{action}': {e}")
            raise

        # Log observation structure on first few steps for debugging
        if self.public_state.turn_count < 3:
            logger.info(f"Turn {self.public_state.turn_count} observation keys: {list(obs.keys())}")

        # Update state from NLE observation - no defensive coding
        player_stats = obs["player_stats"]  # Will KeyError if missing

        # Track previous values for reward calculation
        prev_score = self.private_state.score
        prev_depth = self.private_state.depth_reached

        # Update private state
        self.private_state.score = player_stats["score"]
        self.private_state.depth_reached = max(
            self.private_state.depth_reached, player_stats["depth"]
        )
        self.private_state.experience_level = player_stats["experience_level"]

        # Update public state
        self.public_state.dungeon_level = player_stats["depth"]
        self.public_state.position = (player_stats["y"], player_stats["x"])
        self.public_state.ascii_map = obs["ascii_map"]
        self.public_state.message = obs["message"]
        self.public_state.cursor_position = obs.get(
            "cursor", (player_stats["y"], player_stats["x"])
        )
        self.public_state.in_menu = obs.get("in_menu", False)
        self.public_state.menu_items = obs.get("menu_text", [])

        # Update character stats - require all fields
        self.public_state.character_stats = {
            "hp": player_stats["hp"],
            "max_hp": player_stats["max_hp"],
            "strength": player_stats["strength"],
            "dexterity": player_stats["dexterity"],
            "constitution": player_stats["constitution"],
            "intelligence": player_stats["intelligence"],
            "wisdom": player_stats["wisdom"],
            "charisma": player_stats["charisma"],
            "gold": player_stats["gold"],
            "experience": player_stats["experience_points"],
            "level": player_stats["experience_level"],
            "ac": player_stats["ac"],
            "score": player_stats["score"],
        }

        # Update inventory
        self.public_state.inventory = (
            self._process_inventory(obs["inventory"]) if "inventory" in obs else []
        )

        # Handle termination from NLE
        if done:
            self.public_state.terminated = True
            self.private_state.terminated = True
            # Log info to understand structure
            logger.info(f"Game ended - info: {info}")
            if "end_status" in info and info["end_status"] == 0:  # 0 means death
                self.public_state.message = info.get(
                    "death_reason", "You died!"
                )  # death_reason might not always exist
            else:
                self.public_state.message = "Game ended."

        # Update achievements before calculating rewards
        newly_unlocked = self.public_state.achievements.update_from_observation(obs, prev_obs)
        self.public_state.achievements_unlocked.update(
            self.public_state.achievements.get_unlocked_achievements()
        )

        # Log newly unlocked achievements
        if newly_unlocked:
            logger.info(f"Achievements unlocked: {list(newly_unlocked.keys())}")

        # Calculate rewards
        # Base reward from NLE
        nle_reward = reward

        # Additional reward shaping
        step_reward = await self.reward_stack.step_reward(self.public_state, action)

        self.private_state.reward_last = nle_reward + step_reward
        self.private_state.total_reward += self.private_state.reward_last

        # Calculate Balrog-style reward
        self.private_state.balrog_reward_last = calculate_balrog_reward(obs, prev_obs)
        self.private_state.balrog_total_reward += self.private_state.balrog_reward_last

        # Log balrog reward changes with context
        if self.private_state.balrog_reward_last > 0:
            print(
                f"🏆 BALROG REWARD: +{self.private_state.balrog_reward_last:.3f} (total: {self.private_state.balrog_total_reward:.3f})"
            )
            balrog_score = self.public_state.achievements.balrog_progress.percent
            print(
                f"   Balrog score: {balrog_score}% (dungeon: {self.public_state.achievements.balrog_progress.dungeon_progression}, exp: {self.public_state.achievements.balrog_progress.experience_progression})"
            )

        # NOW update last_nle_obs for next step
        self.last_nle_obs = obs

        return self.private_state, self.public_state

    def __del__(self):
        """Cleanup NLE environment on deletion."""
        if hasattr(self, "nle"):
            self.nle.close()

    async def _serialize_engine(self) -> NetHackEngineSnapshot:
        """Serialize current state."""
        if self.public_state is None or self.private_state is None:
            raise RuntimeError("Cannot serialize uninitialized engine")

        # Get NLE state
        nle_state = None
        try:
            nle_state_bytes = await asyncio.to_thread(self.nle.get_state)
            # Convert bytes to base64 string for JSON serialization
            nle_state = base64.b64encode(nle_state_bytes).decode("ascii")
        except Exception as e:
            logger.warning(f"Failed to serialize NLE state: {e}")

        task_dict = await self.task_instance.serialize()
        logger.debug(f"Serialized task instance: {task_dict}")

        return NetHackEngineSnapshot(
            task_instance_dict=task_dict,
            engine_snapshot={
                "public_state": {
                    "dungeon_level": self.public_state.dungeon_level,
                    "character_stats": self.public_state.character_stats,
                    "inventory": self.public_state.inventory,
                    "position": self.public_state.position,
                    "ascii_map": self.public_state.ascii_map,
                    "message": self.public_state.message,
                    "cursor_position": self.public_state.cursor_position,
                    "turn_count": self.public_state.turn_count,
                    "max_turns": self.public_state.max_turns,
                    "last_action": self.public_state.last_action,
                    "terminated": self.public_state.terminated,
                    "in_menu": self.public_state.in_menu,
                    "menu_items": self.public_state.menu_items,
                },
                "private_state": {
                    "reward_last": self.private_state.reward_last,
                    "total_reward": self.private_state.total_reward,
                    "terminated": self.private_state.terminated,
                    "truncated": self.private_state.truncated,
                    "score": self.private_state.score,
                    "depth_reached": self.private_state.depth_reached,
                    "experience_level": self.private_state.experience_level,
                    "monsters_killed": self.private_state.monsters_killed,
                    "items_collected": self.private_state.items_collected,
                },
                "character_role": self.character_role,
                "progress_last_depth": self.progress_component.last_depth,
                "score_last_score": self.score_component.last_score,
            },
            nle_state=nle_state,
        )

    @classmethod
    async def _deserialize_engine(cls, snapshot: NetHackEngineSnapshot) -> "NetHackEngine":
        """Restore from serialized state."""
        from .taskset import NetHackTaskInstance

        task_instance = await NetHackTaskInstance.deserialize(snapshot.task_instance_dict)
        if task_instance is None:
            raise ValueError("Failed to deserialize task instance")
        engine = cls(task_instance)

        # Restore state
        engine_data = snapshot.engine_snapshot
        pub_data = engine_data["public_state"]
        priv_data = engine_data["private_state"]

        engine.public_state = NetHackPublicState(
            dungeon_level=pub_data["dungeon_level"],
            character_stats=pub_data["character_stats"],
            inventory=pub_data["inventory"],
            position=(pub_data["position"][0], pub_data["position"][1]),
            ascii_map=pub_data["ascii_map"],
            message=pub_data["message"],
            cursor_position=(
                pub_data["cursor_position"][0],
                pub_data["cursor_position"][1],
            ),
            turn_count=pub_data["turn_count"],
            max_turns=pub_data["max_turns"],
            last_action=pub_data["last_action"],
            terminated=pub_data["terminated"],
            in_menu=pub_data["in_menu"],
            menu_items=pub_data["menu_items"],
        )

        engine.private_state = NetHackPrivateState(
            reward_last=priv_data["reward_last"],
            total_reward=priv_data["total_reward"],
            terminated=priv_data["terminated"],
            truncated=priv_data["truncated"],
            score=priv_data["score"],
            depth_reached=priv_data["depth_reached"],
            experience_level=priv_data["experience_level"],
            monsters_killed=priv_data["monsters_killed"],
            items_collected=priv_data["items_collected"],
        )

        engine.character_role = engine_data["character_role"]

        # Restore reward component states
        engine.progress_component.last_depth = engine_data["progress_last_depth"]
        engine.score_component.last_score = engine_data["score_last_score"]

        # Restore NLE state if available
        if snapshot.nle_state:
            try:
                nle_state_bytes = base64.b64decode(snapshot.nle_state)
                await asyncio.to_thread(engine.nle.set_state, nle_state_bytes)
            except Exception as e:
                logger.warning(f"Failed to restore NLE state: {e}")
                # If we can't restore NLE state, reset it
                await asyncio.to_thread(engine.nle.reset)

        return engine

    def get_current_states_for_observation(
        self,
    ) -> Tuple[NetHackPrivateState, NetHackPublicState]:
        """Get current states without advancing."""
        if self.public_state is None or self.private_state is None:
            raise RuntimeError("Engine not initialized")
        return self.private_state, self.public_state


class NetHackObservationCallable(GetObservationCallable):
    """Standard observation callable for NetHack."""

    async def get_observation(
        self, pub: NetHackPublicState, priv: NetHackPrivateState
    ) -> InternalObservation:
        observation = {
            "ascii_map": pub.ascii_map,
            "message": pub.message,
            "character_stats": pub.character_stats,
            "inventory_summary": self._format_inventory(pub.inventory),
            "dungeon_level": pub.dungeon_level,
            "position": pub.position,
            "turn_count": pub.turn_count,
            "last_action": pub.last_action,
            "reward_last": priv.reward_last,
            "total_reward": priv.total_reward,
            "balrog_reward_last": priv.balrog_reward_last,
            "balrog_total_reward": priv.balrog_total_reward,
            "score": priv.score,
            "experience_level": priv.experience_level,
            "terminated": priv.terminated,
            "in_menu": pub.in_menu,
            "menu_items": pub.menu_items if pub.in_menu else [],
            "achievements_unlocked": pub.achievements_unlocked,
            "achievements_summary": self._format_achievements(pub.achievements_unlocked),
        }
        return observation  # type: ignore[return-value]

    def _format_inventory(self, inventory: List[Dict[str, Any]]) -> str:
        """Format inventory for display."""
        if not inventory:
            return "Your inventory is empty."

        items = []
        for item in inventory:
            items.append(f"- {item['name']} (count: {item.get('count', 1)})")
        return "\n".join(items)

    def _format_achievements(self, achievements: Dict[str, bool]) -> str:
        """Format achievements for display."""
        unlocked = [name for name, status in achievements.items() if status]
        if not unlocked:
            return "None unlocked yet"
        if len(unlocked) <= 5:
            return ", ".join(unlocked)
        else:
            return f"{', '.join(unlocked[:5])} and {len(unlocked) - 5} more"


class NetHackCheckpointObservationCallable(GetObservationCallable):
    """Checkpoint observation callable for NetHack."""

    async def get_observation(
        self, pub: NetHackPublicState, priv: NetHackPrivateState
    ) -> InternalObservation:
        observation = {
            "final_score": priv.score,
            "max_depth": priv.depth_reached,
            "experience_level": priv.experience_level,
            "monsters_killed": priv.monsters_killed,
            "items_collected": priv.items_collected,
            "turn_count_final": pub.turn_count,
            "total_reward": priv.total_reward,
            "balrog_total_reward": priv.balrog_total_reward,
            "terminated": priv.terminated,
            "truncated": priv.truncated,
            "character_role": pub.character_stats.get("role", "unknown"),
            "achievements_unlocked": list(pub.achievements_unlocked.keys()),
            "achievements_count": len([v for v in pub.achievements_unlocked.values() if v]),
            "achievement_stats": {
                "depth_reached": pub.achievements.depth_reached,
                "monsters_killed": pub.achievements.monsters_killed,
                "gold_collected": pub.achievements.gold_collected,
                "items_collected": pub.achievements.items_picked_up,
                "max_level": pub.achievements.max_level_reached,
                "turns_survived": pub.achievements.turns_survived,
                "balrog_score": pub.achievements.balrog_progress.percent,
            },
        }
        return observation  # type: ignore[return-value]
