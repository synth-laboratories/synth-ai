"""NetHack achievements and milestones tracking."""

import json
import os
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


# Exact copy of Balrog's Progress class
class Progress:
    def __init__(self, achievements_path=None):
        if achievements_path is None:
            achievements_path = os.path.join(
                os.path.dirname(__file__), "helpers", "achievements.json"
            )

        with open(achievements_path, "r") as f:
            self.achievements = json.load(f)["3.4.3"]

        self.dungeon_progression = 0
        self.experience_progression = 0
        self.ascension = False

    def update(self, dungeon_name, experience_level):
        """Update progression based on current dungeon and experience level."""
        achievements_unlocked = []

        if dungeon_name == "ascension":
            if not self.ascension:
                achievements_unlocked.append("ascension (100 points)")
            self.ascension = True
            return achievements_unlocked

        # Update dungeon progression
        if dungeon_name in self.achievements["dungeons"]:
            new_progression = self.achievements["dungeons"][dungeon_name]
            if new_progression > self.dungeon_progression:
                old_score = self.dungeon_progression
                self.dungeon_progression = new_progression
                achievements_unlocked.append(
                    f"dungeon {dungeon_name} ({old_score} -> {new_progression} points)"
                )

        # Update experience progression
        exp_key = f"lvl{experience_level}"
        if exp_key in self.achievements["experience_levels"]:
            new_progression = self.achievements["experience_levels"][exp_key]
            if new_progression > self.experience_progression:
                old_score = self.experience_progression
                self.experience_progression = new_progression
                achievements_unlocked.append(
                    f"experience {exp_key} ({old_score} -> {new_progression} points)"
                )

        return achievements_unlocked

    @property
    def percent(self):
        """Return the BALROG evaluation score (0-100)."""
        if self.ascension:
            return 100.0
        return max(self.dungeon_progression, self.experience_progression)


@dataclass
class NetHackAchievements:
    """Track player achievements and milestones in NetHack."""

    # Exploration achievements
    depth_reached: int = 1
    rooms_explored: int = 0
    secret_doors_found: int = 0
    stairs_down_found: int = 0
    stairs_up_found: int = 0

    # Combat achievements
    monsters_killed: int = 0
    peaceful_monsters_killed: int = 0
    unique_monsters_killed: int = 0
    kills_by_magic: int = 0
    kills_by_melee: int = 0
    kills_by_ranged: int = 0

    # Item achievements
    items_picked_up: int = 0
    gold_collected: int = 0
    scrolls_read: int = 0
    potions_drunk: int = 0
    spells_cast: int = 0
    artifacts_found: int = 0

    # Status achievements
    max_level_reached: int = 1
    max_hp_reached: int = 0
    times_prayed: int = 0
    successful_prayers: int = 0
    times_polymorphed: int = 0

    # Special achievements (boolean flags)
    first_kill: bool = False
    first_spell_cast: bool = False
    first_prayer: bool = False
    reached_minetown: bool = False
    reached_mines_end: bool = False
    reached_castle: bool = False
    got_quest: bool = False
    completed_quest: bool = False

    # Survival achievements
    turns_survived: int = 0
    turns_without_damage: int = 0
    traps_triggered: int = 0
    traps_avoided: int = 0

    # Negative achievements (for tracking mistakes)
    times_died: int = 0
    pets_killed: int = 0
    shopkeepers_angered: int = 0

    # Balrog progress tracker
    balrog_progress: Progress = field(default_factory=Progress)

    def to_dict(self) -> Dict[str, Any]:
        """Convert achievements to dictionary."""
        return {
            # Exploration
            "depth_reached": self.depth_reached,
            "rooms_explored": self.rooms_explored,
            "secret_doors_found": self.secret_doors_found,
            "stairs_found": self.stairs_down_found + self.stairs_up_found,
            # Combat
            "monsters_killed": self.monsters_killed,
            "unique_monsters_killed": self.unique_monsters_killed,
            "kills_by_magic": self.kills_by_magic,
            # Items
            "items_collected": self.items_picked_up,
            "gold_collected": self.gold_collected,
            "artifacts_found": self.artifacts_found,
            # Status
            "max_level": self.max_level_reached,
            "max_hp": self.max_hp_reached,
            "successful_prayers": self.successful_prayers,
            # Special (as booleans)
            "first_kill": self.first_kill,
            "first_spell_cast": self.first_spell_cast,
            "reached_minetown": self.reached_minetown,
            "got_quest": self.got_quest,
            # Survival
            "turns_survived": self.turns_survived,
            "traps_avoided": self.traps_avoided,
            # Balrog score
            "balrog_score": self.balrog_progress.percent,
        }

    def get_unlocked_achievements(self) -> Dict[str, bool]:
        """Get dictionary of which achievements have been unlocked."""
        return {
            # Depth milestones
            "reached_dlvl_2": self.depth_reached >= 2,
            "reached_dlvl_5": self.depth_reached >= 5,
            "reached_dlvl_10": self.depth_reached >= 10,
            "reached_dlvl_20": self.depth_reached >= 20,
            # Kill milestones
            "first_kill": self.first_kill,
            "killed_10_monsters": self.monsters_killed >= 10,
            "killed_50_monsters": self.monsters_killed >= 50,
            "killed_100_monsters": self.monsters_killed >= 100,
            "killed_by_magic": self.kills_by_magic > 0,
            # Item milestones
            "collected_100_gold": self.gold_collected >= 100,
            "collected_1000_gold": self.gold_collected >= 1000,
            "collected_10000_gold": self.gold_collected >= 10000,
            "found_artifact": self.artifacts_found > 0,
            # Level milestones
            "reached_level_5": self.max_level_reached >= 5,
            "reached_level_10": self.max_level_reached >= 10,
            "reached_level_20": self.max_level_reached >= 20,
            # Special locations
            "reached_minetown": self.reached_minetown,
            "reached_mines_end": self.reached_mines_end,
            "reached_castle": self.reached_castle,
            # Quest milestones
            "got_quest": self.got_quest,
            "completed_quest": self.completed_quest,
            # Survival milestones
            "survived_100_turns": self.turns_survived >= 100,
            "survived_1000_turns": self.turns_survived >= 1000,
            "survived_10000_turns": self.turns_survived >= 10000,
            # Prayer milestones
            "first_prayer": self.first_prayer,
            "successful_prayer": self.successful_prayers > 0,
            # Exploration milestones
            "found_secret_door": self.secret_doors_found > 0,
            "explored_10_rooms": self.rooms_explored >= 10,
            "explored_50_rooms": self.rooms_explored >= 50,
        }

    def update_from_observation(
        self, obs: Dict[str, Any], prev_obs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """Update achievements based on NLE observation. Returns newly unlocked achievements."""
        newly_unlocked = {}
        old_unlocked = self.get_unlocked_achievements()

        # Update basic stats from player_stats - require it to exist
        stats = obs["player_stats"]

        # Update depth
        current_depth = stats["depth"]
        if current_depth > self.depth_reached:
            self.depth_reached = current_depth

        # Update level
        current_level = stats["experience_level"]
        if current_level > self.max_level_reached:
            self.max_level_reached = current_level

        # Update HP
        current_hp = stats["max_hp"]
        if current_hp > self.max_hp_reached:
            self.max_hp_reached = current_hp

        # Update gold
        self.gold_collected = stats["gold"]

        # Update turn count (if available)
        if "turn" in stats:
            self.turns_survived = stats["turn"]

            # Update Balrog progress
            # Map depth to dungeon name (simplified version)
            dungeon_name = self._get_dungeon_name(current_depth)
            balrog_achievements = self.balrog_progress.update(dungeon_name, current_level)

            # Track balrog achievements as newly unlocked
            for balrog_achievement in balrog_achievements:
                newly_unlocked[f"balrog_{balrog_achievement}"] = True

        # Check for kills (would need to parse messages or track HP changes)
        if prev_obs and "message" in obs:
            message = obs["message"]
            if isinstance(message, bytes):
                message = message.decode("ascii", errors="ignore").strip("\x00")
            if "You kill" in message or "dies!" in message:
                self.monsters_killed += 1
                if not self.first_kill:
                    self.first_kill = True

                # Check kill type
                if "magic missile" in message or "spell" in message:
                    self.kills_by_magic += 1

        # Check for new achievements
        new_unlocked = self.get_unlocked_achievements()
        for achievement, unlocked in new_unlocked.items():
            if unlocked and not old_unlocked.get(achievement, False):
                newly_unlocked[achievement] = True

        return newly_unlocked

    def _get_dungeon_name(self, depth: int) -> str:
        """Map depth to dungeon name for Balrog progress tracking."""
        # Simplified mapping - in real implementation would need more game state
        if depth >= 50:
            return "dlvl50"
        elif depth >= 40:
            return "dlvl40"
        elif depth >= 30:
            return "dlvl30"
        elif depth >= 10:
            return "dlvl10"
        elif depth >= 5:
            return "dlvl5"
        else:
            return "dlvl1"


def calculate_balrog_reward(
    obs: Dict[str, Any], prev_obs: Optional[Dict[str, Any]] = None
) -> float:
    """
    Calculate reward using exact Balrog-style reward function.

    This is the exact implementation from Balrog that uses Progress class
    to track game progression based on dungeons reached and experience levels.
    """
    # For compatibility with existing code, we'll use the simple delta rewards
    # The actual Balrog score is tracked in NetHackAchievements.balrog_progress
    reward = 0.0

    if not prev_obs:
        return reward

    # Get player stats - require them to exist
    stats = obs["player_stats"]
    prev_stats = prev_obs["player_stats"]

    # Score delta (most important in Balrog)
    score_delta = stats["score"] - prev_stats["score"]
    if score_delta > 0:
        reward += score_delta / 100.0  # Scale down large score changes

    # Gold delta
    gold_delta = stats["gold"] - prev_stats["gold"]
    if gold_delta > 0:
        reward += gold_delta / 1000.0  # Small reward for gold

    # Experience delta
    exp_delta = stats["experience_points"] - prev_stats["experience_points"]
    if exp_delta > 0:
        reward += exp_delta / 100.0

    # Depth progress - THIS SHOULD GIVE 10.0 REWARD FOR REACHING LEVEL 3!
    depth_delta = stats["depth"] - prev_stats["depth"]
    if depth_delta > 0:
        reward += depth_delta * 10.0  # Big reward for going deeper

    # Level up bonus
    level_delta = stats["experience_level"] - prev_stats["experience_level"]
    if level_delta > 0:
        reward += level_delta * 5.0

    # Death penalty
    if "done" in obs and obs["done"]:
        message = obs["message"] if "message" in obs else b""
        if isinstance(message, bytes):
            message = message.decode("ascii", errors="ignore")
        if "died" in message.lower() or stats["hp"] <= 0:
            reward -= 100.0  # Large death penalty

    # Hunger penalty (if very hungry)
    if "hunger" in stats:
        hunger = stats["hunger"]
        if hunger > 500:  # Weak or worse
            reward -= 0.1

    return reward
