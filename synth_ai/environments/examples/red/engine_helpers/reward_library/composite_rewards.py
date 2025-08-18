"""
Composite & Milestone Reward Components

Rewards for achievement combinations and progression streaks.
"""

from typing import Any, Dict, Set

from synth_ai.environments.environment.rewards.core import RewardComponent


class PerfectGymRunReward(RewardComponent):
    """Reward for defeating gym without losing any Pokemon - +200 points"""

    def __init__(self):
        self.gym_maps = {20, 21, 22, 23, 24, 25, 26, 27}
        self.perfect_gyms: Set[int] = set()
        self.gym_start_party_state: Dict[int, list] = {}

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        current_map = state["map_id"]

        if current_map in self.gym_maps:
            # Track gym entry
            prev_map = action.get("prev_map_id", -1)
            if prev_map not in self.gym_maps:
                # Entering gym - record party state
                self.gym_start_party_state[current_map] = state.get("party", [])

            # Check for gym completion
            prev_in_battle = action.get("prev_in_battle", False)
            current_in_battle = state["in_battle"]
            battle_outcome = state.get("battle_outcome", 0)

            if (
                prev_in_battle
                and not current_in_battle
                and battle_outcome == 1
                and current_map not in self.perfect_gyms
            ):
                # Gym leader defeated - check if perfect run
                start_party = self.gym_start_party_state.get(current_map, [])
                current_party = state.get("party", [])

                # Check if all Pokemon maintained their HP
                perfect = True
                for i, (start_pkmn, current_pkmn) in enumerate(zip(start_party, current_party)):
                    if current_pkmn.get("hp_current", 0) < start_pkmn.get("hp_current", 0):
                        perfect = False
                        break

                if perfect:
                    self.perfect_gyms.add(current_map)
                    return 200.0

        return 0.0


class AreaMasteryReward(RewardComponent):
    """Reward for full area completion - +100 points"""

    def __init__(self):
        self.mastered_areas: Set[int] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Placeholder - would need comprehensive area tracking
        return 0.0


class SpeedrunMilestonesReward(RewardComponent):
    """Reward for reaching story points within time limits - +50 points"""

    def __init__(self):
        self.milestones_reached: Set[str] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Placeholder - would need time tracking
        return 0.0


class ExplorationStreakReward(RewardComponent):
    """Reward for consecutive new area discoveries - +2 points per consecutive area"""

    def __init__(self):
        self.streak = 0
        self.last_area = -1

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        current_map = state["map_id"]
        prev_map = action.get("prev_map_id", -1)

        if current_map != prev_map and current_map != self.last_area:
            # New area discovered
            self.streak += 1
            self.last_area = current_map
            return 2.0 * self.streak
        elif current_map == prev_map:
            # Stayed in same area - reset streak
            self.streak = 0

        return 0.0


class BattleWinStreakReward(RewardComponent):
    """Reward for consecutive battle wins - +3 points per consecutive win"""

    def __init__(self):
        self.win_streak = 0

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state["in_battle"]
        battle_outcome = state.get("battle_outcome", 0)

        if prev_in_battle and not current_in_battle:
            if battle_outcome == 1:  # Victory
                self.win_streak += 1
                return 3.0 * self.win_streak
            else:  # Loss
                self.win_streak = 0

        return 0.0


class PerfectDayReward(RewardComponent):
    """Reward for a session with no Pokemon fainting - +100 points"""

    def __init__(self):
        self.perfect_day_achieved = False
        self.any_pokemon_fainted = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.perfect_day_achieved:
            return 0.0

        # Check if any Pokemon fainted
        party = state.get("party", [])
        for pokemon in party:
            if pokemon.get("hp_current", 1) == 0:
                self.any_pokemon_fainted = True
                break

        # Check for end of session (would need session detection)
        # Placeholder implementation
        step_count = state.get("step_count", 0)
        if step_count >= 100 and not self.any_pokemon_fainted:  # Example session length
            self.perfect_day_achieved = True
            return 100.0

        return 0.0
