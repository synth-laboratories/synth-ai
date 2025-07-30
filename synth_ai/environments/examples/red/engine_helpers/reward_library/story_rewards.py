"""
Story & Achievement Progression Reward Components

Rewards for major milestones, story gates, and achievements.
"""

from synth_ai.environments.environment.rewards.core import RewardComponent
from typing import Dict, Any, Set


class GymBadgeEarnedReward(RewardComponent):
    """Reward for earning gym badges - +150 points per badge (cumulative)"""

    def __init__(self):
        self.previous_badge_count = 0

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        current_badges = state.get("badges", 0)

        # Convert badge bitmask to count
        badge_count = bin(current_badges).count("1")

        if badge_count > self.previous_badge_count:
            new_badges = badge_count - self.previous_badge_count
            self.previous_badge_count = badge_count
            return new_badges * 150.0

        return 0.0


class HMAcquisitionReward(RewardComponent):
    """Reward for getting HMs - +75 points"""

    def __init__(self):
        self.hms_acquired: Set[int] = set()
        # HM item IDs (would be loaded from game data)
        self.hm_items = {200, 201, 202, 203, 204}  # Example HM IDs

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        inventory = state.get("inventory", [])
        total_reward = 0.0

        for item in inventory:
            item_id = item.get("item_id", 0)
            if item_id in self.hm_items and item_id not in self.hms_acquired:
                self.hms_acquired.add(item_id)
                total_reward += 75.0

        return total_reward


class EliteFourAccessReward(RewardComponent):
    """Reward for reaching Pokemon League - +300 points"""

    def __init__(self):
        self.elite_four_accessed = False
        self.elite_four_map = 100  # Pokemon League entrance

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.elite_four_accessed:
            return 0.0

        if state["map_id"] == self.elite_four_map:
            self.elite_four_accessed = True
            return 300.0

        return 0.0


class HallOfFameEntryReward(RewardComponent):
    """Reward for becoming Champion - +1000 points"""

    def __init__(self):
        self.hall_of_fame_entered = False
        self.hall_of_fame_map = 105  # Hall of Fame room

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.hall_of_fame_entered:
            return 0.0

        if state["map_id"] == self.hall_of_fame_map:
            self.hall_of_fame_entered = True
            return 1000.0

        return 0.0


class RivalBattleCompletionReward(RewardComponent):
    """Reward for each scripted rival encounter - +50 points"""

    def __init__(self):
        self.rival_battles_completed: Set[int] = set()
        # Rival battle locations
        self.rival_battle_maps = {3, 22, 25, 30}  # Oak's lab, Route 22, etc.

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state["in_battle"]
        battle_outcome = state.get("battle_outcome", 0)
        current_map = state["map_id"]

        # Completed rival battle
        if (
            prev_in_battle
            and not current_in_battle
            and battle_outcome == 1
            and current_map in self.rival_battle_maps
            and current_map not in self.rival_battles_completed
        ):
            self.rival_battles_completed.add(current_map)
            return 50.0

        return 0.0


class TeamRocketDefeatReward(RewardComponent):
    """Reward for each Team Rocket encounter - +40 points"""

    def __init__(self):
        self.rocket_encounters: Set[tuple] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # This would need Team Rocket battle detection
        # Placeholder implementation
        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state["in_battle"]
        battle_outcome = state.get("battle_outcome", 0)

        if prev_in_battle and not current_in_battle and battle_outcome == 1:
            # Check if in Team Rocket location
            rocket_maps = {50, 51, 52}  # Example Team Rocket hideout maps
            if state["map_id"] in rocket_maps:
                encounter_key = (state["player_x"], state["player_y"], state["map_id"])
                if encounter_key not in self.rocket_encounters:
                    self.rocket_encounters.add(encounter_key)
                    return 40.0

        return 0.0


class LegendaryEncounterReward(RewardComponent):
    """Reward for encountering legendary Pokemon - +200 points"""

    def __init__(self):
        self.legendary_encounters: Set[int] = set()
        self.legendary_maps = {60, 61, 62, 70}  # Legendary Pokemon locations

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        current_map = state["map_id"]

        if current_map in self.legendary_maps and current_map not in self.legendary_encounters:
            # Check if battle started (legendary encounter)
            prev_in_battle = action.get("prev_in_battle", False)
            current_in_battle = state["in_battle"]

            if not prev_in_battle and current_in_battle:
                self.legendary_encounters.add(current_map)
                return 200.0

        return 0.0


class SilphCoCompletionReward(RewardComponent):
    """Reward for completing major story dungeons - +100 points"""

    def __init__(self):
        self.silph_co_completed = False
        self.silph_co_maps = set(range(80, 90))  # Silph Co floors

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.silph_co_completed:
            return 0.0

        # Check if exiting Silph Co after completion
        prev_map = action.get("prev_map_id", -1)
        current_map = state["map_id"]

        if prev_map in self.silph_co_maps and current_map not in self.silph_co_maps:
            # Assume completion if leaving Silph Co
            self.silph_co_completed = True
            return 100.0

        return 0.0


class SafariZoneSuccessReward(RewardComponent):
    """Reward for successful Safari Zone runs - +30 points"""

    def __init__(self):
        self.safari_zone_runs = 0
        self.safari_zone_maps = {90, 91, 92, 93}  # Safari Zone areas

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Check if exiting Safari Zone with new Pokemon
        prev_map = action.get("prev_map_id", -1)
        current_map = state["map_id"]

        if prev_map in self.safari_zone_maps and current_map not in self.safari_zone_maps:
            # Check if Pokemon count increased
            prev_party_count = len(action.get("prev_party", []))
            current_party_count = len(state.get("party", []))

            if current_party_count > prev_party_count:
                return 30.0

        return 0.0


class GameCornerPrizesReward(RewardComponent):
    """Reward for earning significant Game Corner prizes - +20 points"""

    def __init__(self):
        self.game_corner_prizes: Set[int] = set()
        self.prize_items = {300, 301, 302}  # Game Corner prize item IDs

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        inventory = state.get("inventory", [])
        total_reward = 0.0

        for item in inventory:
            item_id = item.get("item_id", 0)
            if item_id in self.prize_items and item_id not in self.game_corner_prizes:
                self.game_corner_prizes.add(item_id)
                total_reward += 20.0

        return total_reward


class FossilRevivalReward(RewardComponent):
    """Reward for reviving fossils - +40 points"""

    def __init__(self):
        self.fossils_revived: Set[int] = set()
        self.fossil_pokemon = {138, 140, 142}  # Omanyte, Kabuto, Aerodactyl

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        party = state.get("party", [])
        total_reward = 0.0

        for pokemon in party:
            species_id = pokemon.get("species_id", 0)
            if species_id in self.fossil_pokemon and species_id not in self.fossils_revived:
                self.fossils_revived.add(species_id)
                total_reward += 40.0

        return total_reward
