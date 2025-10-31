from typing import Any, Dict, Set

from synth_ai.environments.environment.rewards.core import RewardComponent


# ===== COMPREHENSIVE POKEMON RED PROGRESS REWARD SYSTEM =====
# Designed for deterministic rewards that guide toward beating Brock at Pewter Gym


class RouteExplorationReward(RewardComponent):
    """High rewards for reaching key areas on the path to Pewter Gym - guides exploration"""

    def __init__(self):
        self.key_areas_reached: Set[int] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        current_map = state["map_id"]
        prev_map = action.get("prev_map_id", -1)

        # Key maps and rewards for progressing toward Pewter Gym
        area_rewards = {
            0: 0.0,  # Pallet Town (starting point)
            1: 2.0,  # Route 1 - First step out of town (+2.0)
            2: 1.5,  # Viridian City - Major hub (+1.5)
            3: 1.0,  # Route 22 - Path to League (+1.0)
            4: 1.0,  # Route 2 - To Viridian Forest (+1.0)
            5: 2.0,  # Viridian Forest - Dense area (+2.0)
            6: 1.5,  # Pewter City - Target city (+1.5)
            7: 5.0,  # Pewter Gym - GOAL AREA (+5.0 for entering gym)
        }

        if current_map in area_rewards and current_map not in self.key_areas_reached:
            if prev_map != current_map:  # Only reward when actually entering new area
                self.key_areas_reached.add(current_map)
                return area_rewards[current_map]

        return 0.0


class StrategicTrainingReward(RewardComponent):
    """Rewards for building Pokemon strength strategically"""

    def __init__(self):
        self.level_milestones: Set[int] = set()
        self.last_level = 0

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        current_level = state.get("party_level", 0)
        prev_level = action.get("prev_party_level", 0)

        # Reward reaching key level milestones
        milestone_rewards = {
            8: 1.0,   # Level 8 - Good for early battles
            12: 2.0,  # Level 12 - Ready for Brock
            15: 3.0,  # Level 15 - Strong Pokemon
        }

        if current_level > prev_level and current_level in milestone_rewards:
            if current_level not in self.level_milestones:
                self.level_milestones.add(current_level)
                return milestone_rewards[current_level]

        # Small reward for any level up (0.2 points)
        if current_level > prev_level:
            return 0.2

        return 0.0


class BattleProgressionReward(RewardComponent):
    """Rewards for winning battles and gaining experience"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state.get("in_battle", False)
        battle_outcome = state.get("battle_outcome", 0)

        # Large reward for battle victory (+1.0)
        if prev_in_battle and not current_in_battle and battle_outcome == 1:
            return 1.0

        # Small reward for entering battle (+0.1) - shows engagement
        if not prev_in_battle and current_in_battle:
            return 0.1

        return 0.0


class GymPreparationReward(RewardComponent):
    """Rewards for preparing to challenge Brock"""

    def __init__(self):
        self.prepared_for_gym = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.prepared_for_gym:
            return 0.0

        # Check if in Pewter City area and have decent Pokemon
        if state["map_id"] in [6, 7]:  # Pewter City or Gym
            party_level = state.get("party_level", 0)
            party_count = len(state.get("party", []))

            # Reward being prepared for gym battle
            if party_level >= 10 and party_count >= 1:
                self.prepared_for_gym = True
                return 3.0  # Significant reward for being gym-ready

        return 0.0


class ItemCollectionReward(RewardComponent):
    """Rewards for collecting useful items"""

    def __init__(self):
        self.items_collected: Set[int] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_inventory = action.get("prev_inventory", [])
        current_inventory = state.get("inventory", [])

        # Check for new items
        prev_item_ids = {item["item_id"] for item in prev_inventory}
        current_item_ids = {item["item_id"] for item in current_inventory}

        new_items = current_item_ids - prev_item_ids

        # Reward valuable items for gym preparation
        valuable_items = {1, 2, 3, 4, 5, 10, 11, 12, 13}  # Potions, Balls, etc.
        reward = 0.0

        for item_id in new_items:
            if item_id not in self.items_collected:
                self.items_collected.add(item_id)
                if item_id in valuable_items:
                    reward += 0.5  # +0.5 per valuable item
                else:
                    reward += 0.1  # +0.1 per other item

        return reward


class HealingManagementReward(RewardComponent):
    """Rewards for keeping Pokemon healthy"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_party = action.get("prev_party", [])
        current_party = state.get("party", [])

        if not prev_party or not current_party:
            return 0.0

        # Reward healing Pokemon back to full health
        prev_hp_pct = sum(p.get("hp_percentage", 0) for p in prev_party) / len(prev_party)
        current_hp_pct = sum(p.get("hp_percentage", 0) for p in current_party) / len(current_party)

        # Significant improvement in health
        if current_hp_pct > prev_hp_pct + 20:  # Healed at least 20% overall
            return 0.8

        # Small reward for maintaining good health
        if current_hp_pct >= 80 and prev_hp_pct >= 80:
            return 0.05

        return 0.0


class EfficientExplorationReward(RewardComponent):
    """Rewards for exploring efficiently without getting lost"""

    def __init__(self):
        self.positions_visited: Set[tuple] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Track unique positions visited in each map
        position_key = (state["map_id"], state["player_x"], state["player_y"])

        if position_key not in self.positions_visited:
            self.positions_visited.add(position_key)
            return 0.02  # Small reward for discovering new areas

        return 0.0


class BadgeVictoryReward(RewardComponent):
    """HUGE reward for achieving the main goal - Boulder Badge"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_badges = action.get("prev_badges", 0)
        current_badges = state.get("badges", 0)

        # Check if Boulder Badge (bit 0) was newly earned
        boulder_badge_mask = 0x01
        prev_has_badge = prev_badges & boulder_badge_mask
        current_has_badge = current_badges & boulder_badge_mask

        if not prev_has_badge and current_has_badge:
            return 50.0  # MASSIVE reward for completing the main objective

        return 0.0


class StepPenaltyComponent(RewardComponent):
    """Small penalty for each step to encourage efficiency"""

    def __init__(self, penalty: float = 0.0):  # Changed from -0.005 to 0.0
        self.penalty = penalty

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        return self.penalty


# ===== LEGACY COMPONENTS (kept for compatibility) =====


class BadgeRewardComponent(RewardComponent):
    """Legacy badge reward - now handled by BadgeVictoryReward"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        return 0.0  # Handled by BadgeVictoryReward


class MapTransitionComponent(RewardComponent):
    """Legacy map transition - now handled by RouteExplorationReward"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        return 0.0  # Handled by RouteExplorationReward


class BattleVictoryComponent(RewardComponent):
    """Legacy battle victory - now handled by BattleProgressionReward"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        return 0.0  # Handled by BattleProgressionReward


class LevelUpComponent(RewardComponent):
    """Legacy level up - now handled by StrategicTrainingReward"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        return 0.0  # Handled by StrategicTrainingReward


class XPGainComponent(RewardComponent):
    """Legacy XP gain - now handled by StrategicTrainingReward"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        return 0.0  # Handled by StrategicTrainingReward
