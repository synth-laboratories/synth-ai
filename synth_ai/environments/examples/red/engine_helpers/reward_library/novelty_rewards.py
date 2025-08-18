"""
Novelty & Exploration Bonus Reward Components

Rewards for first-time experiences and curiosity.
"""

from typing import Any, Dict, Set

from synth_ai.environments.environment.rewards.core import RewardComponent


class FirstBattleReward(RewardComponent):
    """Reward for engaging in first battle - +20 points"""

    def __init__(self):
        self.first_battle = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.first_battle:
            return 0.0

        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state["in_battle"]

        if not prev_in_battle and current_in_battle:
            self.first_battle = True
            return 20.0
        return 0.0


class FirstPokemonCenterVisitReward(RewardComponent):
    """Reward for first healing - +15 points"""

    def __init__(self):
        self.first_heal = False
        self.pokemon_center_maps = {4, 8, 12, 16}  # Pokemon Center map IDs

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.first_heal:
            return 0.0

        if state["map_id"] in self.pokemon_center_maps:
            # Check for HP restoration (simplified)
            party = state.get("party", [])
            for pokemon in party:
                if pokemon.get("hp_current", 0) == pokemon.get("hp_max", 0):
                    self.first_heal = True
                    return 15.0
        return 0.0


class FirstPokemartPurchaseReward(RewardComponent):
    """Reward for first item purchase - +10 points"""

    def __init__(self):
        self.first_purchase = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.first_purchase:
            return 0.0

        prev_money = action.get("prev_money", 0)
        current_money = state.get("money", 0)

        if current_money < prev_money and prev_money > 0:
            self.first_purchase = True
            return 10.0
        return 0.0


class FirstSaveReward(RewardComponent):
    """Reward for saving the game - +5 points"""

    def __init__(self):
        self.first_save = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.first_save:
            return 0.0

        # This would need save detection
        if state.get("game_saved", False):
            self.first_save = True
            return 5.0
        return 0.0


class MenuExplorationReward(RewardComponent):
    """Reward for opening and exploring different menus - +3 points"""

    def __init__(self):
        self.menus_explored: Set[str] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # This would need menu state tracking
        # Placeholder implementation
        return 0.0


class ButtonDiscoveryReward(RewardComponent):
    """Reward for discovering START menu, SELECT uses - +5 points"""

    def __init__(self):
        self.buttons_discovered: Set[str] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # This would need button usage tracking
        # Placeholder implementation
        return 0.0


class FeatureDiscoveryReward(RewardComponent):
    """Reward for discovering PC, daycare, etc. - +10 points"""

    def __init__(self):
        self.features_discovered: Set[str] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # This would need feature usage detection
        # Placeholder implementation
        return 0.0
