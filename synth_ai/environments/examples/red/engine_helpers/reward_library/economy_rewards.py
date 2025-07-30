"""
Economy & Resource Management Reward Components

Rewards for money management and inventory optimization.
"""

from synth_ai.environments.environment.rewards.core import RewardComponent
from typing import Dict, Any, Set


class FirstEarningsReward(RewardComponent):
    """Reward for earning first money from battles - +10 points"""

    def __init__(self):
        self.first_earnings = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.first_earnings:
            return 0.0

        prev_money = action.get("prev_money", 0)
        current_money = state.get("money", 0)

        if current_money > prev_money and prev_money == 0:
            self.first_earnings = True
            return 10.0
        return 0.0


class WealthMilestonesReward(RewardComponent):
    """Reward for reaching money milestones - +25 points"""

    def __init__(self):
        self.milestones_reached: Set[int] = set()
        self.milestones = [1000, 5000, 10000, 50000]

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        current_money = state.get("money", 0)
        total_reward = 0.0

        for milestone in self.milestones:
            if current_money >= milestone and milestone not in self.milestones_reached:
                self.milestones_reached.add(milestone)
                total_reward += 25.0

        return total_reward


class SmartPurchasesReward(RewardComponent):
    """Reward for buying useful items - +10 points"""

    def __init__(self):
        self.useful_items = {4, 5, 6, 10, 11, 12}  # Pokeballs, Potions

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_inventory = action.get("prev_inventory", [])
        current_inventory = state.get("inventory", [])
        prev_money = action.get("prev_money", 0)
        current_money = state.get("money", 0)

        # Money decreased (purchase made)
        if current_money < prev_money:
            prev_items = {item.get("item_id") for item in prev_inventory}
            current_items = {item.get("item_id") for item in current_inventory}
            new_items = current_items - prev_items

            for item_id in new_items:
                if item_id in self.useful_items:
                    return 10.0
        return 0.0


class RarePurchaseReward(RewardComponent):
    """Reward for buying expensive items - +20 points"""

    def __init__(self):
        self.expensive_items = {50, 51, 52}  # TMs, evolution stones

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_inventory = action.get("prev_inventory", [])
        current_inventory = state.get("inventory", [])
        prev_money = action.get("prev_money", 0)
        current_money = state.get("money", 0)

        # Money decreased significantly (expensive purchase)
        if current_money < prev_money - 1000:
            prev_items = {item.get("item_id") for item in prev_inventory}
            current_items = {item.get("item_id") for item in current_inventory}
            new_items = current_items - prev_items

            for item_id in new_items:
                if item_id in self.expensive_items:
                    return 20.0
        return 0.0


class InventoryOrganizationReward(RewardComponent):
    """Reward for effective bag management - +5 points"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # This would need inventory management tracking
        # Placeholder implementation
        return 0.0


class HealingItemUsageReward(RewardComponent):
    """Reward for timely use of potions/healing items - +3 points"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Check for healing item usage when Pokemon HP is low
        party = state.get("party", [])
        prev_inventory = action.get("prev_inventory", [])
        current_inventory = state.get("inventory", [])

        # Item count decreased (item used)
        if len(current_inventory) < len(prev_inventory):
            for pokemon in party:
                hp_percentage = pokemon.get("hp_current", 0) / max(pokemon.get("hp_max", 1), 1)
                if hp_percentage < 0.5:  # Low HP
                    return 3.0
        return 0.0


class PokeballEfficiencyReward(RewardComponent):
    """Reward for successful captures - +5 points"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Check for Pokemon capture (party size increase + pokeball usage)
        prev_party_count = len(action.get("prev_party", []))
        current_party_count = len(state.get("party", []))
        prev_inventory = action.get("prev_inventory", [])
        current_inventory = state.get("inventory", [])

        # Pokemon captured and pokeball used
        if current_party_count > prev_party_count and len(current_inventory) < len(prev_inventory):
            return 5.0
        return 0.0
