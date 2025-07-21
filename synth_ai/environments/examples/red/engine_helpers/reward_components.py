from synth_ai.environments.environment.rewards.core import RewardComponent
from typing import Dict, Any, Set


class BadgeRewardComponent(RewardComponent):
    """Reward for earning gym badges"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_badges = action.get("prev_badges", 0)
        current_badges = state["badges"]
        new_badges = current_badges & ~prev_badges
        badge_count = bin(new_badges).count("1")
        return badge_count * 1.0


class MapTransitionComponent(RewardComponent):
    """Reward for exploring new areas"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_map = action.get("prev_map_id", -1)
        current_map = state["map_id"]
        return 0.1 if current_map != prev_map else 0.0


class BattleVictoryComponent(RewardComponent):
    """Reward for winning battles"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state["in_battle"]
        battle_outcome = state["battle_outcome"]

        # Transitioning from battle to not in battle with victory
        if prev_in_battle and not current_in_battle and battle_outcome == 1:
            return 0.5
        return 0.0


class LevelUpComponent(RewardComponent):
    """Reward for Pokemon leveling up"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_level = action.get("prev_party_level", 0)
        current_level = state["party_level"]
        level_gain = max(0, current_level - prev_level)
        return level_gain * 0.3


class XPGainComponent(RewardComponent):
    """Small reward for XP gains"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_xp = action.get("prev_party_xp", 0)
        current_xp = state["party_xp"]
        xp_gain = max(0, current_xp - prev_xp)
        return xp_gain * 0.001  # Very small multiplier


class StepPenaltyComponent(RewardComponent):
    """Small penalty for each step to encourage efficiency"""

    def __init__(self, penalty: float = -0.001):
        self.penalty = penalty

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        return self.penalty


class MenuPenaltyComponent(RewardComponent):
    """Penalty for excessive menu usage"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # This would need more sophisticated menu tracking
        return 0.0


# ===== NEW EARLY GAME PALLET TOWN REWARDS =====


class ExitHouseReward(RewardComponent):
    """High reward for first time leaving the starting house - +2.0 points"""

    def __init__(self):
        self.house_exited = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.house_exited:
            return 0.0

        prev_map = action.get("prev_map_id", -1)
        current_map = state["map_id"]

        # Exit from house to town (assuming house maps are 1,2 and town is 0)
        if prev_map in [1, 2] and current_map == 0:
            self.house_exited = True
            return 2.0
        return 0.0


class NPCInteractionReward(RewardComponent):
    """Reward for talking to NPCs - +0.8 points per unique NPC"""

    def __init__(self):
        self.npcs_talked_to: Set[tuple] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Detect NPC conversations
        if state["text_box_active"] and not action.get("prev_text_box_active", False):
            # Use position as NPC identifier
            npc_key = (state["player_x"], state["player_y"], state["map_id"])
            if npc_key not in self.npcs_talked_to:
                self.npcs_talked_to.add(npc_key)
                return 0.8
        return 0.0


class OakLabDiscoveryReward(RewardComponent):
    """High reward for finding and entering Oak's lab - +2.5 points"""

    def __init__(self):
        self.lab_discovered = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.lab_discovered:
            return 0.0

        prev_map = action.get("prev_map_id", -1)
        current_map = state["map_id"]

        # Entering Oak's lab (assuming map 3)
        if prev_map == 0 and current_map == 3:
            self.lab_discovered = True
            return 2.5
        return 0.0


class StarterPokemonReward(RewardComponent):
    """Very high reward for getting first Pokemon - +10.0 points"""

    def __init__(self):
        self.starter_obtained = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.starter_obtained:
            return 0.0

        # Detect getting first Pokemon
        prev_party_count = len(action.get("prev_party", []))
        current_party_count = len(state.get("party", []))

        if prev_party_count == 0 and current_party_count == 1:
            if state["map_id"] == 3:  # In Oak's lab
                self.starter_obtained = True
                return 10.0
        return 0.0


class FirstBattleReward(RewardComponent):
    """High reward for engaging in first battle - +5.0 points"""

    def __init__(self):
        self.first_battle = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.first_battle:
            return 0.0

        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state["in_battle"]

        if not prev_in_battle and current_in_battle:
            self.first_battle = True
            return 5.0
        return 0.0


class DirectionExplorationReward(RewardComponent):
    """Reward for trying all movement directions - +1.0 points when complete"""

    def __init__(self):
        self.directions_tried: Set[str] = set()
        self.reward_given = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.reward_given:
            return 0.0

        # Track movement directions based on position changes
        prev_x = action.get("prev_player_x", state["player_x"])
        prev_y = action.get("prev_player_y", state["player_y"])
        current_x = state["player_x"]
        current_y = state["player_y"]

        if current_x > prev_x:
            self.directions_tried.add("RIGHT")
        elif current_x < prev_x:
            self.directions_tried.add("LEFT")
        elif current_y > prev_y:
            self.directions_tried.add("DOWN")
        elif current_y < prev_y:
            self.directions_tried.add("UP")

        if len(self.directions_tried) >= 4:
            self.reward_given = True
            return 1.0
        return 0.0


class BuildingExplorationReward(RewardComponent):
    """Reward for entering different buildings - +0.5 points per building"""

    def __init__(self):
        self.buildings_entered: Set[int] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_map = action.get("prev_map_id", -1)
        current_map = state["map_id"]

        # Entering a new building from town
        if (
            prev_map == 0 and current_map > 0 and current_map not in [1, 2]
        ):  # From town to new building
            if current_map not in self.buildings_entered:
                self.buildings_entered.add(current_map)
                return 0.5
        return 0.0


class ObjectInteractionReward(RewardComponent):
    """Reward for pressing A on various objects - +0.3 points per object"""

    def __init__(self):
        self.objects_interacted: Set[tuple] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Detect A button interactions that trigger text
        if state["text_box_active"] and not action.get("prev_text_box_active", False):
            object_key = (state["player_x"], state["player_y"], state["map_id"])
            if object_key not in self.objects_interacted:
                self.objects_interacted.add(object_key)
                return 0.3
        return 0.0


class TownExplorationReward(RewardComponent):
    """Reward for thorough town exploration - +0.1 per new position"""

    def __init__(self):
        self.positions_visited: Set[tuple] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if state["map_id"] == 0:  # In Pallet Town
            position_key = (state["player_x"], state["player_y"])
            if position_key not in self.positions_visited:
                self.positions_visited.add(position_key)
                return 0.1
        return 0.0


class RouteAttemptReward(RewardComponent):
    """Reward for trying to leave town (triggers story) - +3.0 points"""

    def __init__(self):
        self.route_attempted = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.route_attempted:
            return 0.0

        # Detect reaching the edge of Pallet Town (attempting to go north)
        if state["map_id"] == 0:  # In Pallet Town
            if state["player_y"] <= 1:  # At northern edge
                self.route_attempted = True
                return 3.0
        return 0.0
