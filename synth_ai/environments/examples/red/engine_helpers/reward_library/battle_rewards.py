"""
Battle & Combat Reward Components

Rewards for battle victories, combat strategy, and battle efficiency.
"""

from synth_ai.environments.environment.rewards.core import RewardComponent
from typing import Dict, Any, Set


class WildPokemonDefeatedReward(RewardComponent):
    """Reward for defeating wild Pokemon - +3 points per defeat"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state["in_battle"]
        battle_outcome = state.get("battle_outcome", 0)

        # Exiting battle with victory (outcome = 1)
        if prev_in_battle and not current_in_battle and battle_outcome == 1:
            # Check if it was a wild Pokemon battle (no trainer)
            # This would need additional state to distinguish wild vs trainer battles
            return 3.0
        return 0.0


class TrainerBattleVictoryReward(RewardComponent):
    """Reward for defeating trainers - +15 points"""

    def __init__(self):
        self.trainers_defeated: Set[tuple] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state["in_battle"]
        battle_outcome = state.get("battle_outcome", 0)

        # Exiting battle with victory
        if prev_in_battle and not current_in_battle and battle_outcome == 1:
            # Use location as trainer identifier
            trainer_key = (state["player_x"], state["player_y"], state["map_id"])
            if trainer_key not in self.trainers_defeated:
                self.trainers_defeated.add(trainer_key)
                return 15.0
        return 0.0


class GymLeaderVictoryReward(RewardComponent):
    """Reward for defeating gym leaders - +100 points"""

    def __init__(self):
        self.gym_leaders_defeated: Set[int] = set()
        # Gym map IDs (would be loaded from game data)
        self.gym_maps = {20, 21, 22, 23, 24, 25, 26, 27}

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state["in_battle"]
        battle_outcome = state.get("battle_outcome", 0)
        current_map = state["map_id"]

        # Victory in a gym
        if (
            prev_in_battle
            and not current_in_battle
            and battle_outcome == 1
            and current_map in self.gym_maps
        ):
            if current_map not in self.gym_leaders_defeated:
                self.gym_leaders_defeated.add(current_map)
                return 100.0
        return 0.0


class EliteFourMemberVictoryReward(RewardComponent):
    """Reward for defeating Elite Four members - +200 points each"""

    def __init__(self):
        self.elite_four_defeated: Set[int] = set()
        # Elite Four room IDs (would be loaded from game data)
        self.elite_four_maps = {100, 101, 102, 103}

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state["in_battle"]
        battle_outcome = state.get("battle_outcome", 0)
        current_map = state["map_id"]

        # Victory against Elite Four
        if (
            prev_in_battle
            and not current_in_battle
            and battle_outcome == 1
            and current_map in self.elite_four_maps
        ):
            if current_map not in self.elite_four_defeated:
                self.elite_four_defeated.add(current_map)
                return 200.0
        return 0.0


class ChampionVictoryReward(RewardComponent):
    """Reward for defeating the Champion - +500 points"""

    def __init__(self):
        self.champion_defeated = False
        self.champion_map = 104  # Champion room ID

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.champion_defeated:
            return 0.0

        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state["in_battle"]
        battle_outcome = state.get("battle_outcome", 0)
        current_map = state["map_id"]

        # Victory against Champion
        if (
            prev_in_battle
            and not current_in_battle
            and battle_outcome == 1
            and current_map == self.champion_map
        ):
            self.champion_defeated = True
            return 500.0
        return 0.0


class TypeAdvantageUsageReward(RewardComponent):
    """Reward for using super effective moves - +5 points"""

    def __init__(self):
        self.super_effective_count = 0

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # This would need move effectiveness tracking
        # Placeholder implementation - would need battle log analysis
        if state["in_battle"]:
            # Simplified: assume some moves are super effective
            # Real implementation would track move types vs opponent types
            move_used = action.get("move_used")
            opponent_type = action.get("opponent_type")

            if move_used and opponent_type:
                if self._is_super_effective(move_used, opponent_type):
                    return 5.0
        return 0.0

    def _is_super_effective(self, move_type: str, opponent_type: str) -> bool:
        """Check if move is super effective against opponent"""
        # Simplified type effectiveness chart
        effectiveness = {
            ("water", "fire"): True,
            ("fire", "grass"): True,
            ("grass", "water"): True,
            ("electric", "water"): True,
            # Add more type matchups
        }
        return effectiveness.get((move_type, opponent_type), False)


class CriticalHitReward(RewardComponent):
    """Reward for landing critical hits - +3 points"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # This would need battle log analysis to detect critical hits
        # Placeholder implementation
        if state["in_battle"]:
            critical_hit = action.get("critical_hit", False)
            if critical_hit:
                return 3.0
        return 0.0


class StatusEffectUsageReward(RewardComponent):
    """Reward for successfully applying status effects - +5 points"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # This would need status effect tracking
        # Placeholder implementation
        if state["in_battle"]:
            status_applied = action.get("status_applied")
            if status_applied in ["paralysis", "poison", "sleep", "burn", "freeze"]:
                return 5.0
        return 0.0


class OHKOReward(RewardComponent):
    """Reward for one-shot defeats - +10 points"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # This would need damage tracking to detect OHKO
        # Placeholder implementation
        if state["in_battle"]:
            opponent_defeated = action.get("opponent_defeated", False)
            damage_dealt = action.get("damage_dealt", 0)
            opponent_max_hp = action.get("opponent_max_hp", 100)

            # OHKO if damage equals or exceeds max HP
            if opponent_defeated and damage_dealt >= opponent_max_hp:
                return 10.0
        return 0.0


class FlawlessVictoryReward(RewardComponent):
    """Reward for winning without taking damage - +20 points"""

    def __init__(self):
        self.battle_start_hp: Dict[int, int] = {}  # Track HP at battle start

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state["in_battle"]
        battle_outcome = state.get("battle_outcome", 0)

        # Track battle start
        if not prev_in_battle and current_in_battle:
            # Battle started - record current HP
            party = state.get("party", [])
            for i, pokemon in enumerate(party):
                self.battle_start_hp[i] = pokemon.get("hp_current", 0)

        # Check for flawless victory
        elif prev_in_battle and not current_in_battle and battle_outcome == 1:
            # Battle ended in victory - check if HP unchanged
            party = state.get("party", [])
            flawless = True
            for i, pokemon in enumerate(party):
                start_hp = self.battle_start_hp.get(i, 0)
                current_hp = pokemon.get("hp_current", 0)
                if current_hp < start_hp:
                    flawless = False
                    break

            # Clear battle HP tracking
            self.battle_start_hp.clear()

            if flawless:
                return 20.0

        return 0.0


class UnderleveledVictoryReward(RewardComponent):
    """Reward for winning with significantly lower level Pokemon - +25 points"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state["in_battle"]
        battle_outcome = state.get("battle_outcome", 0)

        # Victory with level disadvantage
        if prev_in_battle and not current_in_battle and battle_outcome == 1:
            player_level = action.get("player_pokemon_level", 0)
            opponent_level = action.get("opponent_pokemon_level", 0)

            # Reward if player Pokemon is 5+ levels lower
            if opponent_level - player_level >= 5:
                return 25.0
        return 0.0


class BattleStreakReward(RewardComponent):
    """Reward for consecutive battle wins - +5 points per battle in streak"""

    def __init__(self):
        self.current_streak = 0

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state["in_battle"]
        battle_outcome = state.get("battle_outcome", 0)

        # Battle ended
        if prev_in_battle and not current_in_battle:
            if battle_outcome == 1:  # Victory
                self.current_streak += 1
                return 5.0
            else:  # Loss or other outcome
                self.current_streak = 0

        return 0.0
