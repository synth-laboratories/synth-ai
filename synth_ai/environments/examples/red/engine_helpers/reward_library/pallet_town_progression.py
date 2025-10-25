"""
Ultra-Rich Reward Shaping for Pallet Town First Section

This module provides fine-grained reward components that track important
achievements in the initial Pallet Town sequence: leaving the house, finding
Oak's lab, talking to Oak, starting the rival battle, attacking and damaging
the opponent, winning the battle, getting a party member, and leaving the lab.

Each milestone is carefully weighted to provide dense, meaningful feedback
for reinforcement learning agents learning to play Pokemon Red.
"""

from typing import Any, Dict

from synth_ai.environments.environment.rewards.core import RewardComponent


class LeaveBedroomReward(RewardComponent):
    """
    Reward for going downstairs from bedroom to main floor.
    This is the first meaningful action in the game.
    
    Reward: +20 points (one-time)
    """

    def __init__(self):
        self.triggered = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.triggered:
            return 0.0

        prev_map = action.get("prev_map_id", -1)
        current_map = state.get("map_id", -1)

        # Detect moving from bedroom (map 38/0x26) to downstairs (map 37/0x25)
        # In Red's house, bedroom is map 38, downstairs is map 37
        if prev_map == 38 and current_map == 37:
            self.triggered = True
            return 20.0
        return 0.0


class ExitHouseFirstTimeReward(RewardComponent):
    """
    Reward for leaving the starting house and entering Pallet Town.
    This is a major milestone showing the agent understands doors.
    
    Reward: +30 points (one-time)
    """

    def __init__(self):
        self.triggered = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.triggered:
            return 0.0

        prev_map = action.get("prev_map_id", -1)
        current_map = state.get("map_id", -1)

        # Exit from house (map 37) to Pallet Town (likely map 0-36 range)
        # Detect leaving house interior to outdoor area
        if prev_map == 37 and current_map != 37 and current_map != 38:
            self.triggered = True
            return 30.0
        return 0.0


class FindOakLabReward(RewardComponent):
    """
    Reward for discovering and entering Oak's Lab for the first time.
    This shows the agent can navigate the town and find the lab.
    
    Reward: +40 points (one-time)
    """

    def __init__(self):
        self.lab_found = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.lab_found:
            return 0.0

        prev_map = action.get("prev_map_id", -1)
        current_map = state.get("map_id", -1)

        # Entering Oak's lab (map 3/0x03) from Pallet Town (map 0)
        if prev_map == 0 and current_map == 3:
            self.lab_found = True
            return 40.0
        return 0.0


class TalkToOakReward(RewardComponent):
    """
    Reward for first conversation with Professor Oak in the lab.
    This is detected by text box activation in Oak's lab.
    
    Reward: +50 points (one-time)
    """

    def __init__(self):
        self.oak_talked_to = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.oak_talked_to:
            return 0.0

        # Detect first dialogue in Oak's lab
        if state.get("map_id", -1) == 3 and state.get("text_box_active", False):
            prev_text_active = action.get("prev_text_box_active", False)
            if not prev_text_active:
                self.oak_talked_to = True
                return 50.0
        return 0.0


class ReceiveStarterPokemonReward(RewardComponent):
    """
    Reward for receiving your first Pokemon from Oak.
    This is a major story milestone detected by party count changing from 0 to 1.
    
    Reward: +100 points (one-time)
    """

    def __init__(self):
        self.starter_received = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.starter_received:
            return 0.0

        # Detect receiving first Pokemon
        prev_party_count = action.get("prev_party_count", 0)
        current_party_count = state.get("party_count", 0)

        if prev_party_count == 0 and current_party_count == 1:
            # Verify we're in Oak's lab
            if state.get("map_id", -1) == 3:
                self.starter_received = True
                return 100.0
        return 0.0


class EnterFirstBattleReward(RewardComponent):
    """
    Reward for entering the first rival battle in Oak's lab.
    This shows the agent has progressed through dialogue and triggered the battle.
    
    Reward: +75 points (one-time)
    """

    def __init__(self):
        self.first_battle_entered = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.first_battle_entered:
            return 0.0

        # Detect entering battle for the first time
        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state.get("in_battle", False)

        if not prev_in_battle and current_in_battle:
            # Verify we're in Oak's lab (rival battle)
            if state.get("map_id", -1) == 3:
                self.first_battle_entered = True
                return 75.0
        return 0.0


class DealDamageToRivalReward(RewardComponent):
    """
    Reward for successfully attacking and damaging the rival's Pokemon.
    This is detected by a decrease in enemy HP during battle.
    
    Reward: +5 points per HP damage dealt (cumulative, capped at 10 total rewards)
    """

    def __init__(self):
        self.damage_instances = 0
        self.max_instances = 10
        self.prev_enemy_hp = None

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.damage_instances >= self.max_instances:
            return 0.0

        # Track damage during battle
        if state.get("in_battle", False):
            current_enemy_hp = state.get("enemy_hp_current", 0)
            prev_enemy_hp = action.get("prev_enemy_hp_current", current_enemy_hp)

            # Detect HP decrease (damage dealt)
            if prev_enemy_hp > current_enemy_hp > 0:
                damage = prev_enemy_hp - current_enemy_hp
                self.damage_instances += 1
                # Give +5 points per instance of damage
                return 5.0

        return 0.0


class ReduceEnemyHPByHalfReward(RewardComponent):
    """
    Reward for reducing enemy HP below 50% for the first time.
    This shows the agent is making significant progress in battle.
    
    Reward: +25 points (one-time)
    """

    def __init__(self):
        self.half_hp_achieved = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.half_hp_achieved:
            return 0.0

        if state.get("in_battle", False):
            enemy_hp_pct = state.get("enemy_hp_percentage", 0.0)
            prev_enemy_hp_pct = action.get("prev_enemy_hp_percentage", 100.0)

            # Detect crossing below 50% threshold
            if prev_enemy_hp_pct >= 50.0 and enemy_hp_pct < 50.0:
                self.half_hp_achieved = True
                return 25.0

        return 0.0


class ReduceEnemyHPToLowReward(RewardComponent):
    """
    Reward for reducing enemy HP below 25% (critical/low HP).
    This shows the agent is close to winning the battle.
    
    Reward: +35 points (one-time)
    """

    def __init__(self):
        self.low_hp_achieved = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.low_hp_achieved:
            return 0.0

        if state.get("in_battle", False):
            enemy_hp_pct = state.get("enemy_hp_percentage", 0.0)
            prev_enemy_hp_pct = action.get("prev_enemy_hp_percentage", 100.0)

            # Detect crossing below 25% threshold
            if prev_enemy_hp_pct >= 25.0 and enemy_hp_pct < 25.0:
                self.low_hp_achieved = True
                return 35.0

        return 0.0


class WinFirstBattleReward(RewardComponent):
    """
    Reward for winning the first battle against the rival.
    This is the culmination of the battle sequence.
    
    Reward: +150 points (one-time)
    """

    def __init__(self):
        self.first_battle_won = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.first_battle_won:
            return 0.0

        # Detect winning a battle (transition from in_battle to not in_battle with win outcome)
        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state.get("in_battle", False)
        battle_outcome = state.get("battle_outcome", 0)

        # battle_outcome: 0=ongoing, 1=win, 2=lose
        if prev_in_battle and not current_in_battle and battle_outcome == 1:
            # Verify it's in Oak's lab (the rival battle)
            if state.get("map_id", -1) == 3:
                self.first_battle_won = True
                return 150.0

        return 0.0


class ExitLabAfterBattleReward(RewardComponent):
    """
    Reward for leaving Oak's Lab after receiving Pokemon and winning the battle.
    This completes the initial Pallet Town sequence.
    
    Reward: +60 points (one-time, requires having a party member)
    """

    def __init__(self):
        self.exited_with_pokemon = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.exited_with_pokemon:
            return 0.0

        prev_map = action.get("prev_map_id", -1)
        current_map = state.get("map_id", -1)

        # Exit from lab (map 3) to town (map 0)
        if prev_map == 3 and current_map == 0:
            # Verify we have at least one Pokemon
            if state.get("party_count", 0) > 0:
                self.exited_with_pokemon = True
                return 60.0

        return 0.0


class FirstBattleEfficiencyReward(RewardComponent):
    """
    Reward for winning the first battle efficiently (in fewer turns).
    Encourages the agent to learn optimal battle strategies early.
    
    Reward: +20 points if won in ≤5 turns, +10 if ≤8 turns
    """

    def __init__(self):
        self.efficiency_rewarded = False
        self.max_turns_seen = 0

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.efficiency_rewarded:
            return 0.0

        # Track turn count during battle
        if state.get("in_battle", False):
            self.max_turns_seen = max(self.max_turns_seen, state.get("battle_turn", 0))

        # When battle ends with a win, assess efficiency
        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state.get("in_battle", False)
        battle_outcome = state.get("battle_outcome", 0)

        if prev_in_battle and not current_in_battle and battle_outcome == 1:
            if state.get("map_id", -1) == 3:  # Rival battle in lab
                self.efficiency_rewarded = True
                if self.max_turns_seen <= 5:
                    return 20.0
                elif self.max_turns_seen <= 8:
                    return 10.0

        return 0.0


class KeepPokemonHealthyReward(RewardComponent):
    """
    Reward for keeping your Pokemon's HP above 50% during the first battle.
    Encourages defensive play and resource management.
    
    Reward: +30 points (one-time, checked at end of first battle)
    """

    def __init__(self):
        self.health_bonus_given = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.health_bonus_given:
            return 0.0

        # Check health status when battle ends
        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state.get("in_battle", False)
        battle_outcome = state.get("battle_outcome", 0)

        if prev_in_battle and not current_in_battle and battle_outcome == 1:
            if state.get("map_id", -1) == 3:  # Rival battle in lab
                # Check if first Pokemon has >50% HP
                if len(state.get("party_pokemon", [])) > 0:
                    first_pokemon = state.get("party_pokemon", [])[0]
                    hp_pct = first_pokemon.get("hp_percentage", 0)
                    if hp_pct > 50.0:
                        self.health_bonus_given = True
                        return 30.0

        return 0.0


class NavigationSpeedReward(RewardComponent):
    """
    Reward for completing the Pallet Town sequence quickly (by step count).
    Encourages efficient navigation and minimal wandering.
    
    Reward: Scales based on step count (fewer steps = higher reward)
    """

    def __init__(self):
        self.step_count = 0
        self.sequence_complete = False
        self.reward_given = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.reward_given:
            return 0.0

        # Track steps
        self.step_count += 1

        # Check if sequence is complete (exited lab with Pokemon after battle)
        prev_map = action.get("prev_map_id", -1)
        current_map = state.get("map_id", -1)

        if prev_map == 3 and current_map == 0:  # Exiting lab
            if state.get("party_count", 0) > 0:  # Have Pokemon
                self.sequence_complete = True
                self.reward_given = True

                # Award points based on efficiency
                # Optimal path is roughly 30-40 steps
                if self.step_count <= 40:
                    return 50.0  # Very efficient
                elif self.step_count <= 60:
                    return 30.0  # Good
                elif self.step_count <= 80:
                    return 15.0  # Acceptable
                else:
                    return 5.0  # Completed but slow

        return 0.0


# Composite reward for the complete Pallet Town sequence
class PalletTownProgressionCompositeReward(RewardComponent):
    """
    Composite reward that combines all Pallet Town progression milestones.
    
    Total possible points: ~600+
    - Leave bedroom: 20
    - Exit house: 30
    - Find lab: 40
    - Talk to Oak: 50
    - Get starter: 100
    - Enter battle: 75
    - Deal damage: 50 (10 instances × 5)
    - Half HP: 25
    - Low HP: 35
    - Win battle: 150
    - Exit lab: 60
    - Efficiency: 20
    - Keep healthy: 30
    - Navigation: 50
    
    This provides dense, meaningful feedback throughout the entire sequence.
    """

    def __init__(self):
        self.components = [
            LeaveBedroomReward(),
            ExitHouseFirstTimeReward(),
            FindOakLabReward(),
            TalkToOakReward(),
            ReceiveStarterPokemonReward(),
            EnterFirstBattleReward(),
            DealDamageToRivalReward(),
            ReduceEnemyHPByHalfReward(),
            ReduceEnemyHPToLowReward(),
            WinFirstBattleReward(),
            ExitLabAfterBattleReward(),
            FirstBattleEfficiencyReward(),
            KeepPokemonHealthyReward(),
            NavigationSpeedReward(),
        ]

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        total_reward = 0.0
        for component in self.components:
            reward = await component.score(state, action)
            total_reward += reward
        return total_reward

