"""
Social & NPC Interaction Reward Components

Rewards for dialogue, information gathering, and NPC interactions.
"""

from typing import Any, Dict, Set

from synth_ai.environments.environment.rewards.core import RewardComponent


class NewNPCConversationReward(RewardComponent):
    """Reward for talking to each unique NPC for the first time - +5 points"""

    def __init__(self):
        self.npcs_talked_to: Set[tuple] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if state["text_box_active"] and not action.get("prev_text_box_active", False):
            npc_key = (state["player_x"], state["player_y"], state["map_id"])
            if npc_key not in self.npcs_talked_to:
                self.npcs_talked_to.add(npc_key)
                return 5.0
        return 0.0


class HelpfulInformationReceivedReward(RewardComponent):
    """Reward for getting useful hints, directions, or game tips - +10 points"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # This would need dialogue content analysis
        # Placeholder implementation
        if state["text_box_active"] and not action.get("prev_text_box_active", False):
            # Simplified: reward for certain locations known to give helpful info
            helpful_locations = {(5, 3, 0), (2, 4, 3)}  # Example helpful NPC locations
            location = (state["player_x"], state["player_y"], state["map_id"])
            if location in helpful_locations:
                return 10.0
        return 0.0


class StoryDialogueProgressionReward(RewardComponent):
    """Reward for advancing story through key NPCs - +15 points"""

    def __init__(self):
        self.story_npcs_talked_to: Set[tuple] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Story NPCs in key locations
        story_locations = {(3, 4, 3), (5, 2, 0)}  # Oak's lab, important NPCs
        location = (state["player_x"], state["player_y"], state["map_id"])

        if (
            state["text_box_active"]
            and not action.get("prev_text_box_active", False)
            and location in story_locations
            and location not in self.story_npcs_talked_to
        ):
            self.story_npcs_talked_to.add(location)
            return 15.0
        return 0.0


class ProfessorOakInteractionsReward(RewardComponent):
    """Reward for meaningful interactions with Professor Oak - +20 points"""

    def __init__(self):
        self.oak_interactions = 0

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Oak's lab interactions
        if (
            state["map_id"] == 3
            and state["text_box_active"]
            and not action.get("prev_text_box_active", False)
        ):
            # Check if this is likely Oak (center of lab)
            if 3 <= state["player_x"] <= 5 and 4 <= state["player_y"] <= 6:
                return 20.0
        return 0.0


class NPCGiftReceivedReward(RewardComponent):
    """Reward for receiving Pokemon or items from NPCs - +15 points"""

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Check for item/Pokemon acquisition during NPC interaction
        prev_inventory_count = len(action.get("prev_inventory", []))
        current_inventory_count = len(state.get("inventory", []))
        prev_party_count = len(action.get("prev_party", []))
        current_party_count = len(state.get("party", []))

        # Gift received if items/Pokemon increased during text interaction
        if state["text_box_active"] and (
            current_inventory_count > prev_inventory_count or current_party_count > prev_party_count
        ):
            return 15.0
        return 0.0


class TradeCompletionReward(RewardComponent):
    """Reward for completing in-game trades - +25 points"""

    def __init__(self):
        self.trades_completed: Set[tuple] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Trade locations (would be loaded from game data)
        trade_locations = {(2, 3, 15), (4, 5, 20)}  # Example trade locations
        location = (state["player_x"], state["player_y"], state["map_id"])

        if location in trade_locations and location not in self.trades_completed:
            # Check for Pokemon species change (trade occurred)
            prev_party = action.get("prev_party", [])
            current_party = state.get("party", [])

            if len(prev_party) == len(current_party):
                prev_species = {p.get("species_id") for p in prev_party}
                current_species = {p.get("species_id") for p in current_party}

                if prev_species != current_species:
                    self.trades_completed.add(location)
                    return 25.0
        return 0.0


class NameRaterUsageReward(RewardComponent):
    """Reward for using nickname services - +5 points"""

    def __init__(self):
        self.name_rater_used = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.name_rater_used:
            return 0.0

        # Name Rater location (would be loaded from game data)
        name_rater_location = (3, 2, 25)  # Example location
        location = (state["player_x"], state["player_y"], state["map_id"])

        if (
            location == name_rater_location
            and state["text_box_active"]
            and not action.get("prev_text_box_active", False)
        ):
            self.name_rater_used = True
            return 5.0
        return 0.0
