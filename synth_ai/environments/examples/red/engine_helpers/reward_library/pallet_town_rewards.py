"""
Pallet Town Early Game Reward Components

Rewards specifically designed for the first 50 steps of Pokemon Red,
focusing on house exploration, town discovery, and story triggers.
"""

from synth_ai.environments.environment.rewards.core import RewardComponent
from typing import Dict, Any, Set


class LeaveStartingRoomReward(RewardComponent):
    """Reward for going downstairs from bedroom - +15 points"""

    def __init__(self):
        self.triggered = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.triggered:
            return 0.0

        prev_map = action.get("prev_map_id", -1)
        current_map = state["map_id"]
        prev_y = action.get("prev_player_y", -1)
        current_y = state["player_y"]

        # Detect going downstairs from bedroom (map change + y coordinate change)
        if prev_map != current_map and prev_y > current_y:
            self.triggered = True
            return 15.0
        return 0.0


class TalkToMomReward(RewardComponent):
    """Reward for first conversation with mom - +10 points"""

    def __init__(self):
        self.mom_talked_to = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.mom_talked_to:
            return 0.0

        # Check if we're in mom's house and had a text interaction
        if state["map_id"] in [1, 2] and state["text_box_active"]:  # Assuming house maps
            prev_text_active = action.get("prev_text_box_active", False)
            if not prev_text_active and state["text_box_active"]:
                self.mom_talked_to = True
                return 10.0
        return 0.0


class InteractWithTVReward(RewardComponent):
    """Reward for checking the TV downstairs - +5 points"""

    def __init__(self):
        self.tv_checked = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.tv_checked:
            return 0.0

        # Detect TV interaction in house
        if state["map_id"] in [1, 2] and state["text_box_active"]:
            prev_text_active = action.get("prev_text_box_active", False)
            if not prev_text_active and state["text_box_active"]:
                # Simple heuristic: TV is usually in certain positions
                player_x, player_y = state["player_x"], state["player_y"]
                if (player_x, player_y) in [
                    (3, 4),
                    (4, 4),
                    (5, 4),
                ]:  # Common TV positions
                    self.tv_checked = True
                    return 5.0
        return 0.0


class CheckComputerReward(RewardComponent):
    """Reward for interacting with PC in room - +5 points"""

    def __init__(self):
        self.pc_checked = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.pc_checked:
            return 0.0

        # Detect PC interaction in bedroom
        if state["map_id"] == 1 and state["text_box_active"]:  # Bedroom
            prev_text_active = action.get("prev_text_box_active", False)
            if not prev_text_active and state["text_box_active"]:
                # PC is usually in upper right of bedroom
                player_x, player_y = state["player_x"], state["player_y"]
                if player_x >= 6 and player_y <= 3:
                    self.pc_checked = True
                    return 5.0
        return 0.0


class HouseFullyExploredReward(RewardComponent):
    """Reward for checking all interactive objects in starting house - +20 points"""

    def __init__(self):
        self.interactions: Set[str] = set()
        self.required_interactions = {"tv", "pc", "mom", "bookshelf", "poster"}

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if len(self.interactions) >= len(self.required_interactions):
            return 0.0

        # Track interactions in house
        if state["map_id"] in [1, 2] and state["text_box_active"]:
            prev_text_active = action.get("prev_text_box_active", False)
            if not prev_text_active and state["text_box_active"]:
                player_x, player_y = state["player_x"], state["player_y"]

                # Map positions to interaction types
                if (player_x, player_y) in [(3, 4), (4, 4), (5, 4)]:
                    self.interactions.add("tv")
                elif player_x >= 6 and player_y <= 3:
                    self.interactions.add("pc")
                elif (player_x, player_y) in [(1, 4), (2, 4)]:
                    self.interactions.add("mom")
                # Add more position mappings as needed

                if len(self.interactions) >= len(self.required_interactions):
                    return 20.0
        return 0.0


class ExitHouseReward(RewardComponent):
    """Reward for first time leaving the starting house - +20 points"""

    def __init__(self):
        self.house_exited = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.house_exited:
            return 0.0

        prev_map = action.get("prev_map_id", -1)
        current_map = state["map_id"]

        # Exit from house to town
        if prev_map in [1, 2] and current_map == 0:  # House to Pallet Town
            self.house_exited = True
            return 20.0
        return 0.0


class ExploreTownReward(RewardComponent):
    """Reward for each new building/house entered - +5 points"""

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
                return 5.0
        return 0.0


class TalkToNPCsReward(RewardComponent):
    """Reward for each unique NPC conversation in Pallet Town - +8 points"""

    def __init__(self):
        self.npcs_talked_to: Set[tuple] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Detect NPC conversations in Pallet Town
        if state["map_id"] == 0 and state["text_box_active"]:  # Pallet Town
            prev_text_active = action.get("prev_text_box_active", False)
            if not prev_text_active and state["text_box_active"]:
                # Use position as NPC identifier
                npc_key = (state["player_x"], state["player_y"], state["map_id"])
                if npc_key not in self.npcs_talked_to:
                    self.npcs_talked_to.add(npc_key)
                    return 8.0
        return 0.0


class OakLabDiscoveryReward(RewardComponent):
    """Reward for finding and entering Oak's lab - +25 points"""

    def __init__(self):
        self.lab_discovered = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.lab_discovered:
            return 0.0

        prev_map = action.get("prev_map_id", -1)
        current_map = state["map_id"]

        # Entering Oak's lab (map 3)
        if prev_map == 0 and current_map == 3:
            self.lab_discovered = True
            return 25.0
        return 0.0


class AttemptRoute1Reward(RewardComponent):
    """Reward for trying to leave town (triggers Oak encounter) - +30 points"""

    def __init__(self):
        self.route_attempted = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.route_attempted:
            return 0.0

        # Detect reaching the edge of Pallet Town (attempting to go north)
        if state["map_id"] == 0:  # In Pallet Town
            if state["player_y"] <= 1:  # At northern edge
                self.route_attempted = True
                return 30.0
        return 0.0


class OakEncounterReward(RewardComponent):
    """Reward for triggering Professor Oak to stop you - +50 points"""

    def __init__(self):
        self.oak_encountered = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.oak_encountered:
            return 0.0

        # Detect Oak stopping you (usually involves specific dialogue)
        if state["text_box_active"] and not action.get("prev_text_box_active", False):
            # Check if we're in a situation where Oak would appear
            if state["map_id"] == 0 and state["player_y"] <= 2:
                self.oak_encountered = True
                return 50.0
        return 0.0


class FollowOakToLabReward(RewardComponent):
    """Reward for returning to lab with Oak - +40 points"""

    def __init__(self):
        self.followed_oak = False
        self.oak_encounter_happened = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.followed_oak:
            return 0.0

        # Track Oak encounter first
        if not self.oak_encounter_happened and state["map_id"] == 0 and state["player_y"] <= 2:
            if state["text_box_active"]:
                self.oak_encounter_happened = True

        # Then reward entering lab after encounter
        if self.oak_encounter_happened:
            prev_map = action.get("prev_map_id", -1)
            current_map = state["map_id"]
            if prev_map == 0 and current_map == 3:  # Town to lab
                self.followed_oak = True
                return 40.0
        return 0.0


class ChooseStarterPokemonReward(RewardComponent):
    """Reward for selecting first Pokemon - +100 points"""

    def __init__(self):
        self.starter_chosen = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.starter_chosen:
            return 0.0

        # Detect getting first Pokemon
        prev_party_count = len(action.get("prev_party", []))
        current_party_count = len(state.get("party", []))

        if prev_party_count == 0 and current_party_count == 1:
            if state["map_id"] == 3:  # In Oak's lab
                self.starter_chosen = True
                return 100.0
        return 0.0


class RivalEncounterReward(RewardComponent):
    """Reward for meeting and naming rival - +30 points"""

    def __init__(self):
        self.rival_met = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.rival_met:
            return 0.0

        # Detect rival encounter (specific dialogue patterns)
        if state["map_id"] == 3 and state["text_box_active"]:  # In Oak's lab
            # This is a simplified check - in reality you'd analyze dialogue content
            prev_text_active = action.get("prev_text_box_active", False)
            if not prev_text_active and state["text_box_active"]:
                # Check if we have at least one Pokemon (starter chosen)
                if len(state.get("party", [])) >= 1:
                    self.rival_met = True
                    return 30.0
        return 0.0


class FirstPokemonBattleReward(RewardComponent):
    """Reward for the first battle with rival - +75 points"""

    def __init__(self):
        self.first_battle_done = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.first_battle_done:
            return 0.0

        # Detect entering battle for first time
        prev_in_battle = action.get("prev_in_battle", False)
        current_in_battle = state["in_battle"]

        if not prev_in_battle and current_in_battle:
            if state["map_id"] == 3:  # In Oak's lab
                self.first_battle_done = True
                return 75.0
        return 0.0


class MenuDiscoveryReward(RewardComponent):
    """Reward for opening START menu for first time - +10 points"""

    def __init__(self):
        self.menu_discovered = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.menu_discovered:
            return 0.0

        # This would need menu state tracking - simplified implementation
        # In real implementation, you'd track when START is pressed
        buttons_pressed = action.get("buttons_pressed", [])
        if "START" in buttons_pressed and not self.menu_discovered:
            self.menu_discovered = True
            return 10.0
        return 0.0


class PokemonMenuReward(RewardComponent):
    """Reward for checking Pokemon party status - +15 points"""

    def __init__(self):
        self.pokemon_menu_checked = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.pokemon_menu_checked:
            return 0.0

        # Simplified - would need menu navigation tracking
        # This is a placeholder for actual menu state detection
        if len(state.get("party", [])) > 0:  # Has Pokemon
            # Assume menu was checked if we have Pokemon and certain conditions
            self.pokemon_menu_checked = True
            return 15.0
        return 0.0


class BagDiscoveryReward(RewardComponent):
    """Reward for opening bag/items menu - +10 points"""

    def __init__(self):
        self.bag_discovered = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.bag_discovered:
            return 0.0

        # Simplified implementation
        if len(state.get("inventory", [])) > 0:
            self.bag_discovered = True
            return 10.0
        return 0.0


class SaveGameReward(RewardComponent):
    """Reward for saving the game for first time - +20 points"""

    def __init__(self):
        self.game_saved = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.game_saved:
            return 0.0

        # This would need save state detection
        # Placeholder implementation
        if state.get("game_saved", False):
            self.game_saved = True
            return 20.0
        return 0.0


class TryAllDirectionsReward(RewardComponent):
    """Reward for attempting movement in all 4 directions - +5 points"""

    def __init__(self):
        self.directions_tried: Set[str] = set()
        self.reward_given = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.reward_given:
            return 0.0

        # Track movement directions
        buttons_pressed = action.get("buttons_pressed", [])
        for button in buttons_pressed:
            if button in ["UP", "DOWN", "LEFT", "RIGHT"]:
                self.directions_tried.add(button)

        if len(self.directions_tried) >= 4:
            self.reward_given = True
            return 5.0
        return 0.0


class DoorInteractionReward(RewardComponent):
    """Reward for trying to enter each door/building - +3 points per door"""

    def __init__(self):
        self.doors_tried: Set[tuple] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Detect door interaction attempts
        prev_map = action.get("prev_map_id", -1)
        current_map = state["map_id"]

        if prev_map != current_map and prev_map == 0:  # From town to building
            door_key = (state["player_x"], state["player_y"], current_map)
            if door_key not in self.doors_tried:
                self.doors_tried.add(door_key)
                return 3.0
        return 0.0


class ObjectInteractionReward(RewardComponent):
    """Reward for pressing A on various objects - +3 points per object"""

    def __init__(self):
        self.objects_interacted: Set[tuple] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Detect A button interactions
        buttons_pressed = action.get("buttons_pressed", [])
        if "A" in buttons_pressed and state["text_box_active"]:
            prev_text_active = action.get("prev_text_box_active", False)
            if not prev_text_active:
                object_key = (state["player_x"], state["player_y"], state["map_id"])
                if object_key not in self.objects_interacted:
                    self.objects_interacted.add(object_key)
                    return 3.0
        return 0.0


class SignReadingReward(RewardComponent):
    """Reward for reading town sign and other informational signs - +5 points"""

    def __init__(self):
        self.signs_read: Set[tuple] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Detect sign reading (specific positions in town)
        if state["map_id"] == 0 and state["text_box_active"]:  # Pallet Town
            prev_text_active = action.get("prev_text_box_active", False)
            if not prev_text_active:
                # Town sign is usually at specific coordinates
                sign_positions = [(5, 8), (6, 8), (7, 8)]  # Common sign positions
                player_pos = (state["player_x"], state["player_y"])
                if player_pos in sign_positions:
                    sign_key = (state["player_x"], state["player_y"])
                    if sign_key not in self.signs_read:
                        self.signs_read.add(sign_key)
                        return 5.0
        return 0.0


class CompleteTownExplorationReward(RewardComponent):
    """Reward for visiting every accessible location - +50 points"""

    def __init__(self):
        self.locations_visited: Set[tuple] = set()
        self.required_locations = 20  # Estimated accessible tiles in Pallet Town
        self.reward_given = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.reward_given:
            return 0.0

        if state["map_id"] == 0:  # In Pallet Town
            location_key = (state["player_x"], state["player_y"])
            self.locations_visited.add(location_key)

            if len(self.locations_visited) >= self.required_locations:
                self.reward_given = True
                return 50.0
        return 0.0


class AllNPCsTalkedToReward(RewardComponent):
    """Reward for speaking with every NPC in town - +30 points"""

    def __init__(self):
        self.npcs_talked_to: Set[tuple] = set()
        self.required_npcs = 5  # Estimated NPCs in Pallet Town
        self.reward_given = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.reward_given:
            return 0.0

        # Track NPC conversations
        if state["map_id"] == 0 and state["text_box_active"]:
            prev_text_active = action.get("prev_text_box_active", False)
            if not prev_text_active:
                npc_key = (state["player_x"], state["player_y"])
                self.npcs_talked_to.add(npc_key)

                if len(self.npcs_talked_to) >= self.required_npcs:
                    self.reward_given = True
                    return 30.0
        return 0.0


class ReadyForAdventureReward(RewardComponent):
    """Reward for having starter Pokemon and being ready to leave town - +60 points"""

    def __init__(self):
        self.ready_reward_given = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.ready_reward_given:
            return 0.0

        # Check if player has starter and is at town exit
        has_pokemon = len(state.get("party", [])) > 0
        at_town_exit = state["map_id"] == 0 and state["player_y"] <= 2

        if has_pokemon and at_town_exit:
            self.ready_reward_given = True
            return 60.0
        return 0.0
