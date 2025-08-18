"""
Exploration & Discovery Reward Components

Rewards for map exploration, item discovery, and world interaction.
"""

from typing import Any, Dict, Set

from synth_ai.environments.environment.rewards.core import RewardComponent


class NewAreaDiscoveryReward(RewardComponent):
    """Reward for entering a new map/area for the first time - +10 points"""

    def __init__(self):
        self.discovered_areas: Set[int] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        current_map = state["map_id"]
        if current_map not in self.discovered_areas:
            self.discovered_areas.add(current_map)
            return 10.0
        return 0.0


class AreaCompletionReward(RewardComponent):
    """Reward for visiting all accessible tiles in an area - +5 points"""

    def __init__(self):
        self.area_tiles: Dict[int, Set[tuple]] = {}
        self.completed_areas: Set[int] = set()
        # These would be loaded from game data in a real implementation
        self.area_tile_counts = {
            0: 25,  # Pallet Town
            1: 15,  # House interior
            3: 20,  # Oak's Lab
            # Add more as needed
        }

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        current_map = state["map_id"]
        if current_map in self.completed_areas:
            return 0.0

        # Track tiles visited in this area
        if current_map not in self.area_tiles:
            self.area_tiles[current_map] = set()

        tile = (state["player_x"], state["player_y"])
        self.area_tiles[current_map].add(tile)

        # Check if area is complete
        required_tiles = self.area_tile_counts.get(current_map, 999)
        if len(self.area_tiles[current_map]) >= required_tiles:
            self.completed_areas.add(current_map)
            return 5.0
        return 0.0


class RouteCompletionReward(RewardComponent):
    """Reward for fully exploring a route - +15 points"""

    def __init__(self):
        self.completed_routes: Set[int] = set()
        self.route_progress: Dict[int, Dict[str, bool]] = {}

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        current_map = state["map_id"]

        # Check if this is a route (routes typically have IDs in a certain range)
        if current_map < 10 or current_map in self.completed_routes:
            return 0.0

        # Track route exploration elements
        if current_map not in self.route_progress:
            self.route_progress[current_map] = {
                "all_grass_visited": False,
                "all_trainers_fought": False,
                "all_items_found": False,
            }

        # This is simplified - would need actual game data
        # For now, just reward first full exploration
        if current_map not in self.completed_routes:
            self.completed_routes.add(current_map)
            return 15.0
        return 0.0


class BuildingEntryReward(RewardComponent):
    """Reward for entering buildings - +3 points"""

    def __init__(self):
        self.buildings_entered: Set[int] = set()
        # Building map IDs (would be loaded from game data)
        self.building_maps = {3, 4, 5, 6, 7, 8, 9, 10}  # Example building IDs

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        current_map = state["map_id"]
        prev_map = action.get("prev_map_id", -1)

        # Entering a building for the first time
        if (
            current_map in self.building_maps
            and current_map not in self.buildings_entered
            and prev_map != current_map
        ):
            self.buildings_entered.add(current_map)
            return 3.0
        return 0.0


class HiddenAreaDiscoveryReward(RewardComponent):
    """Reward for finding secret areas - +20 points"""

    def __init__(self):
        self.hidden_areas_found: Set[int] = set()
        # Hidden area map IDs (would be loaded from game data)
        self.hidden_areas = {50, 51, 52}  # Example hidden area IDs

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        current_map = state["map_id"]

        if current_map in self.hidden_areas and current_map not in self.hidden_areas_found:
            self.hidden_areas_found.add(current_map)
            return 20.0
        return 0.0


class HiddenItemFoundReward(RewardComponent):
    """Reward for finding hidden items - +5 points"""

    def __init__(self):
        self.hidden_items_found: Set[tuple] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Detect item acquisition at specific locations
        prev_inventory_count = len(action.get("prev_inventory", []))
        current_inventory_count = len(state.get("inventory", []))

        if current_inventory_count > prev_inventory_count:
            # Item was acquired
            item_location = (state["player_x"], state["player_y"], state["map_id"])
            if item_location not in self.hidden_items_found:
                # Check if this is a hidden item location (would use game data)
                # For now, use heuristic based on position
                if self._is_hidden_item_location(item_location):
                    self.hidden_items_found.add(item_location)
                    return 5.0
        return 0.0

    def _is_hidden_item_location(self, location: tuple) -> bool:
        """Check if location is known to have hidden items"""
        # This would be loaded from game data
        hidden_locations = {
            (3, 5, 0),  # Example hidden item in Pallet Town
            (7, 2, 1),  # Example hidden item in route
        }
        return location in hidden_locations


class FirstItemOfTypeReward(RewardComponent):
    """Reward for finding first item of each type - +10 points"""

    def __init__(self):
        self.item_types_found: Set[str] = set()
        # Item type mappings (would be loaded from game data)
        self.item_types = {
            1: "pokeball",
            2: "pokeball",
            3: "pokeball",
            10: "potion",
            11: "potion",
            12: "potion",
            20: "tm",
            21: "tm",
            22: "tm",
            # Add more mappings
        }

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_inventory = action.get("prev_inventory", [])
        current_inventory = state.get("inventory", [])

        # Check for new items
        prev_item_ids = {item.get("item_id") for item in prev_inventory}
        current_item_ids = {item.get("item_id") for item in current_inventory}
        new_items = current_item_ids - prev_item_ids

        total_reward = 0.0
        for item_id in new_items:
            item_type = self.item_types.get(item_id, "unknown")
            if item_type not in self.item_types_found and item_type != "unknown":
                self.item_types_found.add(item_type)
                total_reward += 10.0

        return total_reward


class RareItemDiscoveryReward(RewardComponent):
    """Reward for finding rare items - +25 points"""

    def __init__(self):
        self.rare_items_found: Set[int] = set()
        # Rare item IDs (would be loaded from game data)
        self.rare_items = {1, 50, 100}  # Master Ball, rare TMs, etc.

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_inventory = action.get("prev_inventory", [])
        current_inventory = state.get("inventory", [])

        # Check for new rare items
        prev_item_ids = {item.get("item_id") for item in prev_inventory}
        current_item_ids = {item.get("item_id") for item in current_inventory}
        new_items = current_item_ids - prev_item_ids

        total_reward = 0.0
        for item_id in new_items:
            if item_id in self.rare_items and item_id not in self.rare_items_found:
                self.rare_items_found.add(item_id)
                total_reward += 25.0

        return total_reward


class KeyItemAcquisitionReward(RewardComponent):
    """Reward for obtaining story-critical items - +30 points"""

    def __init__(self):
        self.key_items_obtained: Set[int] = set()
        # Key item IDs (would be loaded from game data)
        self.key_items = {200, 201, 202}  # Pokedex, Town Map, etc.

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_inventory = action.get("prev_inventory", [])
        current_inventory = state.get("inventory", [])

        # Check for new key items
        prev_item_ids = {item.get("item_id") for item in prev_inventory}
        current_item_ids = {item.get("item_id") for item in current_inventory}
        new_items = current_item_ids - prev_item_ids

        total_reward = 0.0
        for item_id in new_items:
            if item_id in self.key_items and item_id not in self.key_items_obtained:
                self.key_items_obtained.add(item_id)
                total_reward += 30.0

        return total_reward


class FirstWarpUsageReward(RewardComponent):
    """Reward for using doors, cave entrances, etc. for first time - +5 points"""

    def __init__(self):
        self.warp_types_used: Set[str] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        prev_map = action.get("prev_map_id", -1)
        current_map = state["map_id"]

        if prev_map != current_map:
            # Determine warp type based on map transition
            warp_type = self._get_warp_type(prev_map, current_map)
            if warp_type and warp_type not in self.warp_types_used:
                self.warp_types_used.add(warp_type)
                return 5.0
        return 0.0

    def _get_warp_type(self, prev_map: int, current_map: int) -> str:
        """Determine type of warp based on map transition"""
        # This would use game data to classify warps
        if prev_map == 0 and current_map > 0:
            return "door"
        elif prev_map > 10 and current_map > 10:
            return "cave"
        elif abs(prev_map - current_map) == 1:
            return "route_transition"
        return ""


class PCUsageReward(RewardComponent):
    """Reward for first time using Pokemon PC storage - +10 points"""

    def __init__(self):
        self.pc_used = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.pc_used:
            return 0.0

        # Detect PC usage (would need menu state tracking)
        # Simplified: assume PC is used if in Pokemon Center and menu state changes
        if state["map_id"] in [4, 8, 12] and state.get("menu_state", 0) > 0:  # Pokemon Centers
            # This is a placeholder - would need actual PC detection
            if not action.get("prev_menu_state", 0):
                self.pc_used = True
                return 10.0
        return 0.0


class VendingMachineReward(RewardComponent):
    """Reward for discovering and using vending machines - +5 points"""

    def __init__(self):
        self.vending_machines_used: Set[tuple] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Detect vending machine usage
        prev_money = action.get("prev_money", 0)
        current_money = state.get("money", 0)
        prev_inventory_count = len(action.get("prev_inventory", []))
        current_inventory_count = len(state.get("inventory", []))

        # Money decreased and items increased = purchase
        if current_money < prev_money and current_inventory_count > prev_inventory_count:
            # Check if at vending machine location
            vending_location = (state["player_x"], state["player_y"], state["map_id"])
            if self._is_vending_machine_location(vending_location):
                if vending_location not in self.vending_machines_used:
                    self.vending_machines_used.add(vending_location)
                    return 5.0
        return 0.0

    def _is_vending_machine_location(self, location: tuple) -> bool:
        """Check if location has a vending machine"""
        # This would be loaded from game data
        vending_locations = {
            (5, 3, 15),  # Example vending machine location
        }
        return location in vending_locations
