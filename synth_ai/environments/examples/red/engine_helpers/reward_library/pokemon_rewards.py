"""
Pokemon Collection & Management Reward Components

Rewards for catching Pokemon, Pokedex progress, and Pokemon development.
"""

from typing import Any, Dict, Set

from synth_ai.environments.environment.rewards.core import RewardComponent


class FirstPokemonCaughtReward(RewardComponent):
    """Reward for catching the starter or first wild Pokemon - +50 points"""

    def __init__(self):
        self.first_caught = False

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        if self.first_caught:
            return 0.0

        prev_party_count = len(action.get("prev_party", []))
        current_party_count = len(state.get("party", []))

        # First Pokemon acquired (starter or caught)
        if prev_party_count == 0 and current_party_count == 1:
            self.first_caught = True
            return 50.0
        return 0.0


class NewSpeciesCaughtReward(RewardComponent):
    """Reward for each new Pokedex entry - +20 points"""

    def __init__(self):
        self.species_caught: Set[int] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Check for new Pokemon in party
        party = state.get("party", [])
        total_reward = 0.0

        for pokemon in party:
            species_id = pokemon.get("species_id", 0)
            if species_id not in self.species_caught and species_id > 0:
                self.species_caught.add(species_id)
                total_reward += 20.0

        return total_reward


class RarePokemonCaughtReward(RewardComponent):
    """Reward for catching uncommon/rare Pokemon - +40 points"""

    def __init__(self):
        self.rare_pokemon_caught: Set[int] = set()
        # Rare Pokemon species IDs (would be loaded from game data)
        self.rare_species = {
            144,
            145,
            146,  # Legendary birds
            150,  # Mewtwo
            149,  # Dragonite
            130,
            131,  # Gyarados, Lapras
            138,
            139,  # Omanyte, Omastar
            140,
            141,  # Kabuto, Kabutops
            142,  # Aerodactyl
        }

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        party = state.get("party", [])
        total_reward = 0.0

        for pokemon in party:
            species_id = pokemon.get("species_id", 0)
            if species_id in self.rare_species and species_id not in self.rare_pokemon_caught:
                self.rare_pokemon_caught.add(species_id)
                total_reward += 40.0

        return total_reward


class EvolutionStonePokemonReward(RewardComponent):
    """Reward for catching Pokemon that require evolution stones - +30 points"""

    def __init__(self):
        self.evolution_stone_pokemon_caught: Set[int] = set()
        # Pokemon that evolve with stones
        self.evolution_stone_pokemon = {
            25,  # Pikachu (Thunder Stone)
            30,  # Nidorina (Moon Stone)
            33,  # Nidorino (Moon Stone)
            35,  # Clefairy (Moon Stone)
            37,  # Vulpix (Fire Stone)
            39,  # Jigglypuff (Moon Stone)
            44,  # Gloom (Leaf Stone)
            58,  # Growlithe (Fire Stone)
            61,  # Poliwhirl (Water Stone)
            90,  # Shellder (Water Stone)
            102,  # Exeggcute (Leaf Stone)
            108,  # Lickitung (rare)
            120,  # Staryu (Water Stone)
        }

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        party = state.get("party", [])
        total_reward = 0.0

        for pokemon in party:
            species_id = pokemon.get("species_id", 0)
            if (
                species_id in self.evolution_stone_pokemon
                and species_id not in self.evolution_stone_pokemon_caught
            ):
                self.evolution_stone_pokemon_caught.add(species_id)
                total_reward += 30.0

        return total_reward


class PokedexMilestonesReward(RewardComponent):
    """Reward for reaching Pokedex milestones - +100 points for 10, 25, 50, 100, 150"""

    def __init__(self):
        self.milestones_reached: Set[int] = set()
        self.milestones = [10, 25, 50, 100, 150]
        self.unique_species: Set[int] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Track unique species seen/caught
        party = state.get("party", [])
        for pokemon in party:
            species_id = pokemon.get("species_id", 0)
            if species_id > 0:
                self.unique_species.add(species_id)

        total_reward = 0.0
        species_count = len(self.unique_species)

        for milestone in self.milestones:
            if species_count >= milestone and milestone not in self.milestones_reached:
                self.milestones_reached.add(milestone)
                total_reward += 100.0

        return total_reward


class AreaPokedexCompletionReward(RewardComponent):
    """Reward for catching all Pokemon available in an area - +50 points"""

    def __init__(self):
        self.completed_areas: Set[int] = set()
        # Area Pokemon lists (would be loaded from game data)
        self.area_pokemon = {
            0: {16, 17, 18},  # Pallet Town area (Pidgey line)
            1: {10, 11, 13, 14},  # Route 1 (Caterpie, Weedle lines)
            # Add more areas
        }
        self.caught_by_area: Dict[int, Set[int]] = {}

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        current_map = state["map_id"]

        if current_map in self.completed_areas:
            return 0.0

        # Track caught Pokemon in this area
        if current_map not in self.caught_by_area:
            self.caught_by_area[current_map] = set()

        party = state.get("party", [])
        for pokemon in party:
            species_id = pokemon.get("species_id", 0)
            if species_id > 0:
                self.caught_by_area[current_map].add(species_id)

        # Check if area is complete
        required_pokemon = self.area_pokemon.get(current_map, set())
        if required_pokemon.issubset(self.caught_by_area[current_map]):
            self.completed_areas.add(current_map)
            return 50.0

        return 0.0


class TypeCollectionReward(RewardComponent):
    """Reward for catching first Pokemon of each type - +25 points"""

    def __init__(self):
        self.types_collected: Set[str] = set()
        # Pokemon type mappings (simplified)
        self.pokemon_types = {
            1: "grass",
            4: "fire",
            7: "water",
            25: "electric",
            # Add more mappings
        }

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        party = state.get("party", [])
        total_reward = 0.0

        for pokemon in party:
            species_id = pokemon.get("species_id", 0)
            pokemon_type = self.pokemon_types.get(species_id)

            if pokemon_type and pokemon_type not in self.types_collected:
                self.types_collected.add(pokemon_type)
                total_reward += 25.0

        return total_reward


class PokemonEvolutionReward(RewardComponent):
    """Reward for evolving Pokemon - +30 points"""

    def __init__(self):
        self.evolution_count = 0
        self.previous_species: Set[int] = set()

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        # Track species changes (evolution)
        prev_party = action.get("prev_party", [])
        current_party = state.get("party", [])

        prev_species = {p.get("species_id", 0) for p in prev_party}
        current_species = {p.get("species_id", 0) for p in current_party}

        # Check for evolution (new species appears, old species disappears)
        evolved_species = current_species - prev_species

        if evolved_species and self._is_evolution(prev_species, current_species):
            return 30.0

        return 0.0

    def _is_evolution(self, prev_species: Set[int], current_species: Set[int]) -> bool:
        """Check if species change represents evolution"""
        # This would check evolution chains
        # Simplified: any new species with same party size is evolution
        return len(prev_species) == len(current_species) and prev_species != current_species


class LevelMilestonesReward(RewardComponent):
    """Reward for reaching levels 10, 20, 30, 40, 50 with any Pokemon - +10 points"""

    def __init__(self):
        self.level_milestones_reached: Set[tuple] = set()  # (pokemon_id, level)
        self.milestones = [10, 20, 30, 40, 50]

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        party = state.get("party", [])
        total_reward = 0.0

        for i, pokemon in enumerate(party):
            level = pokemon.get("level", 0)

            for milestone in self.milestones:
                milestone_key = (i, milestone)
                if level >= milestone and milestone_key not in self.level_milestones_reached:
                    self.level_milestones_reached.add(milestone_key)
                    total_reward += 10.0

        return total_reward


class MoveLearningReward(RewardComponent):
    """Reward for learning new moves (not replacing) - +5 points"""

    def __init__(self):
        self.known_moves: Set[tuple] = set()  # (pokemon_index, move_id)

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        party = state.get("party", [])
        total_reward = 0.0

        for i, pokemon in enumerate(party):
            moves = pokemon.get("moves", [])
            for move_id in moves:
                move_key = (i, move_id)
                if move_key not in self.known_moves and move_id > 0:
                    self.known_moves.add(move_key)
                    total_reward += 5.0

        return total_reward


class TMHMTeachingReward(RewardComponent):
    """Reward for successfully teaching TMs/HMs - +10 points"""

    def __init__(self):
        self.tm_hm_taught: Set[tuple] = set()  # (pokemon_index, tm_hm_id)
        # TM/HM move IDs (would be loaded from game data)
        self.tm_hm_moves = set(range(15, 65))  # Example TM/HM move range

    async def score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        party = state.get("party", [])
        total_reward = 0.0

        for i, pokemon in enumerate(party):
            moves = pokemon.get("moves", [])
            for move_id in moves:
                if move_id in self.tm_hm_moves:
                    move_key = (i, move_id)
                    if move_key not in self.tm_hm_taught:
                        self.tm_hm_taught.add(move_key)
                        total_reward += 10.0

        return total_reward
