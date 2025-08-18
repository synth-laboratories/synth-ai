"""
MiniGrid Puzzle Loader

This module provides a comprehensive puzzle loading system for MiniGrid environments
with deterministic seed-based selection and difficulty filtering.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from synth_ai.environments.examples.minigrid.environment_mapping import ENVIRONMENT_MAPPING

logger = logging.getLogger(__name__)


@dataclass
class MiniGridPuzzle:
    """Represents a single MiniGrid puzzle configuration."""

    id: str
    environment_name: str
    difficulty: str
    seed: int
    grid_size: Tuple[int, int]
    mission_description: str

    # Environment features
    has_key: bool = False
    has_door: bool = False
    has_lava: bool = False
    has_multi_room: bool = False
    num_objects: int = 0

    # Difficulty metrics
    complexity_score: float = 0.0
    estimated_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert puzzle to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MiniGridPuzzle":
        """Create puzzle from dictionary."""
        return cls(**data)


class MiniGridPuzzleLoader:
    """Manages loading and accessing MiniGrid puzzles with difficulty filtering."""

    def __init__(self):
        self.puzzles: Dict[str, List[MiniGridPuzzle]] = {}
        self.all_puzzles: List[MiniGridPuzzle] = []
        self._loaded = False

        # Difficulty seed mappings based on user's detailed analysis
        self.difficulty_seeds = {
            "ultra_easy": [
                0,
                1,
                2,
                3,
                4,
                5,
                7,
                8,
                9,
                10,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
            ],
            "easy": [11, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
            "medium": [35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49],
            "hard": [6, 42, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        }

    def load_puzzles(self) -> None:
        """Load all MiniGrid puzzles from the environment mapping."""
        if self._loaded:
            return

        logger.info("Loading MiniGrid puzzles from environment mapping...")

        # Clear existing data
        self.puzzles.clear()
        self.all_puzzles.clear()

        # Initialize difficulty categories
        for difficulty in self.difficulty_seeds.keys():
            self.puzzles[difficulty] = []

        # Load puzzles for each difficulty
        for difficulty, seeds in self.difficulty_seeds.items():
            for seed in seeds:
                puzzle = self._create_puzzle_from_seed(seed, difficulty)
                if puzzle:
                    self.puzzles[difficulty].append(puzzle)
                    self.all_puzzles.append(puzzle)

        self._loaded = True
        logger.info(
            f"Loaded {len(self.all_puzzles)} MiniGrid puzzles across {len(self.puzzles)} difficulties"
        )

    def _create_puzzle_from_seed(self, seed: int, difficulty: str) -> Optional[MiniGridPuzzle]:
        """Create a puzzle from a seed and difficulty."""
        if seed not in ENVIRONMENT_MAPPING:
            logger.warning(f"Seed {seed} not found in environment mapping")
            return None

        env_name = ENVIRONMENT_MAPPING[seed]
        puzzle_id = f"{difficulty}_{seed:03d}"

        # Extract environment features
        has_key = "DoorKey" in env_name or "Unlock" in env_name or "KeyCorridor" in env_name
        has_door = "Door" in env_name or "Room" in env_name or "Unlock" in env_name
        has_lava = "Lava" in env_name
        has_multi_room = "MultiRoom" in env_name or "FourRooms" in env_name

        # Estimate grid size from environment name
        grid_size = self._estimate_grid_size(env_name)

        # Count objects
        num_objects = 0
        if has_key:
            num_objects += 1
        if has_door:
            num_objects += 1
        if "Pickup" in env_name:
            num_objects += 1
        if "Fetch" in env_name:
            if "N2" in env_name:
                num_objects += 2
            elif "N3" in env_name:
                num_objects += 3

        # Generate mission description
        mission_description = self._generate_mission_description(
            env_name, has_key, has_door, has_lava, has_multi_room
        )

        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(
            grid_size, num_objects, has_key, has_door, has_lava, has_multi_room
        )

        # Estimate steps
        estimated_steps = self._estimate_steps(grid_size, complexity_score)

        return MiniGridPuzzle(
            id=puzzle_id,
            environment_name=env_name,
            difficulty=difficulty,
            seed=seed,
            grid_size=grid_size,
            mission_description=mission_description,
            has_key=has_key,
            has_door=has_door,
            has_lava=has_lava,
            has_multi_room=has_multi_room,
            num_objects=num_objects,
            complexity_score=complexity_score,
            estimated_steps=estimated_steps,
        )

    def _estimate_grid_size(self, env_name: str) -> Tuple[int, int]:
        """Estimate grid size from environment name."""
        if "5x5" in env_name:
            return (5, 5)
        elif "6x6" in env_name:
            return (6, 6)
        elif "8x8" in env_name:
            return (8, 8)
        elif "16x16" in env_name:
            return (16, 16)
        elif "FourRooms" in env_name:
            return (19, 19)
        elif "MultiRoom-N2" in env_name:
            return (15, 15)
        elif "MultiRoom-N4" in env_name:
            return (19, 19)
        elif "MultiRoom-N6" in env_name:
            return (25, 25)
        elif "LavaGapS5" in env_name:
            return (5, 7)
        elif "LavaGapS6" in env_name:
            return (6, 8)
        elif "LavaGapS7" in env_name:
            return (7, 9)
        elif "CrossingS9" in env_name:
            return (9, 9)
        elif "CrossingS11" in env_name:
            return (11, 11)
        else:
            return (7, 7)  # Default

    def _generate_mission_description(
        self, env_name: str, has_key: bool, has_door: bool, has_lava: bool, has_multi_room: bool
    ) -> str:
        """Generate mission description based on environment features."""
        if "Empty" in env_name:
            return "Navigate the grid to reach the goal"
        elif "DoorKey" in env_name:
            return "Find the key, unlock the door, and reach the goal"
        elif "Unlock" in env_name:
            return "Use keys to unlock doors and reach the goal"
        elif "MultiRoom" in env_name:
            return "Navigate through multiple rooms to reach the goal"
        elif "LavaGap" in env_name:
            return "Jump over lava gaps to reach the goal"
        elif "LavaCrossing" in env_name:
            return "Navigate through lava fields to reach the goal"
        elif "Fetch" in env_name:
            return "Pick up the required objects"
        elif "PutNear" in env_name:
            return "Pick up objects and place them near other objects"
        elif "KeyCorridor" in env_name:
            return "Navigate corridors with keys and doors"
        elif "FourRooms" in env_name:
            return "Navigate through four connected rooms to reach the goal"
        else:
            return "Complete the mission to reach the goal"

    def _calculate_complexity_score(
        self,
        grid_size: Tuple[int, int],
        num_objects: int,
        has_key: bool,
        has_door: bool,
        has_lava: bool,
        has_multi_room: bool,
    ) -> float:
        """Calculate complexity score based on environment features."""
        width, height = grid_size
        base_score = (width * height) / 100.0  # Normalized by grid size

        # Add complexity for features
        if has_key:
            base_score += 0.5
        if has_door:
            base_score += 0.3
        if has_lava:
            base_score += 0.7
        if has_multi_room:
            base_score += 1.0

        # Add complexity for objects
        base_score += num_objects * 0.2

        return base_score

    def _estimate_steps(self, grid_size: Tuple[int, int], complexity_score: float) -> int:
        """Estimate number of steps required."""
        width, height = grid_size
        base_steps = width + height  # Manhattan distance estimate

        # Scale by complexity
        estimated = int(base_steps * (1.0 + complexity_score))

        return max(10, estimated)  # Minimum 10 steps

    def get_puzzles_by_difficulty(self, difficulty: str) -> List[MiniGridPuzzle]:
        """Get all puzzles for a specific difficulty level."""
        if not self._loaded:
            self.load_puzzles()
        return self.puzzles.get(difficulty, [])

    def get_puzzle_by_seed(self, difficulty: str, seed: int) -> Optional[MiniGridPuzzle]:
        """
        Get a puzzle deterministically using a seed via modular arithmetic.
        Same seed will always return the same puzzle for a given difficulty.
        """
        puzzles = self.get_puzzles_by_difficulty(difficulty)
        if not puzzles:
            return None

        # Use modular arithmetic to map seed to puzzle index
        index = seed % len(puzzles)
        return puzzles[index]

    def get_puzzle_by_index(self, difficulty: str, index: int) -> Optional[MiniGridPuzzle]:
        """Get a puzzle by its index within a difficulty level."""
        puzzles = self.get_puzzles_by_difficulty(difficulty)
        if 0 <= index < len(puzzles):
            return puzzles[index]
        return None

    def get_random_puzzle(self, difficulty: str) -> Optional[MiniGridPuzzle]:
        """Get a random puzzle from the specified difficulty."""
        import random

        puzzles = self.get_puzzles_by_difficulty(difficulty)
        if not puzzles:
            return None
        return random.choice(puzzles)

    def get_available_difficulties(self) -> List[str]:
        """Get list of available difficulty levels."""
        return list(self.difficulty_seeds.keys())

    def get_metadata_for_filtering(self) -> Dict[str, Any]:
        """Get metadata to help with filtering across environments."""
        if not self._loaded:
            self.load_puzzles()

        return {
            "environment_type": "minigrid",
            "total_puzzles": len(self.all_puzzles),
            "difficulties": self.get_available_difficulties(),
            "difficulty_counts": {
                difficulty: len(puzzles) for difficulty, puzzles in self.puzzles.items()
            },
            "features": {
                "has_navigation": True,
                "has_keys": True,
                "has_doors": True,
                "has_lava": True,
                "has_multi_room": True,
                "grid_based": True,
                "puzzle_type": "navigation",
            },
            "difficulty_ranges": {
                "ultra_easy": {"grid_size": (5, 7), "complexity": (0.0, 1.0), "environments": 22},
                "easy": {"grid_size": (6, 15), "complexity": (1.0, 2.0), "environments": 12},
                "medium": {"grid_size": (7, 21), "complexity": (2.0, 4.0), "environments": 14},
                "hard": {"grid_size": (8, 37), "complexity": (4.0, 8.0), "environments": 12},
            },
        }

    def get_total_puzzle_count(self) -> int:
        """Get total number of puzzles across all difficulties."""
        if not self._loaded:
            self.load_puzzles()
        return len(self.all_puzzles)

    def get_difficulty_counts(self) -> Dict[str, int]:
        """Get count of puzzles per difficulty level."""
        if not self._loaded:
            self.load_puzzles()
        return {difficulty: len(puzzles) for difficulty, puzzles in self.puzzles.items()}

    def filter_puzzles(self, **kwargs) -> List[MiniGridPuzzle]:
        """Filter puzzles by various criteria."""
        if not self._loaded:
            self.load_puzzles()

        filtered = self.all_puzzles

        # Filter by difficulty
        if "difficulty" in kwargs:
            filtered = [p for p in filtered if p.difficulty == kwargs["difficulty"]]

        # Filter by features
        if "has_key" in kwargs:
            filtered = [p for p in filtered if p.has_key == kwargs["has_key"]]
        if "has_door" in kwargs:
            filtered = [p for p in filtered if p.has_door == kwargs["has_door"]]
        if "has_lava" in kwargs:
            filtered = [p for p in filtered if p.has_lava == kwargs["has_lava"]]
        if "has_multi_room" in kwargs:
            filtered = [p for p in filtered if p.has_multi_room == kwargs["has_multi_room"]]

        # Filter by grid size
        if "max_width" in kwargs:
            filtered = [p for p in filtered if p.grid_size[0] <= kwargs["max_width"]]
        if "max_height" in kwargs:
            filtered = [p for p in filtered if p.grid_size[1] <= kwargs["max_height"]]

        # Filter by complexity
        if "max_complexity" in kwargs:
            filtered = [p for p in filtered if p.complexity_score <= kwargs["max_complexity"]]

        return filtered


# Global puzzle loader instance
_puzzle_loader: Optional[MiniGridPuzzleLoader] = None


def get_puzzle_loader() -> MiniGridPuzzleLoader:
    """Get the global puzzle loader instance."""
    global _puzzle_loader
    if _puzzle_loader is None:
        _puzzle_loader = MiniGridPuzzleLoader()
    return _puzzle_loader


# Convenience functions
def get_puzzles_by_difficulty(difficulty: str) -> List[MiniGridPuzzle]:
    """Convenience function to get puzzles by difficulty."""
    return get_puzzle_loader().get_puzzles_by_difficulty(difficulty)


def get_puzzle_by_seed(difficulty: str, seed: int) -> Optional[MiniGridPuzzle]:
    """Convenience function to get a puzzle by seed."""
    return get_puzzle_loader().get_puzzle_by_seed(difficulty, seed)


def get_puzzle_by_index(difficulty: str, index: int) -> Optional[MiniGridPuzzle]:
    """Convenience function to get a puzzle by index."""
    return get_puzzle_loader().get_puzzle_by_index(difficulty, index)


def get_available_difficulties() -> List[str]:
    """Convenience function to get available difficulties."""
    return get_puzzle_loader().get_available_difficulties()
