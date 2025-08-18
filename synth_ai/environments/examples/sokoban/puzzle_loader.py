"""
Puzzle loader for pre-generated verified Sokoban puzzles.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SokobanPuzzle:
    """Represents a verified solvable Sokoban puzzle."""

    id: str
    difficulty: str
    num_boxes: int
    dim_room: Tuple[int, int]
    room_fixed: List[List[int]]
    room_state: List[List[int]]
    box_mapping: Dict[str, Any]
    solution_path: List[int]
    solution_length: int
    generation_seed: int
    max_steps: int

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert room data to numpy arrays for use with the engine."""
        return np.array(self.room_fixed), np.array(self.room_state)

    def to_engine_snapshot(self) -> Dict[str, Any]:
        """Convert puzzle to engine snapshot format."""
        return {
            "dim_room": list(self.dim_room),
            "room_fixed": self.room_fixed,
            "room_state": self.room_state,
            "box_mapping": self.box_mapping,
            "boxes_on_target": self._count_boxes_on_target(),
            "max_steps": self.max_steps,
            "num_boxes": self.num_boxes,
        }

    def _count_boxes_on_target(self) -> int:
        """Count boxes currently on targets (value 3)."""
        room_state = np.array(self.room_state)
        return int(np.sum(room_state == 3))


class SokobanPuzzleLoader:
    """Manages loading and accessing pre-generated Sokoban puzzles."""

    def __init__(self, puzzle_file_path: Optional[Path] = None):
        """
        Initialize the puzzle loader.

        Args:
            puzzle_file_path: Path to the JSON file containing puzzles.
                             If None, uses default path.
        """
        if puzzle_file_path is None:
            puzzle_file_path = Path(__file__).parent / "verified_puzzles.json"

        self.puzzle_file_path = puzzle_file_path
        self.puzzles: Dict[str, List[SokobanPuzzle]] = {}
        self.metadata: Dict[str, Any] = {}
        self._loaded = False

    def load_puzzles(self) -> None:
        """Load puzzles from the JSON file."""
        if self._loaded:
            return

        if not self.puzzle_file_path.exists():
            raise FileNotFoundError(f"Puzzle file not found: {self.puzzle_file_path}")

        try:
            with open(self.puzzle_file_path, "r") as f:
                data = json.load(f)

            self.metadata = data.get("metadata", {})
            puzzle_data = data.get("puzzles", {})

            # Convert to SokobanPuzzle objects
            for difficulty, puzzle_list in puzzle_data.items():
                self.puzzles[difficulty] = []
                for puzzle_dict in puzzle_list:
                    puzzle = SokobanPuzzle(
                        id=puzzle_dict["id"],
                        difficulty=puzzle_dict["difficulty"],
                        num_boxes=puzzle_dict["num_boxes"],
                        dim_room=tuple(puzzle_dict["dim_room"]),
                        room_fixed=puzzle_dict["room_fixed"],
                        room_state=puzzle_dict["room_state"],
                        box_mapping=puzzle_dict["box_mapping"],
                        solution_path=puzzle_dict["solution_path"],
                        solution_length=puzzle_dict["solution_length"],
                        generation_seed=puzzle_dict["generation_seed"],
                        max_steps=puzzle_dict["max_steps"],
                    )
                    self.puzzles[difficulty].append(puzzle)

            self._loaded = True
            logger.info(
                f"Loaded {self.get_total_puzzle_count()} puzzles from {self.puzzle_file_path}"
            )

        except Exception as e:
            logger.error(f"Error loading puzzles: {e}")
            raise

    def get_puzzle_by_id(self, puzzle_id: str) -> Optional[SokobanPuzzle]:
        """Get a specific puzzle by its ID."""
        self.load_puzzles()

        for difficulty_puzzles in self.puzzles.values():
            for puzzle in difficulty_puzzles:
                if puzzle.id == puzzle_id:
                    return puzzle
        return None

    def get_puzzles_by_difficulty(self, difficulty: str) -> List[SokobanPuzzle]:
        """Get all puzzles for a specific difficulty level."""
        self.load_puzzles()
        return self.puzzles.get(difficulty, [])

    def get_random_puzzle(self, difficulty: str) -> Optional[SokobanPuzzle]:
        """Get a random puzzle from the specified difficulty level."""
        puzzles = self.get_puzzles_by_difficulty(difficulty)
        if not puzzles:
            return None
        return random.choice(puzzles)

    def get_puzzle_by_index(self, difficulty: str, index: int) -> Optional[SokobanPuzzle]:
        """Get a puzzle by its index within a difficulty level."""
        puzzles = self.get_puzzles_by_difficulty(difficulty)
        if 0 <= index < len(puzzles):
            return puzzles[index]
        return None

    def get_puzzle_by_seed(self, difficulty: str, seed: int) -> Optional[SokobanPuzzle]:
        """
        Get a puzzle deterministically using a seed via modular arithmetic.
        Same seed will always return the same puzzle for a given difficulty.

        Args:
            difficulty: The difficulty level
            seed: Integer seed for deterministic selection

        Returns:
            SokobanPuzzle or None if no puzzles available
        """
        puzzles = self.get_puzzles_by_difficulty(difficulty)
        if not puzzles:
            return None

        # Use modular arithmetic to map seed to puzzle index
        index = seed % len(puzzles)
        return puzzles[index]

    def get_available_difficulties(self) -> List[str]:
        """Get list of available difficulty levels."""
        self.load_puzzles()
        return list(self.puzzles.keys())

    def get_puzzle_count(self, difficulty: str) -> int:
        """Get the number of puzzles for a specific difficulty."""
        return len(self.get_puzzles_by_difficulty(difficulty))

    def get_total_puzzle_count(self) -> int:
        """Get the total number of puzzles across all difficulties."""
        self.load_puzzles()
        return sum(len(puzzles) for puzzles in self.puzzles.values())

    def get_puzzles_by_criteria(
        self,
        difficulty: Optional[str] = None,
        num_boxes: Optional[int] = None,
        min_solution_length: Optional[int] = None,
        max_solution_length: Optional[int] = None,
        max_results: Optional[int] = None,
    ) -> List[SokobanPuzzle]:
        """
        Get puzzles matching specific criteria.

        Args:
            difficulty: Filter by difficulty level
            num_boxes: Filter by number of boxes
            min_solution_length: Minimum solution length
            max_solution_length: Maximum solution length
            max_results: Maximum number of results to return

        Returns:
            List of matching puzzles
        """
        self.load_puzzles()

        all_puzzles = []
        if difficulty:
            all_puzzles = self.get_puzzles_by_difficulty(difficulty)
        else:
            for difficulty_puzzles in self.puzzles.values():
                all_puzzles.extend(difficulty_puzzles)

        # Apply filters
        filtered_puzzles = []
        for puzzle in all_puzzles:
            if num_boxes is not None and puzzle.num_boxes != num_boxes:
                continue
            if min_solution_length is not None and puzzle.solution_length < min_solution_length:
                continue
            if max_solution_length is not None and puzzle.solution_length > max_solution_length:
                continue
            filtered_puzzles.append(puzzle)

        # Limit results
        if max_results is not None:
            filtered_puzzles = filtered_puzzles[:max_results]

        return filtered_puzzles

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the puzzle set."""
        self.load_puzzles()
        return self.metadata

    def get_metadata_for_filtering(self) -> Dict[str, Any]:
        """Get metadata to help with filtering across environments."""
        self.load_puzzles()

        return {
            "environment_type": "sokoban",
            "total_puzzles": self.get_total_puzzle_count(),
            "difficulties": self.get_available_difficulties(),
            "difficulty_counts": {
                difficulty: len(puzzles) for difficulty, puzzles in self.puzzles.items()
            },
            "features": {
                "has_boxes": True,
                "has_targets": True,
                "has_player": True,
                "grid_based": True,
                "puzzle_type": "box_pushing",
            },
            "difficulty_ranges": {
                "ultra_easy": {"boxes": 1, "grid_size": (5, 5), "solution_length": (3, 8)},
                "easy": {"boxes": 1, "grid_size": (6, 6), "solution_length": (8, 15)},
                "medium": {"boxes": 2, "grid_size": (7, 7), "solution_length": (15, 30)},
                "hard": {"boxes": 3, "grid_size": (8, 8), "solution_length": (30, 60)},
            },
        }

    def print_summary(self) -> None:
        """Print a summary of loaded puzzles."""
        self.load_puzzles()

        print(f"Sokoban Puzzle Summary:")
        print(f"Total puzzles: {self.get_total_puzzle_count()}")
        print(f"Difficulties: {', '.join(self.get_available_difficulties())}")
        print()

        for difficulty in self.get_available_difficulties():
            puzzles = self.get_puzzles_by_difficulty(difficulty)
            if puzzles:
                avg_solution_length = sum(p.solution_length for p in puzzles) / len(puzzles)
                min_solution_length = min(p.solution_length for p in puzzles)
                max_solution_length = max(p.solution_length for p in puzzles)

                print(f"{difficulty}:")
                print(f"  Count: {len(puzzles)}")
                print(f"  Avg solution length: {avg_solution_length:.1f}")
                print(f"  Solution length range: {min_solution_length}-{max_solution_length}")
                print(f"  Boxes: {puzzles[0].num_boxes}")
                print(f"  Room size: {puzzles[0].dim_room}")
                print()


# Global instance for easy access
_global_loader = None


def get_puzzle_loader() -> SokobanPuzzleLoader:
    """Get the global puzzle loader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = SokobanPuzzleLoader()
    return _global_loader


def get_puzzle_by_id(puzzle_id: str) -> Optional[SokobanPuzzle]:
    """Convenience function to get a puzzle by ID."""
    return get_puzzle_loader().get_puzzle_by_id(puzzle_id)


def get_random_puzzle(difficulty: str) -> Optional[SokobanPuzzle]:
    """Convenience function to get a random puzzle."""
    return get_puzzle_loader().get_random_puzzle(difficulty)


def get_puzzle_by_index(difficulty: str, index: int) -> Optional[SokobanPuzzle]:
    """Convenience function to get a puzzle by index."""
    return get_puzzle_loader().get_puzzle_by_index(difficulty, index)


def get_puzzle_by_seed(difficulty: str, seed: int) -> Optional[SokobanPuzzle]:
    """Convenience function to get a puzzle by seed."""
    return get_puzzle_loader().get_puzzle_by_seed(difficulty, seed)
