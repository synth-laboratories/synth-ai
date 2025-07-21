#!/usr/bin/env python3
"""
Generate verified solvable Sokoban puzzles.

This script creates 500 solvable Sokoban puzzles (100 each for 5 difficulty levels)
and saves them as JSON. Each puzzle is verified to be solvable using BFS.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from synth_ai.environments.examples.sokoban.engine_helpers.room_utils import (
    generate_room,
    get_shortest_action_path,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
    box_mapping: Dict[str, List[int]]
    solution_path: List[int]
    solution_length: int
    generation_seed: int
    max_steps: int


# Define difficulty configurations
DIFFICULTY_CONFIGS = {
    "ultra_easy": {
        "num_boxes": 1,
        "dim_room": (5, 5),
        "max_steps": 50,
        "target_solution_length": (3, 8),
        "search_depth": 30,
    },
    "easy": {
        "num_boxes": 1,
        "dim_room": (6, 6),
        "max_steps": 80,
        "target_solution_length": (8, 15),
        "search_depth": 50,
    },
    "medium": {
        "num_boxes": 2,
        "dim_room": (7, 7),
        "max_steps": 120,
        "target_solution_length": (15, 30),
        "search_depth": 80,
    },
    "hard": {
        "num_boxes": 3,
        "dim_room": (8, 8),
        "max_steps": 200,
        "target_solution_length": (30, 60),
        "search_depth": 120,
    },
}


def verify_puzzle_solvable(
    room_fixed: np.ndarray, room_state: np.ndarray, max_depth: int = 200
) -> Optional[List[int]]:
    """
    Verify that a puzzle is solvable using BFS and return the solution path.

    Args:
        room_fixed: The fixed room structure (walls, targets, floors)
        room_state: The current room state (player, boxes)
        max_depth: Maximum search depth

    Returns:
        List of actions if solvable, None if not solvable
    """
    try:
        solution_path = get_shortest_action_path(room_fixed, room_state, MAX_DEPTH=max_depth)
        return solution_path if solution_path else None
    except Exception as e:
        logger.warning(f"Error verifying puzzle: {e}")
        return None


def setup_instances_directory() -> Path:
    """Create the instances directory if it doesn't exist."""
    instances_dir = Path(__file__).parent / "instances"
    instances_dir.mkdir(exist_ok=True)
    return instances_dir


def get_jsonl_path(instances_dir: Path, difficulty: str) -> Path:
    """Get the JSONL file path for a difficulty level."""
    return instances_dir / f"{difficulty}.jsonl"


def save_puzzle_to_jsonl(puzzle: SokobanPuzzle, jsonl_path: Path):
    """Save a single puzzle to a JSONL file."""
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(asdict(puzzle), default=convert_numpy_types) + "\n")


def convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def load_existing_puzzles(jsonl_path: Path) -> Set[str]:
    """Load existing puzzle IDs from a JSONL file."""
    existing_ids = set()
    if jsonl_path.exists():
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    puzzle_data = json.loads(line.strip())
                    existing_ids.add(puzzle_data["id"])
                except json.JSONDecodeError:
                    continue
    return existing_ids


def count_existing_puzzles(jsonl_path: Path) -> int:
    """Count existing puzzles in a JSONL file."""
    if not jsonl_path.exists():
        return 0
    with open(jsonl_path, "r") as f:
        return sum(1 for line in f if line.strip())


def generate_puzzle_for_difficulty(
    difficulty: str, config: Dict, seed: int, puzzle_id: str
) -> Optional[SokobanPuzzle]:
    """
    Generate a single puzzle for a given difficulty level.

    Args:
        difficulty: The difficulty level name
        config: Configuration for this difficulty
        seed: Random seed for generation
        puzzle_id: Unique identifier for this puzzle

    Returns:
        SokobanPuzzle if successfully generated and verified, None otherwise
    """
    max_attempts = 20

    for attempt in range(max_attempts):
        current_seed = seed + attempt * 1000

        try:
            # Generate room
            room_structure, room_state, box_mapping, action_sequence = generate_room(
                dim=config["dim_room"],
                initial_seed=current_seed,
                num_boxes=config["num_boxes"],
                search_depth=config["search_depth"],
                num_steps=config["search_depth"] // 2,
            )

            # Verify solvability
            solution_path = verify_puzzle_solvable(
                room_structure, room_state, max_depth=config["max_steps"]
            )

            if solution_path is None:
                logger.debug(f"Puzzle {puzzle_id} attempt {attempt + 1} not solvable")
                continue

            solution_length = len(solution_path)
            target_min, target_max = config["target_solution_length"]

            # Check if solution length is within desired range
            if not (target_min <= solution_length <= target_max):
                logger.debug(
                    f"Puzzle {puzzle_id} attempt {attempt + 1} solution length {solution_length} not in range {target_min}-{target_max}"
                )
                continue

            # Convert numpy arrays to lists for JSON serialization
            room_fixed_list = room_structure.tolist()
            room_state_list = room_state.tolist()

            # Convert box mapping to serializable format
            box_mapping_serializable = {}
            for key, value in box_mapping.items():
                if isinstance(key, tuple):
                    # Convert numpy integers to regular integers
                    key_str = f"{int(key[0])},{int(key[1])}"
                    if isinstance(value, tuple):
                        box_mapping_serializable[key_str] = [int(value[0]), int(value[1])]
                    else:
                        box_mapping_serializable[key_str] = value
                else:
                    box_mapping_serializable[str(key)] = value

            puzzle = SokobanPuzzle(
                id=puzzle_id,
                difficulty=difficulty,
                num_boxes=int(config["num_boxes"]),
                dim_room=config["dim_room"],
                room_fixed=room_fixed_list,
                room_state=room_state_list,
                box_mapping=box_mapping_serializable,
                solution_path=[int(action) for action in solution_path],  # Convert to regular ints
                solution_length=int(solution_length),
                generation_seed=int(current_seed),
                max_steps=int(config["max_steps"]),
            )

            logger.info(
                f"Generated {difficulty} puzzle {puzzle_id} (seed: {current_seed}, solution length: {solution_length})"
            )
            return puzzle

        except Exception as e:
            logger.warning(f"Error generating puzzle {puzzle_id} attempt {attempt + 1}: {e}")
            continue

    logger.error(f"Failed to generate puzzle {puzzle_id} after {max_attempts} attempts")
    return None


def generate_all_puzzles(num_per_difficulty: int = 100) -> Dict[str, List[SokobanPuzzle]]:
    """
    Generate all puzzles for all difficulty levels with incremental saving.

    Args:
        num_per_difficulty: Number of puzzles to generate per difficulty level

    Returns:
        Dictionary mapping difficulty names to lists of puzzles
    """
    all_puzzles = {}
    total_puzzles = 0

    # Setup instances directory
    instances_dir = setup_instances_directory()
    logger.info(f"Using instances directory: {instances_dir}")

    for difficulty, config in DIFFICULTY_CONFIGS.items():
        jsonl_path = get_jsonl_path(instances_dir, difficulty)
        existing_ids = load_existing_puzzles(jsonl_path)
        existing_count = count_existing_puzzles(jsonl_path)

        logger.info(f"Processing {difficulty} difficulty...")
        logger.info(f"  Found {existing_count} existing puzzles")
        logger.info(f"  Target: {num_per_difficulty} puzzles")

        puzzles = []
        base_seed = hash(difficulty) % 100000

        # Generate puzzles until we have enough
        i = 0
        generated_this_run = 0
        while (
            len(puzzles) + existing_count < num_per_difficulty and i < num_per_difficulty * 5
        ):  # Safety limit
            puzzle_id = f"{difficulty}_{i:03d}"

            # Skip if already exists
            if puzzle_id in existing_ids:
                i += 1
                continue

            puzzle = generate_puzzle_for_difficulty(
                difficulty=difficulty, config=config, seed=base_seed + i, puzzle_id=puzzle_id
            )

            if puzzle:
                puzzles.append(puzzle)
                # Save immediately to JSONL
                save_puzzle_to_jsonl(puzzle, jsonl_path)
                generated_this_run += 1
                total_puzzles += 1
                logger.info(
                    f"Generated and saved {difficulty} puzzle {puzzle_id} ({generated_this_run}/{num_per_difficulty - existing_count} new)"
                )
            else:
                logger.warning(f"Failed to generate puzzle {puzzle_id}")

            i += 1

        all_puzzles[difficulty] = puzzles
        logger.info(
            f"Completed {difficulty}: {generated_this_run} new puzzles generated, {existing_count + len(puzzles)} total"
        )

    logger.info(f"Total new puzzles generated this run: {total_puzzles}")
    return all_puzzles


def load_all_puzzles_from_jsonl(instances_dir: Path) -> Dict[str, List[SokobanPuzzle]]:
    """Load all puzzles from JSONL files."""
    all_puzzles = {}

    for difficulty in DIFFICULTY_CONFIGS.keys():
        jsonl_path = get_jsonl_path(instances_dir, difficulty)
        puzzles = []

        if jsonl_path.exists():
            with open(jsonl_path, "r") as f:
                for line in f:
                    try:
                        puzzle_data = json.loads(line.strip())
                        puzzle = SokobanPuzzle(
                            id=puzzle_data["id"],
                            difficulty=puzzle_data["difficulty"],
                            num_boxes=puzzle_data["num_boxes"],
                            dim_room=tuple(puzzle_data["dim_room"]),
                            room_fixed=puzzle_data["room_fixed"],
                            room_state=puzzle_data["room_state"],
                            box_mapping=puzzle_data["box_mapping"],
                            solution_path=puzzle_data["solution_path"],
                            solution_length=puzzle_data["solution_length"],
                            generation_seed=puzzle_data["generation_seed"],
                            max_steps=puzzle_data["max_steps"],
                        )
                        puzzles.append(puzzle)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Error loading puzzle from {jsonl_path}: {e}")
                        continue

        all_puzzles[difficulty] = puzzles

    return all_puzzles


def save_puzzles_to_json(puzzles: Dict[str, List[SokobanPuzzle]], output_path: Path):
    """
    Save puzzles to JSON file.

    Args:
        puzzles: Dictionary of puzzles by difficulty
        output_path: Path to save the JSON file
    """
    # Convert to serializable format
    serializable_puzzles = {}
    for difficulty, puzzle_list in puzzles.items():
        serializable_puzzles[difficulty] = [asdict(puzzle) for puzzle in puzzle_list]

    # Add metadata
    output_data = {
        "metadata": {
            "version": "1.0",
            "total_puzzles": sum(len(puzzles) for puzzles in serializable_puzzles.values()),
            "difficulties": list(DIFFICULTY_CONFIGS.keys()),
            "generated_at": "2024-01-01T00:00:00Z",  # Will be updated when actually generated
        },
        "puzzles": serializable_puzzles,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=convert_numpy_types)

    logger.info(f"Saved puzzles to {output_path}")


def create_unified_json_from_jsonl():
    """Create a unified JSON file from all JSONL files for the puzzle loader."""
    instances_dir = setup_instances_directory()
    all_puzzles = load_all_puzzles_from_jsonl(instances_dir)

    # Save to JSON
    output_path = Path(__file__).parent / "verified_puzzles.json"
    save_puzzles_to_json(all_puzzles, output_path)

    return all_puzzles


def main():
    """Main function to generate and save all puzzles."""
    logger.info("Starting Sokoban puzzle generation with incremental saving...")

    # Generate puzzles (saves incrementally to JSONL)
    puzzles = generate_all_puzzles(num_per_difficulty=100)

    # Print summary of this run
    logger.info("Puzzle generation complete!")
    logger.info("Summary of this run:")
    for difficulty, puzzle_list in puzzles.items():
        if puzzle_list:
            avg_solution_length = sum(p.solution_length for p in puzzle_list) / len(puzzle_list)
            logger.info(
                f"  {difficulty}: {len(puzzle_list)} new puzzles, avg solution length: {avg_solution_length:.1f}"
            )
        else:
            logger.info(f"  {difficulty}: 0 new puzzles")

    # Show total counts from JSONL files
    instances_dir = setup_instances_directory()
    logger.info("Total puzzles saved:")
    for difficulty in DIFFICULTY_CONFIGS.keys():
        jsonl_path = get_jsonl_path(instances_dir, difficulty)
        total_count = count_existing_puzzles(jsonl_path)
        logger.info(f"  {difficulty}: {total_count} total puzzles")

    # Create unified JSON file for the puzzle loader
    logger.info("Creating unified JSON file for puzzle loader...")
    create_unified_json_from_jsonl()
    logger.info("Unified JSON file created successfully!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--create-json":
        # Just create the unified JSON from existing JSONL files
        logger.info("Creating unified JSON file from existing JSONL files...")
        puzzles = create_unified_json_from_jsonl()
        logger.info("Summary of loaded puzzles:")
        for difficulty, puzzle_list in puzzles.items():
            if puzzle_list:
                avg_solution_length = sum(p.solution_length for p in puzzle_list) / len(puzzle_list)
                logger.info(
                    f"  {difficulty}: {len(puzzle_list)} puzzles, avg solution length: {avg_solution_length:.1f}"
                )
            else:
                logger.info(f"  {difficulty}: 0 puzzles")
    else:
        main()
