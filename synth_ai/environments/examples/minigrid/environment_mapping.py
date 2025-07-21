"""
MiniGrid Environment Mapping Module

This module provides functionality to map any integer seed to one of 60 MiniGrid
environments using modulo arithmetic for deterministic and reproducible
environment selection.
"""

from typing import Tuple

# Environment mapping table (60 total environments)
ENVIRONMENT_MAPPING = {
    # Ultra-Easy (0-4)
    0: "MiniGrid-Empty-5x5-v0",
    1: "MiniGrid-Empty-6x6-v0",
    2: "MiniGrid-Empty-Random-5x5-v0",
    3: "MiniGrid-Empty-Random-6x6-v0",
    4: "MiniGrid-GoToDoor-5x5-v0",
    # Easy (5-14)
    5: "MiniGrid-Empty-8x8-v0",
    6: "MiniGrid-FourRooms-v0",
    7: "MiniGrid-DoorKey-5x5-v0",
    8: "MiniGrid-GoToDoor-6x6-v0",
    9: "MiniGrid-GoToDoor-8x8-v0",
    10: "MiniGrid-Fetch-5x5-N2-v0",
    11: "MiniGrid-Fetch-6x6-N2-v0",
    12: "MiniGrid-PutNear-6x6-N2-v0",
    13: "MiniGrid-Unlock-v0",
    14: "MiniGrid-UnlockPickup-v0",
    # Medium (15-29)
    15: "MiniGrid-DoorKey-6x6-v0",
    16: "MiniGrid-DoorKey-8x8-v0",
    17: "MiniGrid-MultiRoom-N2-S4-v0",
    18: "MiniGrid-LavaGapS5-v0",
    19: "MiniGrid-LavaGapS6-v0",
    20: "MiniGrid-LavaGapS7-v0",
    21: "MiniGrid-SimpleCrossingS9N1-v0",
    22: "MiniGrid-SimpleCrossingS9N2-v0",
    23: "MiniGrid-SimpleCrossingS9N3-v0",
    24: "MiniGrid-Fetch-8x8-N3-v0",
    25: "MiniGrid-PutNear-8x8-N3-v0",
    26: "MiniGrid-RedBlueDoors-6x6-v0",
    27: "MiniGrid-RedBlueDoors-8x8-v0",
    28: "MiniGrid-BlockedUnlockPickup-v0",
    29: "MiniGrid-KeyCorridorS3R1-v0",
    # Hard (30-44)
    30: "MiniGrid-DoorKey-16x16-v0",
    31: "MiniGrid-MultiRoom-N4-S5-v0",
    32: "MiniGrid-MultiRoom-N6-v0",
    33: "MiniGrid-LavaCrossingS9N1-v0",
    34: "MiniGrid-LavaCrossingS9N2-v0",
    35: "MiniGrid-LavaCrossingS9N3-v0",
    36: "MiniGrid-LavaCrossingS11N5-v0",
    37: "MiniGrid-SimpleCrossingS11N5-v0",
    38: "MiniGrid-KeyCorridorS3R2-v0",
    39: "MiniGrid-KeyCorridorS3R3-v0",
    40: "MiniGrid-KeyCorridorS4R3-v0",
    41: "MiniGrid-KeyCorridorS5R3-v0",
    42: "MiniGrid-KeyCorridorS6R3-v0",
    43: "MiniGrid-MemoryS7-v0",
    44: "MiniGrid-MemoryS9-v0",
    # Ultra-Hard (45-54)
    45: "MiniGrid-MemoryS11-v0",
    46: "MiniGrid-MemoryS13-v0",
    47: "MiniGrid-MemoryS13Random-v0",
    48: "MiniGrid-MemoryS17Random-v0",
    49: "MiniGrid-LockedRoom-v0",
    50: "MiniGrid-ObstructedMaze-1Dlh-v0",
    51: "MiniGrid-ObstructedMaze-1Dlhb-v0",
    52: "MiniGrid-ObstructedMaze-2Dlhb-v0",
    53: "MiniGrid-ObstructedMaze-Full-v0",
    54: "MiniGrid-DistShift1-v0",
    # Specialized (55-59)
    55: "MiniGrid-DistShift2-v0",
    56: "MiniGrid-Dynamic-Obstacles-8x8-v0",
    57: "MiniGrid-Dynamic-Obstacles-16x16-v0",
    58: "MiniGrid-Playground-v0",
    59: "MiniGrid-Empty-16x16-v0",
}

# Difficulty mapping
DIFFICULTY_MAPPING = {
    "ultra-easy": (0, 4),
    "easy": (5, 14),
    "medium": (15, 29),
    "hard": (30, 44),
    "ultra-hard": (45, 54),
    "specialized": (55, 59),
}


def get_environment_from_seed(seed: int, hash_seed: bool = True) -> str:
    """
    Map any integer seed to a MiniGrid environment name.

    Args:
        seed: Integer seed
        hash_seed: If True, hash the seed for better distribution

    Returns:
        Environment name string
    """
    if hash_seed:
        # Use hash for better distribution of sequential seeds
        env_index = hash(seed) % 60
    else:
        # Simple modulo
        env_index = seed % 60

    return ENVIRONMENT_MAPPING[env_index]


def get_difficulty_from_seed(seed: int, hash_seed: bool = True) -> str:
    """
    Get difficulty level for a given seed.

    Args:
        seed: Integer seed
        hash_seed: If True, hash the seed for better distribution

    Returns:
        Difficulty level string
    """
    if hash_seed:
        env_index = hash(seed) % 60
    else:
        env_index = seed % 60

    if env_index <= 4:
        return "ultra-easy"
    elif env_index <= 14:
        return "easy"
    elif env_index <= 29:
        return "medium"
    elif env_index <= 44:
        return "hard"
    elif env_index <= 54:
        return "ultra-hard"
    else:
        return "specialized"


def get_minigrid_environment(seed: int, hash_seed: bool = True) -> Tuple[str, str]:
    """
    Get MiniGrid environment name and difficulty from seed.

    Args:
        seed: Integer seed
        hash_seed: If True, hash the seed for better distribution

    Returns:
        (environment_name, difficulty_level)
    """
    env_name = get_environment_from_seed(seed, hash_seed)
    difficulty = get_difficulty_from_seed(seed, hash_seed)

    return env_name, difficulty


def get_environment_by_difficulty(difficulty: str, seed: int = 0) -> str:
    """
    Get a random environment from a specific difficulty level.

    Args:
        difficulty: Difficulty level ("ultra-easy", "easy", "medium", "hard", "ultra-hard", "specialized")
        seed: Seed for selecting within the difficulty range

    Returns:
        Environment name string
    """
    if difficulty not in DIFFICULTY_MAPPING:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    start, end = DIFFICULTY_MAPPING[difficulty]
    range_size = end - start + 1
    env_index = start + (seed % range_size)

    return ENVIRONMENT_MAPPING[env_index]


def get_curriculum_environment(progress: float, seed: int = 0) -> Tuple[str, str]:
    """
    Select environment based on curriculum progress.

    Args:
        progress: Progress value from 0.0 to 1.0
        seed: Seed for environment selection within difficulty

    Returns:
        (environment_name, difficulty_level)
    """
    if progress < 0.2:  # Early stage - ultra-easy
        difficulty = "ultra-easy"
    elif progress < 0.4:  # Beginning - easy
        difficulty = "easy"
    elif progress < 0.6:  # Intermediate - medium
        difficulty = "medium"
    elif progress < 0.8:  # Advanced - hard
        difficulty = "hard"
    else:  # Expert - ultra-hard/specialized
        if progress < 0.9:
            difficulty = "ultra-hard"
        else:
            difficulty = "specialized"

    env_name = get_environment_by_difficulty(difficulty, seed)
    return env_name, difficulty


def validate_environment_name(env_name: str) -> bool:
    """
    Check if an environment name is supported.

    Args:
        env_name: Environment name to validate

    Returns:
        True if supported, False otherwise
    """
    return env_name in ENVIRONMENT_MAPPING.values()


def get_all_environments() -> list[str]:
    """Get list of all supported environment names."""
    return list(ENVIRONMENT_MAPPING.values())


def get_environments_by_difficulty(difficulty: str) -> list[str]:
    """
    Get all environments for a specific difficulty level.

    Args:
        difficulty: Difficulty level

    Returns:
        List of environment names
    """
    if difficulty not in DIFFICULTY_MAPPING:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    start, end = DIFFICULTY_MAPPING[difficulty]
    return [ENVIRONMENT_MAPPING[i] for i in range(start, end + 1)]
