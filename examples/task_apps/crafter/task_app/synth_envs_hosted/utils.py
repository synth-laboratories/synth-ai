"""Utility functions for the task service."""

from typing import Any

import numpy as np


def convert_numpy_to_python(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.

    Args:
        obj: Object that may contain numpy types

    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list | tuple):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj


def sanitize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize observation data for JSON serialization.

    Converts numpy types and removes non-serializable objects.

    Args:
        observation: Raw observation from environment

    Returns:
        Sanitized observation safe for JSON serialization
    """
    if not isinstance(observation, dict):
        return observation

    sanitized = {}
    for key, value in observation.items():
        # Skip non-serializable keys or convert them
        if key in ["semantic_map", "world_material_map", "observation_image"]:
            # These are likely numpy arrays - convert to lists or skip
            if isinstance(value, np.ndarray):
                # For large arrays, we might want to skip or compress
                # For now, skip them as they're likely debug info
                continue
        elif key == "player_position" and isinstance(value, tuple):
            # Convert tuple with potential numpy types
            sanitized[key] = [convert_numpy_to_python(v) for v in value]
        else:
            sanitized[key] = convert_numpy_to_python(value)

    return sanitized
