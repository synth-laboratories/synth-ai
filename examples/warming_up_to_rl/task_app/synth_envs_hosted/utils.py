"""Utility functions for the task service."""

from typing import Any
from urllib.parse import urlparse, urlunparse

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


_CHAT_COMPLETIONS_SUFFIX = "/v1/chat/completions"


def force_normalize_chat_completions_url(raw_url: Any) -> Any:
    """
    Convert ANY malformed inference URL into the correct chat-completions form.
    Ensures path ends with /v1/chat/completions and that query has no '/' segments.
    """
    if not isinstance(raw_url, str):
        return raw_url
    url = raw_url.strip()
    if not url:
        return raw_url

    parsed = urlparse(url)
    path = (parsed.path or "").rstrip("/")
    query = parsed.query or ""

    # If query contains a path, extract and repair
    if query and "/" in query:
        before_slash, after_slash = query.split("/", 1)
        cut_positions = [i for i in [after_slash.find("&"), after_slash.find("?")] if i >= 0]
        cut = min(cut_positions) if cut_positions else len(after_slash)
        path_from_query = "/" + after_slash[:cut]
        extra_query = after_slash[cut + 1 :] if cut < len(after_slash) else ""
        merged_query = before_slash if before_slash else ""
        if extra_query:
            merged_query = f"{merged_query}&{extra_query}" if merged_query else extra_query
        final_path = (
            path_from_query
            if path_from_query.startswith(_CHAT_COMPLETIONS_SUFFIX)
            else f"{path_from_query.rstrip('/')}{_CHAT_COMPLETIONS_SUFFIX}"
        )
        parsed = parsed._replace(path=final_path, query=merged_query)
        url = urlunparse(parsed)
        parsed = urlparse(url)
        path = parsed.path or ""
        query = parsed.query or ""

    # Ensure path suffix
    if not path.endswith(_CHAT_COMPLETIONS_SUFFIX):
        new_path = f"{path}{_CHAT_COMPLETIONS_SUFFIX}" if path else _CHAT_COMPLETIONS_SUFFIX
        parsed = parsed._replace(path=new_path)
        url = urlunparse(parsed)
        parsed = urlparse(url)
        path = parsed.path or ""
        query = parsed.query or ""

    # Last-resort: strip any '/' from query
    if query and "/" in query:
        safe_query = query.split("/")[0]
        parsed = parsed._replace(query=safe_query)
        url = urlunparse(parsed)

    return url


def ensure_chat_completions_url(raw_url: Any, mode: Any = None) -> Any:
    """
    Mode-aware normalizer (RL/EVAL) that returns a valid chat completions URL and
    preserves existing query parameters.
    """
    # For now reuse force normalizer in both modes to guarantee correctness
    return force_normalize_chat_completions_url(raw_url)
