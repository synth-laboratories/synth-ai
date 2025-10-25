"""Utility functions for the task service."""

import logging
from typing import Any
from urllib.parse import parse_qs, urlparse, urlunparse

import numpy as np

logger = logging.getLogger(__name__)

_CHAT_COMPLETIONS_SUFFIX = "/v1/chat/completions"


def ensure_chat_completions_url(raw_url: Any, mode: str | None = None) -> Any:
    """
    Ensure inference URLs point at the chat completions endpoint.
    
    Args:
        raw_url: The inference URL to process
        mode: "rl" applies URL transformations, "eval" uses URLs as-is (deprecated - use RolloutMode enum)
        
    Returns:
        Processed URL (transformed in RL mode, unchanged in EVAL mode)
    """
    # In EVAL mode, use URLs exactly as provided - no transformations
    # Accept both string "eval" (legacy) and RolloutMode.EVAL
    from synth_ai.task.contracts import RolloutMode
    is_eval_mode = (mode == "eval" or mode == RolloutMode.EVAL or 
                    (hasattr(mode, 'value') and mode.value == "eval"))
    
    if is_eval_mode:
        logger.info("ensure_chat_completions_url: EVAL mode - using URL as-is: %s", raw_url)
        return raw_url
        
    # RL mode: apply transformations for compatibility
    if not isinstance(raw_url, str):
        logger.debug("ensure_chat_completions_url: non-string input %r (type=%s)", raw_url, type(raw_url))
        return raw_url
    url = raw_url.strip()
    if not url:
        logger.debug("ensure_chat_completions_url: blank/whitespace URL input")
        return raw_url

    parsed = urlparse(url)
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/v1/chat/completions"):
        logger.debug("ensure_chat_completions_url: URL already normalized %s", url)
        # Already targeting the desired endpoint; keep original to preserve trailing slash.
        return url

    if not path:
        new_path = _CHAT_COMPLETIONS_SUFFIX
    else:
        new_path = f"{path}{_CHAT_COMPLETIONS_SUFFIX}"

    rebuilt = parsed._replace(path=new_path)
    normalized = urlunparse(rebuilt)
    logger.info(
        "ensure_chat_completions_url: RL mode - normalized inference URL from %s to %s",
        url,
        normalized,
    )
    return normalized


def inference_url_to_trace_correlation_id(raw_url: Any, *, required: bool = False, mode: Any = None) -> str | None:
    """
    Extract trace_correlation_id from inference URL query params.
    
    The inference URL should contain ?cid=trace_xxxxx parameter.
    This is THE canonical source for trace_correlation_id - it's what the
    inference server uses to tag traces, so we extract it here.
    
    Args:
        raw_url: Inference URL (should contain ?cid=... query param)
        required: If True, raises AssertionError if trace_correlation_id not found
        mode: RolloutMode or string ("rl" or "eval"). Controls warning behavior - 
              warnings only logged for RL mode, not EVAL mode.
    
    Returns:
        trace_correlation_id if found in URL, None otherwise
        
    Raises:
        AssertionError: If required=True and trace_correlation_id not found
    """
    if not isinstance(raw_url, str):
        logger.debug(
            "inference_url_to_trace_correlation_id: non-string input %r (type=%s)",
            raw_url,
            type(raw_url)
        )
        if required:
            raise AssertionError(
                f"FATAL: inference_url_to_trace_correlation_id requires string URL, got {type(raw_url)}: {raw_url!r}"
            )
        return None
    
    parsed = urlparse(raw_url)
    query_params = parse_qs(parsed.query or "")
    
    # Check all possible parameter names (cid is primary)
    candidates = (
        query_params.get("cid") or 
        query_params.get("trace") or 
        query_params.get("trace_correlation_id") or 
        []
    )
    
    for value in candidates:
        if isinstance(value, str) and value.strip():
            correlation_id = value.strip()
            logger.info(
                "inference_url_to_trace_correlation_id: ✅ extracted id=%s from url=%s",
                correlation_id,
                raw_url,
            )
            # ASSERTION: Correlation ID should look like trace_xxxxx
            assert correlation_id.startswith("trace_"), (
                f"FATAL: trace_correlation_id has unexpected format: {correlation_id!r}. "
                f"Expected to start with 'trace_'"
            )
            return correlation_id
    
    # Not found - check if we're in EVAL mode (trace_correlation_id not required for eval)
    from synth_ai.task.contracts import RolloutMode
    is_eval_mode = (mode == "eval" or mode == RolloutMode.EVAL or 
                    (hasattr(mode, 'value') and mode.value == "eval"))
    
    if is_eval_mode:
        # For EVAL mode, missing trace_correlation_id is expected - log as debug, not warning
        logger.debug(
            "inference_url_to_trace_correlation_id: No trace_correlation_id in EVAL mode (expected) url=%s query_params=%s",
            raw_url,
            list(query_params.keys())
        )
    else:
        # For RL mode, missing trace_correlation_id is concerning
        logger.warning(
            "inference_url_to_trace_correlation_id: ❌ NO trace_correlation_id found in url=%s query_params=%s",
            raw_url,
            list(query_params.keys())
        )
    
    if required:
        raise AssertionError(
            f"FATAL: trace_correlation_id REQUIRED but not found in inference_url!\n"
            f"\n"
            f"URL: {raw_url}\n"
            f"Query params found: {list(query_params.keys())}\n"
            f"\n"
            f"The inference_url MUST contain ?cid=trace_xxxxx parameter.\n"
            f"This is set by the trainer when generating rollout requests.\n"
        )
    
    return None


# Legacy alias for backward compatibility
def extract_trace_correlation_id(raw_url: Any, mode: Any = None) -> str | None:
    """DEPRECATED: Use inference_url_to_trace_correlation_id instead."""
    return inference_url_to_trace_correlation_id(raw_url, required=False, mode=mode)


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
