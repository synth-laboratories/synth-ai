"""Utility functions for the task service."""

import logging
from typing import Any
from urllib.parse import parse_qs, urlparse, urlunparse

import numpy as np

logger = logging.getLogger(__name__)

_CHAT_COMPLETIONS_SUFFIX = "/v1/chat/completions"


def force_normalize_chat_completions_url(raw_url: Any) -> str:
    """
    Bulletproof normalizer: converts ANY malformed inference URL into the
    correct chat-completions URL form.

    Rules:
    - Final path MUST end with /v1/chat/completions
    - Query MUST NOT contain any '/' characters (no path segments in query)
    - If the original query contained a path (e.g., '?cid=.../v1/chat/completions'),
      extract that path and move it to the URL path; keep remaining query params
    - Preserve scheme, host, port and existing query params order as much as possible

    Examples:
      https://host?cid=trace_123/v1/chat/completions
        -> https://host/v1/chat/completions?cid=trace_123
      https://host:8000?cid=trace_abc/v1/chat/completions&foo=bar
        -> https://host:8000/v1/chat/completions?cid=trace_abc&foo=bar
      https://host?cid=trace_123/v1/chat/completions?other=param
        -> https://host/v1/chat/completions?cid=trace_123&other=param
    """
    if not isinstance(raw_url, str):
        return raw_url
    url = raw_url.strip()
    if not url:
        return raw_url

    parsed = urlparse(url)
    path = (parsed.path or "").rstrip("/")
    query = parsed.query or ""

    # If query contains a path (has '/'), extract and repair
    if query and "/" in query:
        # Split query at the first '/' (everything before is real query params)
        before_slash, after_slash = query.split("/", 1)

        # after_slash may contain path and then more query params separated by '&' or '?' (malformed)
        sep_indices = [i for i in [after_slash.find("&"), after_slash.find("?")] if i >= 0]
        cut_idx = min(sep_indices) if sep_indices else len(after_slash)
        path_from_query = "/" + after_slash[:cut_idx]  # restore leading '/'
        extra_query = after_slash[cut_idx + 1 :] if cut_idx < len(after_slash) else ""

        # Merge query params: base (before_slash) + extra_query
        merged_query = before_slash
        if extra_query:
            merged_query = f"{merged_query}&{extra_query}" if merged_query else extra_query

        # Decide final path
        if path_from_query.startswith(_CHAT_COMPLETIONS_SUFFIX):
            final_path = path_from_query
        else:
            final_path = f"{path_from_query.rstrip('/')}{_CHAT_COMPLETIONS_SUFFIX}"

        parsed = parsed._replace(path=final_path, query=merged_query)
        url = urlunparse(parsed)
        parsed = urlparse(url)
        path = parsed.path or ""
        query = parsed.query or ""

    # Ensure path ends with chat completions suffix
    if not path.endswith(_CHAT_COMPLETIONS_SUFFIX):
        new_path = f"{path}{_CHAT_COMPLETIONS_SUFFIX}" if path else _CHAT_COMPLETIONS_SUFFIX
        parsed = parsed._replace(path=new_path)
        url = urlunparse(parsed)
        parsed = urlparse(url)
        path = parsed.path or ""
        query = parsed.query or ""

    # Final validation: no '/' in query
    if query and "/" in query:
        # As a last resort, drop anything after the first '/'
        safe_query = query.split("/")[0]
        parsed = parsed._replace(query=safe_query)
        url = urlunparse(parsed)

    return url


def _validate_url_structure(url: str, context: str = "") -> None:
    """
    Validate that a URL has correct structure (path before query, not vice versa).
    
    Raises ValueError if URL is malformed.
    
    Args:
        url: The URL to validate
        context: Optional context for error messages
        
    Raises:
        ValueError: If URL is malformed (path-like segments in query string)
    """
    if not isinstance(url, str) or not url.strip():
        return
    
    try:
        parsed = urlparse(url)
        query = parsed.query or ""
        
        # CRITICAL CHECK: If query contains path-like segments (contains /), it's malformed
        if query and "/" in query:
            path_segment = query.split("/", 1)[1] if "/" in query else ""
            error_msg = (
                f"FATAL [TASK_APP_URL_VALIDATION]: Malformed inference URL detected!\n"
                f"\n"
                f"URL: {url}\n"
                f"Context: {context}\n"
                f"\n"
                f"The URL has a path-like segment ('/{path_segment}') in the query string.\n"
                f"This indicates incorrect URL construction upstream.\n"
                f"\n"
                f"Expected: https://host/v1/chat/completions?cid=trace_123\n"
                f"Malformed: https://host?cid=trace_123/v1/chat/completions\n"
                f"\n"
                f"This should be caught by the trainer, but if you see this,\n"
                f"the trainer's URL validation may have failed.\n"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    except ValueError:
        raise
    except Exception as e:
        logger.warning(f"[URL_VALIDATION] Failed to parse URL: {url} (context: {context}, error: {e})")


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
    query = parsed.query
    
    logger.debug(
        "ensure_chat_completions_url: parsing url=%s -> path=%r query=%r",
        url,
        path,
        query,
    )
    
    # CRITICAL: Check for malformed URLs (path in query) and fix them FIRST
    # Example: https://host?cid=trace_123/v1/chat/completions
    # Should be: https://host/v1/chat/completions?cid=trace_123
    if query and "/" in query:
        logger.error(
            f"[URL_FIX] Detected malformed URL in ensure_chat_completions_url: {url}\n"
            f"Path-like segment found in query string. Attempting to fix..."
        )
        # Split query at first "/" to separate query params from path
        query_parts = query.split("/", 1)
        if len(query_parts) == 2:
            # query_parts[0] is the actual query (e.g., "cid=trace_123")
            # query_parts[1] is the path that was incorrectly put in query
            actual_query = query_parts[0]
            path_and_more = query_parts[1]  # Could be "v1/chat/completions" or "v1/chat/completions&foo=bar"
            
            # Extract the path part (everything before "&" or "?" if present)
            # Handle both "&" (query param separator) and "?" (another malformed query separator)
            if "&" in path_and_more:
                # Path is followed by more query params (separated by &)
                path_segment, extra_query = path_and_more.split("&", 1)
                path_in_query = "/" + path_segment  # Restore leading slash
                # Merge extra query params with actual_query
                actual_query = f"{actual_query}&{extra_query}"
            elif "?" in path_and_more:
                # Path is followed by more query params (separated by ?, which is malformed)
                path_segment, extra_query = path_and_more.split("?", 1)
                path_in_query = "/" + path_segment  # Restore leading slash
                # Merge extra query params with actual_query (use & as separator)
                actual_query = f"{actual_query}&{extra_query}"
            else:
                # No extra query params, just the path
                path_in_query = "/" + path_and_more  # Restore leading slash
            
            # If the path_in_query already contains /v1/chat/completions, use it
            # Otherwise, append /v1/chat/completions
            if path_in_query.startswith("/v1/chat/completions"):
                final_path = path_in_query
            else:
                # Append /v1/chat/completions to whatever path we found
                final_path = path_in_query.rstrip("/") + "/v1/chat/completions"
            
            # Reconstruct URL correctly: path comes before query
            parsed = parsed._replace(path=final_path, query=actual_query)
            fixed_url = urlunparse(parsed)
            logger.warning(f"[URL_FIX] Fixed malformed URL:\n  FROM: {url}\n  TO:   {fixed_url}")
            url = fixed_url
            # Re-parse after fix
            parsed = urlparse(url)
            path = parsed.path.rstrip("/")
            query = parsed.query
        else:
            # Can't parse - this shouldn't happen but validate will catch it
            logger.error(f"[URL_FIX] Could not parse malformed query: {query}")
            _validate_url_structure(url, context="ensure_chat_completions_url input - cannot fix")
    
    if path.endswith("/v1/chat/completions"):
        logger.debug("ensure_chat_completions_url: URL already normalized %s", url)
        # Validate final URL
        _validate_url_structure(url, context="ensure_chat_completions_url output")
        return url

    if not path:
        new_path = _CHAT_COMPLETIONS_SUFFIX
    else:
        new_path = f"{path}{_CHAT_COMPLETIONS_SUFFIX}"

    rebuilt = parsed._replace(path=new_path)
    normalized = urlunparse(rebuilt)
    
    # CRITICAL: Validate the normalized URL
    _validate_url_structure(normalized, context="ensure_chat_completions_url output")
    
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
