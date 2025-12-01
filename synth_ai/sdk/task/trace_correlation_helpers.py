"""Helpers for trace correlation ID extraction and inclusion in task apps.

This module provides utilities for task apps to:
1. Extract trace_correlation_id from rollout requests
2. Include trace_correlation_id in rollout responses (3 required locations)

See monorepo/trace_creation_and_judgement.txt "Fatal Guards" section for requirements.
"""

import importlib
import logging
from datetime import UTC
from typing import Any
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


def extract_trace_correlation_id(
    policy_config: dict[str, Any],
    inference_url: str | None = None,
    mode: Any = None
) -> str | None:
    """
    Extract trace_correlation_id from policy config or inference URL.
    
    This is the standardized method for all task apps to extract the correlation ID
    that the RL trainer generates and passes to the task app.
    
    Args:
        policy_config: Policy configuration dict from RolloutRequest.policy.config
        inference_url: Inference URL (optional, used as fallback)
        mode: RolloutMode or string ("rl" or "eval"). Controls warning behavior - 
              warnings only logged for RL mode, not EVAL mode.
        
    Returns:
        trace_correlation_id if found, None otherwise
        
    Extraction order:
        1. policy_config["trace_correlation_id"] (preferred)
        2. policy_config["trace"] (legacy fallback)
        3. URL query param ?cid=... (fallback)
        4. URL query param ?trace_correlation_id=... (fallback)
    """
    # Try policy_config first (preferred method)
    candidates: list[Any] = [
        policy_config.get("trace_correlation_id"),
        policy_config.get("trace"),
    ]
    
    logger.debug(
        "extract_trace_correlation_id: policy_cfg keys=%s candidates=%s",
        sorted(policy_config.keys()),
        candidates,
    )
    
    for candidate in candidates:
        if isinstance(candidate, str):
            stripped = candidate.strip()
            if stripped:
                logger.info(
                    "extract_trace_correlation_id: extracted from policy_config=%s",
                    stripped
                )
                return stripped
    
    # Determine if we're in EVAL mode (trace_correlation_id not required for eval)
    rollout_mode_cls: Any | None = None
    try:
        contracts_module = importlib.import_module("synth_ai.sdk.task.contracts")
        rollout_mode_cls = getattr(contracts_module, "RolloutMode", None)
    except Exception:
        rollout_mode_cls = None

    is_eval_mode = False
    if rollout_mode_cls is not None:
        try:
            is_eval_mode = (
                mode == "eval"
                or mode == rollout_mode_cls.EVAL
                or getattr(mode, "value", None) == "eval"
            )
        except Exception:
            is_eval_mode = mode == "eval"
    else:
        is_eval_mode = mode == "eval" or getattr(mode, "value", None) == "eval"
    
    # Fallback: try to extract from inference_url query params
    if not inference_url or not isinstance(inference_url, str):
        if is_eval_mode:
            logger.debug(
                "extract_trace_correlation_id: no correlation ID found in policy_config "
                "and no inference_url provided (EVAL mode - expected)"
            )
        else:
            logger.warning(
                "extract_trace_correlation_id: no correlation ID found in policy_config "
                "and no inference_url provided"
            )
        return None
    
    try:
        parsed = urlparse(inference_url)
        query_params = parse_qs(parsed.query or "")
        # Try multiple possible query param names
        for param_name in ["cid", "trace_correlation_id", "trace"]:
            values = query_params.get(param_name)
            if not values:
                continue
            for value in values:
                if isinstance(value, str) and value.strip():
                    correlation_id = value.strip()
                    logger.info(
                        "extract_trace_correlation_id: extracted from URL param %s=%s",
                        param_name,
                        correlation_id,
                    )
                    return correlation_id
    except Exception as e:
        logger.warning(
            "extract_trace_correlation_id: failed to parse inference_url=%s error=%s",
            inference_url,
            e,
        )
    
    if is_eval_mode:
        logger.debug(
            "extract_trace_correlation_id: no trace_correlation_id found in "
            "policy_config or inference_url=%s (EVAL mode - expected)",
            inference_url,
        )
    else:
        logger.warning(
            "extract_trace_correlation_id: no trace_correlation_id found in "
            "policy_config or inference_url=%s",
            inference_url,
        )
    return None


def validate_trace_correlation_id(
    trace_correlation_id: str | None,
    run_id: str,
    policy_config: dict[str, Any],
    fatal: bool = False
) -> str | None:
    """
    Validate that trace_correlation_id was successfully extracted.
    
    Args:
        trace_correlation_id: The extracted correlation ID (or None)
        run_id: Rollout run_id for logging
        policy_config: Policy configuration for debugging
        fatal: If True, raise ValueError on missing ID. If False, log error only.
        
    Returns:
        trace_correlation_id if present, None if missing (when fatal=False)
        
    Raises:
        ValueError: If trace_correlation_id is missing and fatal=True
    """
    if not trace_correlation_id:
        error_msg = (
            f"🚨 CRITICAL: Cannot extract trace_correlation_id!\n"
            "\n"
            f"Run ID: {run_id}\n"
            f"Policy config keys: {sorted(policy_config.keys())}\n"
            f"Inference URL: {policy_config.get('inference_url', 'NOT_SET')}\n"
            "\n"
            "Checked:\n"
            f"1. policy_config['trace_correlation_id']: {policy_config.get('trace_correlation_id')}\n"
            f"2. policy_config['trace']: {policy_config.get('trace')}\n"
            f"3. inference_url query params\n"
            "\n"
            "Task app CANNOT proceed without trace_correlation_id.\n"
            "This indicates the RL trainer is not sending it correctly.\n"
            "\n"
            "See monorepo/trace_creation_and_judgement.txt 'Fatal Guards' section.\n"
        )
        
        if fatal:
            raise ValueError(error_msg)
        else:
            logger.error(error_msg)
    
    return trace_correlation_id


def include_trace_correlation_id_in_response(
    response_data: dict[str, Any],
    trace_correlation_id: str | None,
    run_id: str
) -> dict[str, Any]:
    """
    Include trace_correlation_id in all required locations of rollout response.
    
    Required locations (per Fatal Guards section):
    1. Top-level response["trace_correlation_id"]
    2. response["pipeline_metadata"]["trace_correlation_id"]
    3. Each trajectory["trace_correlation_id"]
    
    Args:
        response_data: RolloutResponse dict (from .model_dump())
        trace_correlation_id: The correlation ID to include
        run_id: Rollout run_id for logging
        
    Returns:
        Modified response_data with trace_correlation_id in all required places
    """
    if not trace_correlation_id:
        logger.error(
            "include_trace_correlation_id_in_response: missing trace_correlation_id "
            "for run_id=%s - cannot include in response",
            run_id
        )
        return response_data
    
    # 1. Add to top-level (REQUIRED)
    if "trace_correlation_id" not in response_data:
        response_data["trace_correlation_id"] = trace_correlation_id
        logger.info(
            "include_trace_correlation_id: added to top-level run_id=%s cid=%s",
            run_id,
            trace_correlation_id
        )
    
    # 2. Add to pipeline_metadata (REQUIRED)
    pipeline_meta = response_data.get("pipeline_metadata")
    if not isinstance(pipeline_meta, dict):
        pipeline_meta = {}
        response_data["pipeline_metadata"] = pipeline_meta
    
    if "trace_correlation_id" not in pipeline_meta:
        pipeline_meta["trace_correlation_id"] = trace_correlation_id
        logger.info(
            "include_trace_correlation_id: added to pipeline_metadata run_id=%s cid=%s",
            run_id,
            trace_correlation_id
        )
    
    # 3. Add to each trajectory (REQUIRED)
    trajectories = response_data.get("trajectories", [])
    if isinstance(trajectories, list):
        for idx, traj in enumerate(trajectories):
            if isinstance(traj, dict) and "trace_correlation_id" not in traj:
                traj["trace_correlation_id"] = trace_correlation_id
                logger.debug(
                    "include_trace_correlation_id: added to trajectory[%d] run_id=%s cid=%s",
                    idx,
                    run_id,
                    trace_correlation_id
                )
    
    logger.info(
        "include_trace_correlation_id: completed run_id=%s cid=%s "
        "added to %d locations (top-level, metadata, %d trajectories)",
        run_id,
        trace_correlation_id,
        2 + len(trajectories),
        len(trajectories)
    )
    
    return response_data


def build_trajectory_trace(
    messages: list[dict[str, Any]],
    response: dict[str, Any] | None = None,
    *,
    correlation_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a trajectory-level trace with event_history for trace strict mode.

    This creates the trace structure required by monorepo's trace_validation.py:
    - trajectory.trace.event_history must be non-empty
    - event_history contains LM call records for input/output extraction

    Args:
        messages: The messages sent to the LLM (input)
        response: The LLM response dict (output). Should include 'choices' or 'content'.
        correlation_id: Trace correlation ID (from ?cid= param)
        session_id: Optional session ID for the trace
        metadata: Optional additional metadata

    Returns:
        A trace dict with event_history suitable for trajectory.trace

    Example:
        trace = build_trajectory_trace(
            messages=[{"role": "user", "content": "Hello"}],
            response={"choices": [{"message": {"content": "Hi!"}}]},
            correlation_id="trace_abc123",
        )
        trajectory = RolloutTrajectory(..., trace=trace)
    """
    import uuid
    from datetime import datetime

    # Build event_history with LM call record
    event_history: list[dict[str, Any]] = []

    # Create an LM call event (the primary event type for input/output extraction)
    lm_event: dict[str, Any] = {
        "event_type": "lm_call",
        "timestamp": datetime.now(UTC).isoformat(),
        "call_record": {
            "messages": messages,
            "response": response or {},
        },
    }

    # Add correlation ID if provided
    if correlation_id:
        lm_event["correlation_id"] = correlation_id

    event_history.append(lm_event)

    trace: dict[str, Any] = {
        "session_id": session_id or str(uuid.uuid4()),
        "event_history": event_history,
        "created_at": datetime.now(UTC).isoformat(),
    }

    if correlation_id:
        trace["correlation_id"] = correlation_id

    if metadata:
        trace["metadata"] = metadata

    logger.debug(
        "build_trajectory_trace: created trace with %d events, session_id=%s, cid=%s",
        len(event_history),
        trace["session_id"],
        correlation_id,
    )

    return trace


def include_event_history_in_trajectories(
    response_data: dict[str, Any],
    messages_by_trajectory: list[list[dict[str, Any]]] | None = None,
    responses_by_trajectory: list[dict[str, Any]] | None = None,
    *,
    run_id: str,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """
    Ensure all trajectories have trace.event_history for trace strict mode.

    This satisfies monorepo's trace_validation.py requirement:
    - validate_response_has_hydrated_trace() checks for event_history

    Args:
        response_data: RolloutResponse dict (from .model_dump())
        messages_by_trajectory: List of messages for each trajectory (for building event_history)
        responses_by_trajectory: List of LLM responses for each trajectory
        run_id: Rollout run_id for logging
        correlation_id: Trace correlation ID

    Returns:
        Modified response_data with event_history in each trajectory.trace
    """
    trajectories = response_data.get("trajectories", [])
    if not isinstance(trajectories, list):
        logger.warning(
            "include_event_history_in_trajectories: trajectories is not a list for run_id=%s",
            run_id,
        )
        return response_data

    for idx, traj in enumerate(trajectories):
        if not isinstance(traj, dict):
            continue

        # Get existing trace or create new one
        trace = traj.get("trace")
        if not isinstance(trace, dict):
            trace = {}
            traj["trace"] = trace

        # Check if event_history already exists and is non-empty
        event_history = trace.get("event_history")
        if isinstance(event_history, list) and len(event_history) > 0:
            logger.debug(
                "include_event_history_in_trajectories: trajectory[%d] already has "
                "%d events, skipping run_id=%s",
                idx,
                len(event_history),
                run_id,
            )
            continue

        # Build event_history from provided messages/responses
        messages = (
            messages_by_trajectory[idx]
            if messages_by_trajectory and idx < len(messages_by_trajectory)
            else []
        )
        response = (
            responses_by_trajectory[idx]
            if responses_by_trajectory and idx < len(responses_by_trajectory)
            else None
        )

        # If no messages provided, try to extract from trajectory steps
        if not messages:
            steps = traj.get("steps", [])
            for step in steps:
                if isinstance(step, dict):
                    obs = step.get("obs", {})
                    if isinstance(obs, dict):
                        step_messages = obs.get("messages")
                        if isinstance(step_messages, list):
                            messages = step_messages
                            break

        # Build the trace with event_history
        new_trace = build_trajectory_trace(
            messages=messages,
            response=response,
            correlation_id=correlation_id or traj.get("trace_correlation_id"),
            metadata={"run_id": run_id, "trajectory_index": idx},
        )

        # Merge with existing trace (preserve existing fields)
        trace.update(new_trace)

        logger.info(
            "include_event_history_in_trajectories: added event_history to "
            "trajectory[%d] run_id=%s events=%d",
            idx,
            run_id,
            len(trace.get("event_history", [])),
        )

    return response_data


def verify_trace_correlation_id_in_response(
    response_data: dict[str, Any],
    expected_correlation_id: str | None,
    run_id: str
) -> bool:
    """
    Verify that trace_correlation_id is present in all required locations.
    
    Args:
        response_data: RolloutResponse dict to verify
        expected_correlation_id: The correlation ID that should be present
        run_id: Rollout run_id for logging
        
    Returns:
        True if all required locations have the correlation ID, False otherwise
    """
    if not expected_correlation_id:
        logger.error(
            "verify_trace_correlation_id: no expected_correlation_id provided for run_id=%s",
            run_id
        )
        return False
    
    errors = []
    
    # Check top-level
    if response_data.get("trace_correlation_id") != expected_correlation_id:
        errors.append(
            f"Top-level missing or mismatch: "
            f"expected={expected_correlation_id} actual={response_data.get('trace_correlation_id')}"
        )
    
    # Check pipeline_metadata
    pipeline_meta = response_data.get("pipeline_metadata", {})
    if not isinstance(pipeline_meta, dict) or pipeline_meta.get("trace_correlation_id") != expected_correlation_id:
        errors.append(
            f"pipeline_metadata missing or mismatch: "
            f"expected={expected_correlation_id} actual={pipeline_meta.get('trace_correlation_id') if isinstance(pipeline_meta, dict) else 'NOT_A_DICT'}"
        )
    
    # Check trajectories
    trajectories = response_data.get("trajectories", [])
    if isinstance(trajectories, list):
        for idx, traj in enumerate(trajectories):
            if isinstance(traj, dict) and traj.get("trace_correlation_id") != expected_correlation_id:
                errors.append(
                    f"trajectory[{idx}] missing or mismatch: "
                    f"expected={expected_correlation_id} actual={traj.get('trace_correlation_id')}"
                )
    
    if errors:
        logger.error(
            "verify_trace_correlation_id: FAILED run_id=%s\n%s",
            run_id,
            "\n".join(errors)
        )
        return False
    
    logger.info(
        "verify_trace_correlation_id: PASSED run_id=%s cid=%s",
        run_id,
        expected_correlation_id
    )
    return True
