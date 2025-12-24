"""Helpers for trace correlation ID extraction and inclusion in task apps.

This module provides utilities for task apps to:
1. Extract trace_correlation_id from rollout requests
2. Include trace_correlation_id in rollout responses (top-level, metadata, trace)

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

    Required locations (trace-only):
    1. Top-level response["trace_correlation_id"]
    2. response["pipeline_metadata"]["trace_correlation_id"]
    3. response["trace"]["metadata"]["trace_correlation_id"] (and session_trace metadata if present)
    
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
    
    # 3. Add to trace metadata (REQUIRED)
    trace_block = response_data.get("trace")
    if isinstance(trace_block, dict):
        trace_meta = trace_block.get("metadata")
        if not isinstance(trace_meta, dict):
            trace_meta = {}
            trace_block["metadata"] = trace_meta
        if "trace_correlation_id" not in trace_meta:
            trace_meta["trace_correlation_id"] = trace_correlation_id
        corr_ids = trace_meta.get("correlation_ids")
        if isinstance(corr_ids, dict):
            corr_map = dict(corr_ids)
        else:
            corr_map = {}
        corr_map.setdefault("trace_correlation_id", trace_correlation_id)
        trace_meta["correlation_ids"] = corr_map

        session_trace = trace_block.get("session_trace")
        if isinstance(session_trace, dict):
            session_meta = session_trace.get("metadata")
            if not isinstance(session_meta, dict):
                session_meta = {}
                session_trace["metadata"] = session_meta
            session_meta.setdefault("trace_correlation_id", trace_correlation_id)

    logger.info(
        "include_trace_correlation_id: completed run_id=%s cid=%s "
        "added to top-level, metadata, and trace",
        run_id,
        trace_correlation_id,
    )
    
    return response_data


def build_trace_payload(
    messages: list[dict[str, Any]],
    response: dict[str, Any] | None = None,
    *,
    correlation_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a v3 trace payload with event_history for trace-only responses.

    Args:
        messages: The messages sent to the LLM (input)
        response: The LLM response dict (output). Should include 'choices' or 'content'.
        correlation_id: Trace correlation ID (from ?cid= param)
        session_id: Optional session ID for the trace
        metadata: Optional additional metadata

    Returns:
        A trace dict with event_history suitable for RolloutResponse.trace
    """
    import uuid
    from datetime import datetime

    event_history: list[dict[str, Any]] = []

    llm_response: dict[str, Any] = {}
    if isinstance(response, dict):
        if "message" in response:
            llm_response = dict(response)
        elif "choices" in response and isinstance(response.get("choices"), list) and response["choices"]:
            first_choice = response["choices"][0] if isinstance(response["choices"][0], dict) else {}
            llm_response = {
                "message": first_choice.get("message") if isinstance(first_choice, dict) else {},
                "usage": response.get("usage", {}),
                "finish_reason": first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
            }
        else:
            llm_response = dict(response)

    llm_event: dict[str, Any] = {
        "type": "lm_call",
        "event_type": "lm_call",
        "timestamp": datetime.now(UTC).isoformat(),
        "llm_request": {"messages": messages},
        "llm_response": llm_response,
    }

    # Add correlation ID if provided
    if correlation_id:
        llm_event["correlation_id"] = correlation_id

    event_history.append(llm_event)

    trace_metadata: dict[str, Any] = dict(metadata or {})
    trace_metadata.setdefault("session_id", session_id or str(uuid.uuid4()))
    if correlation_id:
        trace_metadata.setdefault("trace_correlation_id", correlation_id)
        corr_ids = trace_metadata.get("correlation_ids")
        if isinstance(corr_ids, dict):
            corr_map = dict(corr_ids)
        else:
            corr_map = {}
        corr_map.setdefault("trace_correlation_id", correlation_id)
        trace_metadata["correlation_ids"] = corr_map

    trace: dict[str, Any] = {
        "schema_version": "3.0",
        "event_history": event_history,
        "markov_blanket_message_history": [],
        "metadata": trace_metadata,
    }

    logger.debug(
        "build_trace_payload: created trace with %d events, session_id=%s, cid=%s",
        len(event_history),
        trace_metadata.get("session_id"),
        correlation_id,
    )

    return trace


def build_trajectory_trace(
    messages: list[dict[str, Any]],
    response: dict[str, Any] | None = None,
    *,
    correlation_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias for build_trace_payload."""

    return build_trace_payload(
        messages=messages,
        response=response,
        correlation_id=correlation_id,
        session_id=session_id,
        metadata=metadata,
    )


def include_event_history_in_response(
    response_data: dict[str, Any],
    messages: list[dict[str, Any]] | None = None,
    response: dict[str, Any] | None = None,
    *,
    run_id: str,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """
    Ensure response.trace includes a v3 event_history payload.

    Args:
        response_data: RolloutResponse dict (from .model_dump())
        messages: Messages for the LLM call (for building event_history)
        response: LLM response payload
        run_id: Rollout run_id for logging
        correlation_id: Trace correlation ID

    Returns:
        Modified response_data with event_history in response.trace
    """
    trace_block = response_data.get("trace")
    if not isinstance(trace_block, dict):
        trace_block = {}
        response_data["trace"] = trace_block

    event_history = trace_block.get("event_history")
    session_trace = trace_block.get("session_trace")
    if not event_history and isinstance(session_trace, dict):
        event_history = session_trace.get("event_history")

    if isinstance(event_history, list) and event_history:
        return response_data

    new_trace = build_trace_payload(
        messages=messages or [],
        response=response,
        correlation_id=correlation_id,
        metadata={"run_id": run_id},
    )

    # Merge new trace payload into the existing trace block.
    trace_meta = trace_block.get("metadata")
    if isinstance(trace_meta, dict):
        merged_meta = dict(new_trace.get("metadata", {}))
        merged_meta.update(trace_meta)
        trace_block["metadata"] = merged_meta
    else:
        trace_block["metadata"] = new_trace.get("metadata", {})

    trace_block.setdefault("schema_version", new_trace.get("schema_version"))
    trace_block["event_history"] = new_trace.get("event_history", [])
    trace_block.setdefault(
        "markov_blanket_message_history",
        new_trace.get("markov_blanket_message_history", []),
    )

    if isinstance(session_trace, dict) and "event_history" not in session_trace:
        session_trace["event_history"] = trace_block["event_history"]

    logger.info(
        "include_event_history_in_response: added event_history run_id=%s events=%d",
        run_id,
        len(trace_block.get("event_history", [])),
    )
    return response_data


def include_event_history_in_trajectories(
    response_data: dict[str, Any],
    messages_by_trajectory: list[list[dict[str, Any]]] | None = None,
    responses_by_trajectory: list[dict[str, Any]] | None = None,
    *,
    run_id: str,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias for include_event_history_in_response."""

    messages = messages_by_trajectory[0] if messages_by_trajectory else None
    response = responses_by_trajectory[0] if responses_by_trajectory else None
    return include_event_history_in_response(
        response_data,
        messages=messages,
        response=response,
        run_id=run_id,
        correlation_id=correlation_id,
    )


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
    
    # Check trace metadata
    trace_block = response_data.get("trace")
    trace_meta_id = None
    if isinstance(trace_block, dict):
        trace_meta = trace_block.get("metadata")
        if isinstance(trace_meta, dict):
            trace_meta_id = trace_meta.get("trace_correlation_id")
        if trace_meta_id != expected_correlation_id:
            session_trace = trace_block.get("session_trace")
            if isinstance(session_trace, dict):
                session_meta = session_trace.get("metadata")
                if isinstance(session_meta, dict):
                    trace_meta_id = session_meta.get("trace_correlation_id")
        if trace_meta_id != expected_correlation_id:
            errors.append(
                "trace.metadata missing or mismatch: "
                f"expected={expected_correlation_id} actual={trace_meta_id}"
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
