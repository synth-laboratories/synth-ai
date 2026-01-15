"""Helpers for trace correlation IDs and trace payloads in task apps.

This module provides utilities for task apps to:
1. Validate trace_correlation_id presence
2. Include trace_correlation_id in rollout responses (top-level, metadata, trace)
3. Build v3 trace payloads for trace-only responses

NOTE: As of the contracts update, trace_correlation_id is now a REQUIRED field
in RolloutRequest. For new code, simply use request.trace_correlation_id directly.

See monorepo/trace_creation_and_judgement.txt "Fatal Guards" section for requirements.
"""

import logging
from datetime import UTC
from typing import Any

logger = logging.getLogger(__name__)


def validate_trace_correlation_id(
    trace_correlation_id: str | None,
    policy_config: dict[str, Any] | None = None,
    fatal: bool = False,
    *,
    run_id: str | None = None,  # Deprecated, for logging only
) -> str | None:
    """
    Validate that trace_correlation_id is present.

    NOTE: With the updated contracts, trace_correlation_id is now a REQUIRED
    field in RolloutRequest. This function is primarily for validating
    extraction from legacy sources (inference URLs).

    Args:
        trace_correlation_id: The correlation ID to validate
        policy_config: Policy configuration for debugging (optional)
        fatal: If True, raise ValueError on missing ID. If False, log error only.
        run_id: DEPRECATED - Only used for logging. Use trace_correlation_id for identification.

    Returns:
        trace_correlation_id if present, None if missing (when fatal=False)

    Raises:
        ValueError: If trace_correlation_id is missing and fatal=True
    """
    if not trace_correlation_id:
        inference_url = (
            policy_config.get("inference_url", "NOT_SET") if policy_config else "NOT_SET"
        )
        id_for_log = run_id or trace_correlation_id or "UNKNOWN"
        error_msg = (
            "CRITICAL: Cannot extract trace_correlation_id!\n"
            "\n"
            f"ID: {id_for_log}\n"
            f"Inference URL: {inference_url}\n"
            "\n"
            "Checked:\n"
            "1. inference_url path segments\n"
            "2. inference_url query params\n"
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
    run_id: str | None = None,
) -> dict[str, Any]:
    """
    Include trace_correlation_id in all required locations of rollout response.

    Required locations:
    1. Top-level response["trace_correlation_id"]
    2. response["trace"]["metadata"]["trace_correlation_id"] (and session_trace metadata if present)

    Args:
        response_data: RolloutResponse dict (from .model_dump())
        trace_correlation_id: The correlation ID to include
        run_id: DEPRECATED - Only used for logging. Use trace_correlation_id for identification.

    Returns:
        Modified response_data with trace_correlation_id in all required places
    """
    id_for_log = run_id or trace_correlation_id or "UNKNOWN"
    if not trace_correlation_id:
        logger.error(
            "include_trace_correlation_id_in_response: missing trace_correlation_id "
            "for id=%s - cannot include in response",
            id_for_log,
        )
        return response_data

    # 1. Add to top-level (REQUIRED)
    if "trace_correlation_id" not in response_data:
        response_data["trace_correlation_id"] = trace_correlation_id
        logger.debug(
            "include_trace_correlation_id: added to top-level id=%s cid=%s",
            id_for_log,
            trace_correlation_id,
        )

    # 2. Add to trace metadata (REQUIRED)
    trace_block = response_data.get("trace")
    if isinstance(trace_block, dict):
        trace_meta = trace_block.get("metadata")
        if not isinstance(trace_meta, dict):
            trace_meta = {}
            trace_block["metadata"] = trace_meta
        if "trace_correlation_id" not in trace_meta:
            trace_meta["trace_correlation_id"] = trace_correlation_id
        corr_ids = trace_meta.get("correlation_ids")
        corr_map = dict(corr_ids) if isinstance(corr_ids, dict) else {}
        corr_map.setdefault("trace_correlation_id", trace_correlation_id)
        trace_meta["correlation_ids"] = corr_map

        session_trace = trace_block.get("session_trace")
        if isinstance(session_trace, dict):
            session_meta = session_trace.get("metadata")
            if not isinstance(session_meta, dict):
                session_meta = {}
                session_trace["metadata"] = session_meta
            session_meta.setdefault("trace_correlation_id", trace_correlation_id)

    logger.debug(
        "include_trace_correlation_id: completed id=%s cid=%s "
        "added to top-level and trace metadata",
        id_for_log,
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
        elif (
            "choices" in response
            and isinstance(response.get("choices"), list)
            and response["choices"]
        ):
            first_choice = (
                response["choices"][0] if isinstance(response["choices"][0], dict) else {}
            )
            llm_response = {
                "message": first_choice.get("message") if isinstance(first_choice, dict) else {},
                "usage": response.get("usage", {}),
                "finish_reason": first_choice.get("finish_reason")
                if isinstance(first_choice, dict)
                else None,
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
        corr_map = dict(corr_ids) if isinstance(corr_ids, dict) else {}
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

    logger.debug(
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
    run_id: str | None = None,
) -> bool:
    """
    Verify that trace_correlation_id is present in all required locations.

    Args:
        response_data: RolloutResponse dict to verify
        expected_correlation_id: The correlation ID that should be present
        run_id: DEPRECATED - Only used for logging. Use trace_correlation_id for identification.

    Returns:
        True if all required locations have the correlation ID, False otherwise
    """
    id_for_log = run_id or expected_correlation_id or "UNKNOWN"
    if not expected_correlation_id:
        logger.error(
            "verify_trace_correlation_id: no expected_correlation_id provided for id=%s", id_for_log
        )
        return False

    errors = []

    # Check top-level
    if response_data.get("trace_correlation_id") != expected_correlation_id:
        errors.append(
            f"Top-level missing or mismatch: "
            f"expected={expected_correlation_id} actual={response_data.get('trace_correlation_id')}"
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
        logger.error("verify_trace_correlation_id: FAILED id=%s\n%s", id_for_log, "\n".join(errors))
        return False

    logger.debug(
        "verify_trace_correlation_id: PASSED id=%s cid=%s", id_for_log, expected_correlation_id
    )
    return True
