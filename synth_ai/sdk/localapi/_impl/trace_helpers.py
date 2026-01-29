"""Helpers for trace correlation IDs and trace payloads in task apps."""

from __future__ import annotations

import warnings
from typing import Any

import synth_ai_py


def validate_trace_correlation_id(
    trace_correlation_id: str | None,
    policy_config: dict[str, Any] | None = None,
    fatal: bool = False,
    *,
    run_id: str | None = None,
) -> str | None:
    """Validate that trace_correlation_id is present."""

    if run_id is not None:
        warnings.warn(
            "run_id is deprecated, use trace_correlation_id instead. Will be removed in v2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    inference_url = policy_config.get("inference_url") if policy_config else None
    return synth_ai_py.localapi_validate_trace_correlation_id(
        trace_correlation_id, inference_url, fatal
    )


def include_trace_correlation_id_in_response(
    response_data: dict[str, Any],
    trace_correlation_id: str | None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Include trace_correlation_id in all required locations of rollout response."""

    if run_id is not None:
        warnings.warn(
            "run_id is deprecated, use trace_correlation_id instead. Will be removed in v2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    return synth_ai_py.localapi_include_trace_correlation_id_in_response(
        response_data, trace_correlation_id
    )


def build_trace_payload(
    messages: list[dict[str, Any]],
    response: dict[str, Any] | None = None,
    *,
    correlation_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a v4 trace payload with event_history for trace-only responses."""

    return synth_ai_py.localapi_build_trace_payload(
        messages,
        response,
        correlation_id=correlation_id,
        session_id=session_id,
        metadata=metadata,
    )


def build_trajectory_trace(
    messages: list[dict[str, Any]],
    response: dict[str, Any] | None = None,
    *,
    correlation_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias for build_trace_payload."""

    return synth_ai_py.localapi_build_trajectory_trace(
        messages,
        response,
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
    """Ensure response.trace includes a v4 event_history payload."""

    return synth_ai_py.localapi_include_event_history_in_response(
        response_data,
        messages,
        response,
        run_id=run_id,
        correlation_id=correlation_id,
    )


def include_event_history_in_trajectories(
    response_data: dict[str, Any],
    messages_by_trajectory: list[list[dict[str, Any]]] | None = None,
    responses_by_trajectory: list[dict[str, Any]] | None = None,
    *,
    run_id: str,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias for include_event_history_in_response."""

    return synth_ai_py.localapi_include_event_history_in_trajectories(
        response_data,
        messages_by_trajectory,
        responses_by_trajectory,
        run_id=run_id,
        correlation_id=correlation_id,
    )


def verify_trace_correlation_id_in_response(
    response_data: dict[str, Any],
    expected_correlation_id: str | None,
    run_id: str | None = None,
) -> bool:
    """Verify that trace_correlation_id is present in all required locations."""

    if run_id is not None:
        warnings.warn(
            "run_id is deprecated, use trace_correlation_id instead. Will be removed in v2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    return synth_ai_py.localapi_verify_trace_correlation_id_in_response(
        response_data, expected_correlation_id
    )
