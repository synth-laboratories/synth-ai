"""Helpers for trace correlation ID extraction and inclusion in task apps."""

from __future__ import annotations

import importlib
import logging
import warnings
from typing import Any

import synth_ai_py

logger = logging.getLogger(__name__)


def _is_eval_mode(mode: Any) -> bool:
    rollout_mode_cls: Any | None = None
    try:
        contracts_module = importlib.import_module("synth_ai.sdk.localapi._impl.contracts")
        rollout_mode_cls = getattr(contracts_module, "RolloutMode", None)
    except Exception:
        rollout_mode_cls = None

    if rollout_mode_cls is not None:
        try:
            return (
                mode == "eval"
                or mode == rollout_mode_cls.EVAL
                or getattr(mode, "value", None) == "eval"
            )
        except Exception:
            return mode == "eval"
    return mode == "eval" or getattr(mode, "value", None) == "eval"


def extract_trace_correlation_id(
    policy_config: dict[str, Any], inference_url: str | None = None, mode: Any = None
) -> str | None:
    """Extract trace_correlation_id from inference URL only."""

    is_eval_mode = _is_eval_mode(mode)

    if not inference_url or not isinstance(inference_url, str):
        if is_eval_mode:
            logger.debug(
                "extract_trace_correlation_id: no inference_url provided (EVAL mode - expected)"
            )
        else:
            logger.warning("extract_trace_correlation_id: no inference_url provided")
        return None

    correlation_id = synth_ai_py.localapi_extract_trace_correlation_id(inference_url)
    if correlation_id:
        return correlation_id

    if is_eval_mode:
        logger.debug(
            "extract_trace_correlation_id: no trace_correlation_id found in inference_url=%s (EVAL mode - expected)",
            inference_url,
        )
    else:
        logger.warning(
            "extract_trace_correlation_id: no trace_correlation_id found in inference_url=%s",
            inference_url,
        )
    return None


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
