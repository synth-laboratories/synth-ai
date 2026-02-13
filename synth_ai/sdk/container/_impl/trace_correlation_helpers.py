"""Helpers for trace correlation ID extraction and inclusion in containers."""

from __future__ import annotations

import importlib
import logging
import warnings
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    import synth_ai_py
except Exception:  # pragma: no cover
    synth_ai_py = None

logger = logging.getLogger(__name__)


def _fallback_extract_from_url(inference_url: str) -> str | None:
    try:
        parsed = urlparse(inference_url)
        query = parse_qs(parsed.query)
        for key in ("trace_correlation_id", "cid"):
            values = query.get(key)
            if values:
                return values[0]
    except Exception:
        return None
    return None


def _is_eval_mode(mode: Any) -> bool:
    rollout_mode_cls: Any | None = None
    try:
        contracts_module = importlib.import_module("synth_ai.sdk.container._impl.contracts")
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

    if synth_ai_py is not None and hasattr(synth_ai_py, "container_extract_trace_correlation_id"):
        correlation_id = synth_ai_py.container_extract_trace_correlation_id(inference_url)
    else:
        correlation_id = _fallback_extract_from_url(inference_url)
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
    if synth_ai_py is not None and hasattr(synth_ai_py, "container_validate_trace_correlation_id"):
        return synth_ai_py.container_validate_trace_correlation_id(
            trace_correlation_id, inference_url, fatal
        )

    if trace_correlation_id:
        return trace_correlation_id
    if inference_url:
        extracted = _fallback_extract_from_url(str(inference_url))
        if extracted:
            return extracted
    if fatal:
        raise ValueError("trace_correlation_id is required")
    return None


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
    if synth_ai_py is not None and hasattr(
        synth_ai_py, "container_include_trace_correlation_id_in_response"
    ):
        return synth_ai_py.container_include_trace_correlation_id_in_response(
            response_data, trace_correlation_id
        )

    if not trace_correlation_id:
        return response_data
    updated = dict(response_data)
    updated["trace_correlation_id"] = trace_correlation_id
    trace = updated.get("trace")
    if isinstance(trace, dict):
        trace = dict(trace)
        metadata = trace.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("trace_correlation_id", trace_correlation_id)
        corr_ids = metadata.get("correlation_ids")
        corr_map = dict(corr_ids) if isinstance(corr_ids, dict) else {}
        corr_map.setdefault("trace_correlation_id", trace_correlation_id)
        metadata["correlation_ids"] = corr_map
        trace["metadata"] = metadata
        updated["trace"] = trace
    return updated


def build_trace_payload(
    messages: list[dict[str, Any]],
    response: dict[str, Any] | None = None,
    *,
    correlation_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a v4 trace payload with event_history for trace-only responses."""

    if synth_ai_py is not None and hasattr(synth_ai_py, "container_build_trace_payload"):
        return synth_ai_py.container_build_trace_payload(
            messages,
            response,
            correlation_id=correlation_id,
            session_id=session_id,
            metadata=metadata,
        )

    payload: dict[str, Any] = {"messages": messages}
    if response is not None:
        payload["response"] = response
    if metadata:
        payload["metadata"] = dict(metadata)
    if correlation_id:
        payload.setdefault("metadata", {}).setdefault("trace_correlation_id", correlation_id)
    if session_id:
        payload.setdefault("metadata", {}).setdefault("session_id", session_id)
    return payload


def build_trajectory_trace(
    messages: list[dict[str, Any]],
    response: dict[str, Any] | None = None,
    *,
    correlation_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias for build_trace_payload."""

    if synth_ai_py is not None and hasattr(synth_ai_py, "container_build_trajectory_trace"):
        return synth_ai_py.container_build_trajectory_trace(
            messages,
            response,
            correlation_id=correlation_id,
            session_id=session_id,
            metadata=metadata,
        )
    return build_trace_payload(
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

    if synth_ai_py is not None and hasattr(
        synth_ai_py, "container_include_event_history_in_response"
    ):
        return synth_ai_py.container_include_event_history_in_response(
            response_data,
            messages,
            response,
            run_id=run_id,
            correlation_id=correlation_id,
        )
    return include_trace_correlation_id_in_response(response_data, correlation_id)


def include_event_history_in_trajectories(
    response_data: dict[str, Any],
    messages_by_trajectory: list[list[dict[str, Any]]] | None = None,
    responses_by_trajectory: list[dict[str, Any]] | None = None,
    *,
    run_id: str,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias for include_event_history_in_response."""

    if synth_ai_py is not None and hasattr(
        synth_ai_py, "container_include_event_history_in_trajectories"
    ):
        return synth_ai_py.container_include_event_history_in_trajectories(
            response_data,
            messages_by_trajectory,
            responses_by_trajectory,
            run_id=run_id,
            correlation_id=correlation_id,
        )
    return include_trace_correlation_id_in_response(response_data, correlation_id)


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
    if synth_ai_py is not None and hasattr(
        synth_ai_py, "container_verify_trace_correlation_id_in_response"
    ):
        return synth_ai_py.container_verify_trace_correlation_id_in_response(
            response_data, expected_correlation_id
        )

    if not expected_correlation_id:
        return False
    if response_data.get("trace_correlation_id") == expected_correlation_id:
        return True
    trace = response_data.get("trace")
    if isinstance(trace, dict):
        metadata = trace.get("metadata")
        if isinstance(metadata, dict):
            if metadata.get("trace_correlation_id") == expected_correlation_id:
                return True
            corr_ids = metadata.get("correlation_ids")
            if (
                isinstance(corr_ids, dict)
                and corr_ids.get("trace_correlation_id") == expected_correlation_id
            ):
                return True
    return False
