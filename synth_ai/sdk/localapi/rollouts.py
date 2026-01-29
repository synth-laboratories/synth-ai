"""Helpers for building LocalAPI rollout responses.

## Usage

    response = RolloutResponseBuilder.trace_only(
        trace_correlation_id=request.trace_correlation_id,
        reward=1.0,
        trace=trace_payload,
        inference_url="https://api.usesynth.ai/v1/trial-xyz",
    )

## Key Fields

- `trace_correlation_id`: REQUIRED - Single source of truth for rollout identification
- `reward`: The outcome reward (required) → `metrics.outcome_reward`
- `inference_url`: Inference URL used (top-level)
"""

from __future__ import annotations

from typing import Any

from synth_ai.sdk.localapi._impl.contracts import RolloutMetrics, RolloutResponse


class RolloutResponseBuilder:
    """Convenience builders for rollout responses."""

    @staticmethod
    def trace_only(
        *,
        trace_correlation_id: str,
        reward: float,
        trace: dict[str, Any] | None,
        event_rewards: list[float] | None = None,
        outcome_objectives: dict[str, float] | None = None,
        event_objectives: list[dict[str, float]] | None = None,
        instance_objectives: list[dict[str, float]] | None = None,
        inference_url: str | None = None,
        details: dict[str, Any] | None = None,
        artifact: list[Any] | None = None,
        success_status: Any | None = None,
        status_detail: str | None = None,
    ) -> RolloutResponse:
        """Build a RolloutResponse with standardized metrics.

        Args:
            trace_correlation_id: REQUIRED - Correlation ID (echo from request)
            reward: Outcome reward for this rollout
            trace: v3/v4 trace payload
            event_rewards: Optional per-step rewards for multi-step tasks
            inference_url: Inference URL used for this rollout
            details: Metadata dict (debugging info, not rewards)
        """
        metrics = RolloutMetrics(
            outcome_reward=float(reward),
            event_rewards=event_rewards,
            outcome_objectives=outcome_objectives,
            event_objectives=event_objectives,
            instance_objectives=instance_objectives,
            details=details or {},
        )

        return RolloutResponse(
            trace_correlation_id=trace_correlation_id,
            metrics=metrics,
            trace=_with_trace_metadata(trace, trace_correlation_id),
            inference_url=inference_url,
            artifact=artifact,
            success_status=success_status,
            status_detail=status_detail,
        )


def _with_trace_metadata(
    trace: dict[str, Any] | None,
    trace_correlation_id: str | None,
) -> dict[str, Any] | None:
    if trace is None:
        return None
    if not isinstance(trace, dict):
        return trace

    updated = dict(trace)
    metadata = updated.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if trace_correlation_id:
        metadata.setdefault("trace_correlation_id", trace_correlation_id)
        corr_ids = metadata.get("correlation_ids")
        corr_map = dict(corr_ids) if isinstance(corr_ids, dict) else {}
        corr_map.setdefault("trace_correlation_id", trace_correlation_id)
        metadata["correlation_ids"] = corr_map
    updated["metadata"] = metadata
    return updated
