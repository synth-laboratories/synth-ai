"""Helpers for building LocalAPI rollout responses.

## Usage

    response = RolloutResponseBuilder.trace_only(
        run_id=request.run_id,
        reward=1.0,
        trace=trace_payload,
        trace_correlation_id="trace_abc123",
        inference_url="https://api.usesynth.ai/v1/trial-xyz",
    )

## Key Fields

- `reward`: The outcome reward (required) → `metrics.outcome_reward`
- `trace_correlation_id`: Correlation ID for trace recovery (top-level)
- `inference_url`: Inference URL used (top-level)
"""

from __future__ import annotations

from typing import Any

from synth_ai.sdk.task.contracts import RolloutMetrics, RolloutResponse


class RolloutResponseBuilder:
    """Convenience builders for rollout responses."""

    @staticmethod
    def trace_only(
        *,
        run_id: str,
        reward: float,
        trace: dict[str, Any] | None,
        event_rewards: list[float] | None = None,
        trace_correlation_id: str | None = None,
        inference_url: str | None = None,
        details: dict[str, Any] | None = None,
        aborted: bool = False,
    ) -> RolloutResponse:
        """Build a RolloutResponse with standardized metrics.

        Args:
            run_id: Request run_id to echo back
            reward: Outcome reward for this rollout
            trace: v3 trace payload
            event_rewards: Optional per-step rewards for multi-step tasks
            trace_correlation_id: Correlation ID for trace recovery
            inference_url: Inference URL used for this rollout
            details: Metadata dict (debugging info, not rewards)
            aborted: Whether rollout was aborted early
        """
        metrics = RolloutMetrics(
            outcome_reward=float(reward),
            event_rewards=event_rewards,
            details=details or {},
        )

        return RolloutResponse(
            run_id=run_id,
            metrics=metrics,
            trace=_with_trace_metadata(trace, trace_correlation_id),
            trace_correlation_id=trace_correlation_id,
            inference_url=inference_url,
            aborted=aborted,
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
        if isinstance(corr_ids, dict):
            corr_map = dict(corr_ids)
        else:
            corr_map = {}
        corr_map.setdefault("trace_correlation_id", trace_correlation_id)
        metadata["correlation_ids"] = corr_map
    updated["metadata"] = metadata
    return updated
