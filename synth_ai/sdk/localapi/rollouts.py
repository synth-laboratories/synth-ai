"""Helpers for building LocalAPI rollout responses."""

from __future__ import annotations

from typing import Any, Mapping

from synth_ai.sdk.task.contracts import RolloutMetrics, RolloutResponse


class RolloutResponseBuilder:
    """Convenience builders for trace-only rollout responses."""

    @staticmethod
    def trace_only(
        *,
        run_id: str,
        reward: float,
        trace: dict[str, Any] | None,
        details: dict[str, Any] | None = None,
        metrics: RolloutMetrics | None = None,
        num_steps: int | None = None,
        num_episodes: int = 1,
        aborted: bool = False,
        trace_correlation_id: str | None = None,
        pipeline_metadata: Mapping[str, Any] | None = None,
        branches: Mapping[str, list[str]] | None = None,
    ) -> RolloutResponse:
        """Build a trace-only RolloutResponse with standardized metrics."""

        reward_value = float(reward)
        metrics_payload = metrics or RolloutMetrics(
            episode_returns=[reward_value],
            mean_return=reward_value,
            num_steps=int(num_steps or 1),
            num_episodes=int(num_episodes),
            outcome_score=reward_value,
            events_score=reward_value,
            details=details or {},
        )

        if details:
            merged_details = dict(metrics_payload.details or {})
            merged_details.update(details)
            metrics_payload.details = merged_details

        trace_payload = _with_trace_metadata(trace, trace_correlation_id)

        pipeline_meta = dict(pipeline_metadata or {})
        if trace_correlation_id:
            pipeline_meta.setdefault("trace_correlation_id", trace_correlation_id)

        response = RolloutResponse(
            run_id=run_id,
            branches=dict(branches or {}),
            metrics=metrics_payload,
            aborted=aborted,
            trace_correlation_id=trace_correlation_id,
            trace=trace_payload,
            pipeline_metadata=pipeline_meta,
        )
        return response


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
