"""Helper utilities for building RolloutResponse with proper trace correlation."""

from typing import Any

from synth_ai.sdk.task.contracts import RolloutRequest, RolloutResponse, RolloutMetrics
from synth_ai.sdk.task.trace_correlation_helpers import extract_trace_correlation_id


def build_rollout_response(
    request: RolloutRequest,
    outcome_reward: float,
    policy_config: dict[str, Any],
    inference_url: str | None = None,
    trace: dict[str, Any] | None = None,
    **kwargs
) -> RolloutResponse:
    """Build a RolloutResponse with proper trace correlation ID extraction.

    This helper ensures that trace_correlation_id is properly extracted
    from policy_config, which is required for trace hydration to work.

    Args:
        request: The original rollout request
        outcome_reward: The reward/score for this rollout
        policy_config: The policy configuration from request.policy.config
        inference_url: The inference URL used (optional, extracted from policy_config if not provided)
        trace: Optional trace payload
        **kwargs: Additional metrics kwargs (event_rewards, etc.)

    Returns:
        RolloutResponse with properly extracted trace_correlation_id

    Example:
        >>> response = build_rollout_response(
        ...     request=request,
        ...     outcome_reward=0.95,
        ...     policy_config=request.policy.config,
        ...     inference_url=request.policy.config.get("inference_url"),
        ... )
    """
    # Extract inference URL if not provided
    if inference_url is None:
        inference_url = policy_config.get("inference_url")

    # Filter out trace-related keys before extraction
    policy_cfg_for_trace = {
        key: value
        for key, value in policy_config.items()
        if key not in {"trace_correlation_id", "trace"}
    }

    # Extract correlation ID
    trace_correlation_id = extract_trace_correlation_id(
        policy_config=policy_cfg_for_trace,
        inference_url=str(inference_url or ""),
        mode=request.mode,
    )

    return RolloutResponse(
        run_id=request.run_id,
        metrics=RolloutMetrics(outcome_reward=outcome_reward, **kwargs),
        trace=trace,
        trace_correlation_id=trace_correlation_id,
        inference_url=str(inference_url or ""),
    )
