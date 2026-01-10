"""Helper utilities for building RolloutResponse with proper trace correlation."""

from typing import Any

from synth_ai.sdk.task.contracts import RolloutMetrics, RolloutRequest, RolloutResponse


def build_rollout_response(
    request: RolloutRequest,
    outcome_reward: float,
    inference_url: str | None = None,
    trace: dict[str, Any] | None = None,
    policy_config: dict[str, Any] | None = None,
    **kwargs,
) -> RolloutResponse:
    """Build a RolloutResponse from a RolloutRequest.

    This helper ensures that trace_correlation_id is properly echoed from
    the request, which is required for trace hydration to work.

    Args:
        request: The original rollout request (contains trace_correlation_id)
        outcome_reward: The reward/score for this rollout
        inference_url: The inference URL used (optional, extracted from policy_config if not provided)
        trace: Optional trace payload
        policy_config: Optional - only needed if inference_url not provided
        **kwargs: Additional metrics kwargs (event_rewards, etc.)

    Returns:
        RolloutResponse with trace_correlation_id echoed from request

    Example:
        >>> response = build_rollout_response(
        ...     request=request,
        ...     outcome_reward=0.95,
        ...     inference_url=request.policy.config.get("inference_url"),
        ... )
    """
    # Extract inference URL from policy_config if not provided
    if inference_url is None and policy_config:
        inference_url = policy_config.get("inference_url")

    return RolloutResponse(
        trace_correlation_id=request.trace_correlation_id,
        metrics=RolloutMetrics(outcome_reward=outcome_reward, **kwargs),
        trace=trace,
        inference_url=str(inference_url or ""),
    )
