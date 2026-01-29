"""Helper utilities for building RolloutResponse with proper trace correlation."""

from typing import Any

import synth_ai_py

from synth_ai.sdk.localapi._impl.contracts import RolloutRequest, RolloutResponse


def build_rollout_response(
    request: RolloutRequest,
    outcome_reward: float,
    inference_url: str | None = None,
    trace: dict[str, Any] | None = None,
    policy_config: dict[str, Any] | None = None,
    artifact: list[Any] | None = None,
    success_status: Any | None = None,
    status_detail: str | None = None,
    **kwargs,
) -> RolloutResponse:
    """Build a RolloutResponse from a RolloutRequest."""

    payload = synth_ai_py.localapi_build_rollout_response(
        request,
        outcome_reward,
        inference_url,
        trace,
        policy_config,
        artifact,
        success_status,
        status_detail,
        kwargs if kwargs else None,
    )
    # Ensure we return the contract type
    if isinstance(payload, RolloutResponse):
        return payload
    return RolloutResponse(**payload)
