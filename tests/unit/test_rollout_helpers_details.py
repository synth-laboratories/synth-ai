from __future__ import annotations

from synth_ai.sdk.container._impl.contracts import RolloutEnvSpec, RolloutPolicySpec, RolloutRequest
from synth_ai.sdk.container._impl.rollout_helpers import build_rollout_response


def test_build_rollout_response_preserves_details() -> None:
    request = RolloutRequest(
        trace_correlation_id="trace_test",
        env=RolloutEnvSpec(env_name="test", config={}, seed=0),
        policy=RolloutPolicySpec(config={"inference_url": "http://example.invalid"}),
    )

    response = build_rollout_response(
        request=request,
        outcome_reward=0.5,
        inference_url="http://example.invalid",
        policy_config={"model": "mock"},
        details={"trajectory_len": 3, "achievements": ["foo"]},
    )

    assert response.reward_info.outcome_reward == 0.5
    assert response.reward_info.details == {"trajectory_len": 3, "achievements": ["foo"]}

