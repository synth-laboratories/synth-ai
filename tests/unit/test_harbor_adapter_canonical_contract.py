from __future__ import annotations

from types import SimpleNamespace

from synth_ai.sdk.container._impl.contracts import RolloutRequest
from synth_ai.sdk.container.harbor_adapter import HarborExecutionBackend


def _backend() -> HarborExecutionBackend:
    return HarborExecutionBackend(
        deployment_ref=SimpleNamespace(
            deployment_id="dep_12345678",
            api_key="sk_test",
            rollout_url="https://example.test/rollout",
        )
    )


def test_transform_request_uses_trace_correlation_id_only() -> None:
    backend = _backend()
    request = RolloutRequest(
        trace_correlation_id="trace_1",
        env={},
        policy={},
    )
    payload = backend._transform_request(request)
    assert payload["trace_correlation_id"] == "trace_1"
    assert "run_id" not in payload


def test_transform_response_ignores_reward_mean_fallback() -> None:
    backend = _backend()
    request = RolloutRequest(
        trace_correlation_id="trace_2",
        env={},
        policy={},
    )
    response = backend._transform_response(
        {
            "trace_correlation_id": "trace_2",
            "metrics": {"reward_mean": 0.9},
        },
        request,
    )
    assert response.reward_info.outcome_reward == 0.0
