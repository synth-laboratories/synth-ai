from __future__ import annotations

import pytest

from synth_ai.cli.lib.apps.task_app import _validate_rollout_payload


def _base_payload() -> dict[str, object]:
    return {
        "trace_correlation_id": "trace-123",
        "metrics": {"outcome_reward": 1.0},
        "trajectories": [
            {
                "env_id": "env",
                "policy_id": "policy",
                "steps": [
                    {
                        "info": {
                            "messages": [{"role": "user", "content": "hi"}],
                        }
                    }
                ],
                "length": 1,
                "inference_url": "http://example.com?cid=trace_123",
            }
        ],
    }


def test_rollout_payload_accepts_outcome_reward() -> None:
    """outcome_reward is required and should be accepted."""
    payload = _base_payload()
    payload["metrics"] = {"outcome_reward": 1.0}
    _validate_rollout_payload(payload)


def test_rollout_payload_accepts_outcome_reward_with_objectives() -> None:
    """outcome_reward with outcome_objectives should be accepted."""
    payload = _base_payload()
    payload["metrics"] = {
        "outcome_reward": 1.0,
        "outcome_objectives": {"reward": 1.0, "latency": 0.5},
    }
    _validate_rollout_payload(payload)


def test_rollout_payload_accepts_event_rewards() -> None:
    """outcome_reward with event_rewards should be accepted."""
    payload = _base_payload()
    payload["metrics"] = {
        "outcome_reward": 0.85,
        "event_rewards": [0.8, 0.9, 0.85],
    }
    _validate_rollout_payload(payload)


def test_rollout_payload_rejects_missing_outcome_reward() -> None:
    """Missing outcome_reward should be rejected."""
    payload = _base_payload()
    payload["metrics"] = {"outcome_objectives": {"reward": 1.0}}
    with pytest.raises(ValueError, match="missing required field 'outcome_reward'"):
        _validate_rollout_payload(payload)


def test_rollout_payload_rejects_bad_objective_values() -> None:
    """Non-numeric outcome_objectives values should be rejected."""
    payload = _base_payload()
    payload["metrics"] = {"outcome_reward": 1.0, "outcome_objectives": {"reward": "bad"}}
    with pytest.raises(ValueError, match="outcome_objectives"):
        _validate_rollout_payload(payload)


def test_rollout_payload_rejects_bad_event_rewards() -> None:
    """Non-numeric event_rewards values should be rejected."""
    payload = _base_payload()
    payload["metrics"] = {"outcome_reward": 1.0, "event_rewards": [0.5, "bad", 0.7]}
    with pytest.raises(ValueError, match="event_rewards"):
        _validate_rollout_payload(payload)
