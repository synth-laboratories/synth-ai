from __future__ import annotations

import pytest

from synth_ai.cli.lib.apps.task_app import _validate_rollout_payload


def _base_payload() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "metrics": {},
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


def test_rollout_payload_accepts_objective_metrics() -> None:
    payload = _base_payload()
    payload["metrics"] = {"outcome_objectives": {"reward": 1.0}}
    _validate_rollout_payload(payload)


def test_rollout_payload_accepts_legacy_metrics() -> None:
    payload = _base_payload()
    payload["metrics"] = {"episode_rewards": [1.0], "reward_mean": 1.0, "num_steps": 1}
    _validate_rollout_payload(payload)


def test_rollout_payload_rejects_missing_reward_fields() -> None:
    payload = _base_payload()
    payload["metrics"] = {"num_steps": 1}
    with pytest.raises(ValueError, match="missing required reward fields"):
        _validate_rollout_payload(payload)


def test_rollout_payload_rejects_bad_objective_values() -> None:
    payload = _base_payload()
    payload["metrics"] = {"outcome_objectives": {"reward": "bad"}}
    with pytest.raises(ValueError, match="outcome_objectives"):
        _validate_rollout_payload(payload)
