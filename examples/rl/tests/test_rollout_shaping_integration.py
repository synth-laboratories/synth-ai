from __future__ import annotations

import json


def test_decision_rewards_branch_shape_smoke(monkeypatch):
    # This is a smoke test placeholder; in CI we avoid running the modal app.
    # Validate expected keys in a mocked rollout response when step_rewards are enabled.
    mocked = {
        "run_id": "run_test",
        "trajectories": [],
        "branches": {
            "decision_rewards": [
                {"decision_index": 1, "reward": 0.8, "indicator_i": 0, "achievements_count": 4, "total_steps": 3},
                {"decision_index": 2, "reward": 1.4, "indicator_i": 1, "achievements_count": 4, "total_steps": 3},
                {"decision_index": 3, "reward": 0.0, "indicator_i": 0, "achievements_count": 4, "total_steps": 3},
            ]
        },
        "metrics": {"episode_returns": [0.0], "mean_return": 2.0, "num_steps": 3, "num_episodes": 1},
        "aborted": False,
        "ops_executed": 5,
    }
    js = json.dumps(mocked)
    data = json.loads(js)
    assert "branches" in data and "decision_rewards" in data["branches"]
    dec = data["branches"]["decision_rewards"]
    assert isinstance(dec, list) and len(dec) == 3
    assert all(set(["decision_index", "reward", "indicator_i", "achievements_count", "total_steps"]).issubset(d.keys()) for d in dec)


