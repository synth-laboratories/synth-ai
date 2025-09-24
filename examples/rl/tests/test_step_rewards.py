from __future__ import annotations

from examples.rl.crafter_task_app_helpers.rewards import compute_decision_rewards


def test_compute_decision_rewards_basic():
    dec = [{"indicator_i": 0}, {"indicator_i": 1}, {"indicator_i": 0}]
    out = compute_decision_rewards(decision_summaries=dec, total_achievements=4, step_beta=0.1, indicator_lambda=1.0)
    assert len(out) == 3
    # r1 = (3-1)*0.1*4 + 0 = 0.8
    assert abs(out[0]["reward"] - 0.8) < 1e-6
    # r2 = (3-2)*0.1*4 + 1 = 1.4
    assert abs(out[1]["reward"] - 1.4) < 1e-6
    # r3 = (3-3)*0.1*4 + 0 = 0.0
    assert abs(out[2]["reward"] - 0.0) < 1e-6


def test_compute_decision_rewards_empty():
    out = compute_decision_rewards(decision_summaries=[], total_achievements=0, step_beta=0.0, indicator_lambda=0.0)
    assert out == []


