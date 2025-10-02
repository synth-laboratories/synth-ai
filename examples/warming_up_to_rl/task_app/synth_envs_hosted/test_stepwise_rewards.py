import math

from backend.app.routes.clustered_training.dev.synth_envs_hosted.rollout import (
    compute_stepwise_reward,
)


def test_compute_stepwise_reward_detects_unlock() -> None:
    prev = {"build_workbench": False, "pet_cow": False}
    new = {"build_workbench": True, "pet_cow": False}
    actions = [{"tool": "interact", "args": {"action": "build_workbench"}}]

    info, decision, stats = compute_stepwise_reward(
        prev,
        new,
        decision_index=2,
        actions_summary=actions,
        indicator_lambda=0.75,
    )

    assert info["decision_index"] == 2
    assert info["new_achievements"] == ["build_workbench"]
    assert info["indicator"] == 1
    assert math.isclose(info["reward"], 0.75)

    assert decision["actions"] == actions
    assert decision["indicator"] == 1
    assert math.isclose(decision["r_i"], 0.75)

    assert math.isclose(stats["indicator"], 1.0)
    assert math.isclose(stats["reward"], 0.75)
    assert math.isclose(stats["new_achievements_count"], 1.0)


def test_compute_stepwise_reward_handles_no_unlock() -> None:
    prev = {"collect_wheat": True, "milk_cow": False}
    new = {"collect_wheat": True, "milk_cow": False}
    actions = [{"tool": "interact", "args": {"action": "sleep"}}]

    info, decision, stats = compute_stepwise_reward(
        prev,
        new,
        decision_index=5,
        actions_summary=actions,
        indicator_lambda=1.2,
    )

    assert info["new_achievements"] == []
    assert info["indicator"] == 0
    assert math.isclose(info["reward"], 0.0)

    assert decision["actions"] == actions
    assert decision["indicator"] == 0
    assert math.isclose(decision["r_i"], 0.0)

    assert math.isclose(stats["indicator"], 0.0)
    assert math.isclose(stats["reward"], 0.0)
    assert math.isclose(stats["new_achievements_count"], 0.0)
