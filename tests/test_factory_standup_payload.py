import pytest
from synth_ai.sdk.research.session.factories import FactoriesAPI, _effort_kwargs


def test_plan_standup_preserves_factory_lifecycle_fields() -> None:
    homeostasis_policy = {
        "sampling": {
            "drift_check": "disabled",
            "heartbeat_every_n_runs": 0,
            "forced": [],
        }
    }

    plan = FactoriesAPI.plan_standup(
        {
            "factory": {
                "name": "bounded-eval",
                "status": "active",
                "result_authority_generation": "current",
                "homeostasis_policy": homeostasis_policy,
            },
            "project": {"project_id": "project-1"},
            "efforts": [{"name": "effort-1"}],
        }
    )

    assert plan.factory["status"] == "active"
    assert plan.factory["result_authority_generation"] == "current"
    assert plan.factory["homeostasis_policy"] == homeostasis_policy


def test_standup_effort_collapses_equal_delay_aliases() -> None:
    payload = _effort_kwargs(
        {
            "name": "effort-1",
            "recurrence_policy": {
                "on_run_complete": True,
                "interval_seconds": 0,
                "delay_seconds": 0,
            },
        },
        default_project_id="project-1",
    )

    assert payload["recurrence_policy"] == {
        "on_run_complete": True,
        "delay_seconds": 0,
    }


def test_standup_effort_rejects_conflicting_delay_aliases() -> None:
    with pytest.raises(ValueError, match="delay aliases must have the same value"):
        _effort_kwargs(
            {
                "name": "effort-1",
                "recurrence_policy": {
                    "interval_seconds": 0,
                    "delay_seconds": 30,
                },
            },
            default_project_id="project-1",
        )
