from synth_ai.sdk.research.session.factories import FactoriesAPI


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
