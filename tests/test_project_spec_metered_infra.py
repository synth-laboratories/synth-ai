from __future__ import annotations

from synth_ai.sdk.research.contracts.projects import ProjectSpec


def test_project_spec_serializes_metered_inference_contract() -> None:
    spec = ProjectSpec(
        name="REB hosted inference",
        pool_id="default",
        runtime_kind="sandbox_agent",
        environment_kind="harbor",
        orchestrator_profile_id="orchestrator",
        default_worker_profile_id="worker",
        metered_infra={
            "inference": {
                "enabled": True,
                "provider": "openai",
                "model": "gpt-4o-mini",
                "openai_compatible": True,
            }
        },
    )

    assert spec.to_wire()["metered_infra"] == {
        "inference": {
            "enabled": True,
            "provider": "openai",
            "model": "gpt-4o-mini",
            "openai_compatible": True,
        }
    }


def test_project_spec_defaults_to_an_empty_metered_infra_contract() -> None:
    spec = ProjectSpec(
        name="No metered inference",
        pool_id="default",
        runtime_kind="sandbox_agent",
        environment_kind="harbor",
        orchestrator_profile_id="orchestrator",
        default_worker_profile_id="worker",
    )

    assert spec.to_wire()["metered_infra"] == {}
