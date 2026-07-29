from __future__ import annotations

import pytest
from synth_ai.core.research.public import (
    ActorModel,
    CredentialProvider,
    InferenceProvider,
    ProviderBinding,
    ResourceProvider,
    ResourceRoutingPolicy,
    RunPolicyAccess,
    SwarmSpec,
    ToolProvider,
)


def test_release_models_expose_exact_laguna_and_modal_k3_routes() -> None:
    assert ActorModel.LAGUNA_S_2_1_NVFP4.value == "synth_internal/laguna-s-2.1-nvfp4"
    assert ActorModel.KIMI_K3_MODAL.value == "modal/moonshotai/Kimi-K3"
    assert (
        SwarmSpec(objective="k3", provider=InferenceProvider.MODAL).to_wire()["provider"] == "modal"
    )


def test_swarm_provider_selection_serializes_auto_pin_and_ordered_allowlist() -> None:
    assert (
        SwarmSpec(objective="auto", provider=InferenceProvider.AUTO).to_wire()["provider"] == "auto"
    )
    assert (
        SwarmSpec(objective="pin", provider=InferenceProvider.SYNTH).to_wire()["provider"]
        == "synth"
    )
    assert SwarmSpec(
        objective="allowlist",
        provider=[InferenceProvider.SYNTH, "openai"],
    ).to_wire()["provider"] == ["synth", "openai"]


def test_swarm_provider_field_preserves_existing_positional_order() -> None:
    binding = ProviderBinding(provider=ResourceProvider.OPENROUTER)
    spec = SwarmSpec(
        "positional",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        (),
        None,
        (binding,),
    )

    assert spec.providers == (binding,)
    assert spec.provider is None


@pytest.mark.parametrize(
    "provider",
    [
        [],
        ["auto"],
        ["auto", "openai"],
        ["openai", "openai"],
        ["baseten"],
        ["runpod"],
        ["tinker"],
    ],
)
def test_swarm_provider_selection_rejects_invalid_public_values(
    provider: list[str],
) -> None:
    with pytest.raises(ValueError):
        SwarmSpec(objective="invalid", provider=provider)


def test_routing_policy_serializes_compliance_constraints() -> None:
    assert ResourceRoutingPolicy(
        require_zdr=True,
        require_no_training=True,
        max_retention_days=0,
        allowed_domiciles=("us",),
        allowed_regions=("us-east",),
    ).to_wire() == {
        "allowed_domiciles": ["us"],
        "allowed_regions": ["us-east"],
        "require_zdr": True,
        "require_no_training": True,
        "max_retention_days": 0,
    }
    with pytest.raises(ValueError, match="non-negative integer"):
        ResourceRoutingPolicy(max_retention_days=True)


def test_tinker_cannot_be_selected_for_new_launches() -> None:
    with pytest.raises(ValueError, match="deprecated"):
        ProviderBinding(provider=ResourceProvider.TINKER)
    with pytest.raises(ValueError, match="deprecated"):
        RunPolicyAccess(credential_providers=(CredentialProvider.TINKER,))
    with pytest.raises(ValueError, match="deprecated"):
        RunPolicyAccess(tool_providers=(ToolProvider.TINKER,))


def test_legacy_launch_models_match_provider_contract_and_tinker_ban() -> None:
    from synth_ai.core.research.contracts.run_launch import (
        RunLaunchRequest as LegacyRunLaunchRequest,
    )
    from synth_ai.core.research.contracts.run_state import (
        ManagedResearchRun,
    )
    from synth_ai.core.research.contracts.smr_credential_providers import (
        SmrCredentialProvider,
    )
    from synth_ai.core.research.contracts.smr_inference_providers import (
        SmrInferenceProvider,
        coerce_smr_inference_provider,
    )
    from synth_ai.core.research.contracts.smr_providers import (
        ResourceProvider as LegacyResourceProvider,
    )
    from synth_ai.core.research.contracts.smr_providers import (
        ResourceProviderBinding as LegacyResourceProviderBinding,
    )
    from synth_ai.core.research.contracts.smr_providers import (
        ResourceRoutingPolicy as LegacyResourceRoutingPolicy,
    )
    from synth_ai.core.research.contracts.smr_providers import (
        default_provider_policy,
    )
    from synth_ai.core.research.contracts.smr_run_policy import (
        SmrRunPolicy,
        SmrRunPolicyAccess,
    )
    from synth_ai.core.research.contracts.smr_tool_providers import (
        SmrToolProvider,
    )

    assert coerce_smr_inference_provider("synth") is SmrInferenceProvider.SYNTH
    assert coerce_smr_inference_provider("cursor") is SmrInferenceProvider.CURSOR
    assert LegacyResourceRoutingPolicy(
        require_no_training=True,
        max_retention_days=0,
    ).to_dict() == {
        "require_no_training": True,
        "max_retention_days": 0,
    }
    legacy_tinker_binding = LegacyResourceProviderBinding(
        provider=LegacyResourceProvider.TINKER,
    )
    legacy_tinker_access = SmrRunPolicyAccess(
        credential_providers=(SmrCredentialProvider.TINKER,),
    )
    legacy_policy = default_provider_policy()
    positional_launch = LegacyRunLaunchRequest(
        None,
        None,
        None,
        1,
        (),
        legacy_policy,
    )
    assert positional_launch.provider_policy == legacy_policy
    assert positional_launch.provider is None
    from synth_ai.core.research.session.client import (
        _build_project_run_payload,
    )

    with pytest.raises(ValueError, match="deprecated"):
        _build_project_run_payload(
            providers=[legacy_tinker_binding],
        )
    with pytest.raises(ValueError, match="deprecated"):
        _build_project_run_payload(
            run_policy=SmrRunPolicy(access=legacy_tinker_access),
        )
    with pytest.raises(ValueError, match="deprecated"):
        _build_project_run_payload(
            run_policy=SmrRunPolicy(
                access=SmrRunPolicyAccess(
                    tool_providers=(SmrToolProvider.TINKER,),
                )
            ),
        )
    with pytest.raises(ValueError, match="deprecated"):
        _build_project_run_payload(
            roles={
                "orchestrator": {
                    "model": "gpt-5.4-mini",
                    "provider": "tinker",
                },
                "reviewer": {"model": "gpt-5.4-mini"},
                "worker": {
                    "permitted_models": ["gpt-5.4-mini"],
                    "default_model": "gpt-5.4-mini",
                },
            }
        )
    assert _build_project_run_payload(
        provider_policy={
            "default": {
                "require_no_training": True,
                "max_retention_days": 0,
            }
        }
    )["provider_policy"]["default"] == {
        "require_no_training": True,
        "max_retention_days": 0,
    }
    historical_run = ManagedResearchRun.from_wire(
        {
            "run_id": "run_1",
            "project_id": "project_1",
            "public_state": "queued",
            "providers": [{"provider": "tinker"}],
        }
    )
    assert historical_run.providers[0].provider is LegacyResourceProvider.TINKER


def test_provider_selection_threads_through_legacy_and_mcp_launches() -> None:
    from synth_ai.core.research.contracts.run_launch import (
        RunLaunchRequest as LegacyRunLaunchRequest,
    )
    from synth_ai.core.research.session.client import (
        _build_project_run_payload,
    )
    from synth_ai.mcp.research.request_models import (
        OneOffRunLaunchRequest,
        RunLaunchRequest,
    )

    requested = ["synth", "openai"]
    assert _build_project_run_payload(provider=requested)["provider"] == requested
    assert LegacyRunLaunchRequest(
        intended_horizon_hours=1,
        provider=requested,
    ).to_client_kwargs()["provider"] == tuple(requested)

    project_request = RunLaunchRequest.from_payload(
        {"intended_horizon_hours": 1, "provider": requested}
    )
    one_off_request = OneOffRunLaunchRequest.from_payload(
        {"intended_horizon_hours": 1, "provider": "cursor"}
    )
    assert project_request.client_kwargs()["provider"] == requested
    assert one_off_request.client_kwargs()["provider"] == "cursor"
