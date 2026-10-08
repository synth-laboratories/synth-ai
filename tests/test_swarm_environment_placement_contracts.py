from __future__ import annotations

from datetime import UTC, datetime

import pytest
from synth_ai.sdk.research.contracts.image_releases import (
    ActorRuntimeImageMaterialization,
    ImageSecurityAdmission,
    SecurityAdmissionState,
)
from synth_ai.sdk.research.contracts.placement import (
    PlacementMode,
    PlacementPolicy,
    RunEnvironmentCatalog,
    SharedPlacementGroup,
)
from synth_ai.sdk.research.contracts.swarms import SwarmSpec
from synth_ai.sdk.research.image_releases import _security_body, _security_mutation
from synth_ai.sdk.research.operations import research_operation


def test_environment_and_hybrid_placement_wire_contract() -> None:
    catalog = RunEnvironmentCatalog(
        allowed=("coordination", "rust", "reviewer"),
        role_defaults={"orchestrator": "coordination", "worker": "rust"},
        task_bindings={"cyber/task-1": "rust"},
        actor_selections={"reviewer-1": "reviewer"},
    )
    placement = PlacementPolicy(
        default=PlacementMode.PER_ACTOR,
        groups={
            "engineering": SharedPlacementGroup(
                environment_release_id="rust", max_actors=4, max_heavy_commands=2
            )
        },
    )
    assert catalog.to_wire()["allowed"] == ["coordination", "rust", "reviewer"]
    assert placement.to_wire()["groups"]["engineering"]["mode"] == "shared_group"
    wire = SwarmSpec(
        objective="test",
        environment_catalog=catalog,
        placement=placement,
    ).to_wire()
    assert wire["environments"]["task_bindings"]["cyber/task-1"] == "rust"
    assert wire["placement"]["default"] == "per_actor"


def test_environment_binding_must_be_in_run_allowlist() -> None:
    with pytest.raises(ValueError, match="allowed"):
        RunEnvironmentCatalog(allowed=("rust",), task_bindings={"task": "python"})


def test_shared_default_requires_one_group_per_allowed_environment() -> None:
    with pytest.raises(ValueError, match="every allowed environment"):
        SwarmSpec(
            objective="test",
            environment_catalog=RunEnvironmentCatalog(allowed=("rust", "python")),
            placement=PlacementPolicy(
                default=PlacementMode.SHARED_GROUP,
                groups={"engineering": SharedPlacementGroup(environment_release_id="rust")},
            ),
        )


def test_environment_cannot_belong_to_multiple_shared_groups() -> None:
    with pytest.raises(ValueError, match="only one"):
        PlacementPolicy(
            groups={
                "engineering-a": SharedPlacementGroup(environment_release_id="rust"),
                "engineering-b": SharedPlacementGroup(environment_release_id="rust"),
            }
        )


def test_shared_group_actor_limit_matches_runtime_port_capacity() -> None:
    with pytest.raises(ValueError, match="at most 64"):
        SharedPlacementGroup(environment_release_id="rust", max_actors=65)


def test_security_admission_round_trip_on_image_materialization() -> None:
    admission = ImageSecurityAdmission(
        admission_id="11111111-1111-1111-1111-111111111111",
        state=SecurityAdmissionState.APPROVED,
        manifest_digest="sha256:" + "a" * 64,
        policy_version="v1",
        generation=2,
        expires_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    image = ActorRuntimeImageMaterialization.from_wire(
        {
            "schema_version": "smr-actor-image-materialization-v1",
            "runtime_image_release_id": "22222222-2222-2222-2222-222222222222",
            "status": "active",
            "image_ref": "registry/image@sha256:" + "a" * 64,
            "resolved_digest": "sha256:" + "a" * 64,
            "interface_mode": "synth_actor_runtime",
            "actor_role": "worker",
            "selection_kind": "customer_actor_runtime",
            "capabilities": ["codex_cli"],
            "python_packages": [],
            "package_release_timestamps": {},
            "recipe_digest": None,
            "image_release_id": "imgrel_" + "b" * 64,
            "image_substrates": ["org_registry"],
            "daytona_pullable": True,
            "security_admission": admission.to_wire(),
        }
    )
    assert image.security_admission == admission
    assert image.to_wire()["security_admission"]["state"] == "approved"


def test_image_security_mutation_contract_and_operations() -> None:
    admission = _security_mutation(
        {
            "schema_version": "smr-image-security-mutation-v1",
            "security_admission": {
                "admission_id": "11111111-1111-1111-1111-111111111111",
                "state": "check_failed",
                "manifest_digest": "sha256:" + "a" * 64,
                "policy_version": "v1",
                "generation": 1,
                "expires_at": None,
                "revocation_reason": "scanner timeout",
            },
        }
    )
    assert admission.state is SecurityAdmissionState.CHECK_FAILED
    assert _security_body(reason="retry scan", idempotency_key="retry-001") == {
        "reason": "retry scan",
        "idempotency_key": "retry-001",
    }
    assert research_operation("retry_customer_actor_image_security").idempotent
    assert research_operation("revoke_customer_actor_image_security").idempotent
    assert research_operation("reevaluate_customer_actor_image_security").idempotent
