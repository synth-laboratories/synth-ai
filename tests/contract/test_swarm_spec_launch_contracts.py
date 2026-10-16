"""SwarmSpec launch wire matches the backend SmrProjectTriggerRequest contract.

Finding F1 (L5 final wave, 2026-10-06): SwarmSpec omitted resource_bindings,
deployment_pins and provenance_mode, so swarms.preflight/create were refused
422 resource_inventory_unspecified by backend a9a681df.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import jsonschema
import pytest
from synth_ai.sdk.research.public import (
    DeploymentPin,
    ProvenanceMode,
    RunResourceBindings,
    SwarmSpec,
)

FIXTURES = Path(__file__).with_name("fixtures")
SPEC = json.loads((FIXTURES / "research_openapi.generated.json").read_text())
LAUNCH_PATHS = (
    "/smr/projects/{project_id}/launch-preflight",
    "/smr/projects/{project_id}/trigger",
    "/smr/runs:one-off/launch-preflight",
    "/smr/runs:one-off",
)

PIN = DeploymentPin(
    repository="synth-laboratories/backend",
    commit_sha="a9a681df" + "0" * 32,
    environment="staging",
    resolved_at=datetime(2026, 10, 6, 20, 30, tzinfo=UTC),
    deployment_id="railway-deploy-1",
)


def _validator(path: str) -> jsonschema.Draft202012Validator:
    body = SPEC["paths"][path]["post"]["requestBody"]["content"]["application/json"]["schema"]
    schema = {**body, "components": SPEC["components"]}
    return jsonschema.Draft202012Validator(schema)


@pytest.mark.parametrize("path", LAUNCH_PATHS)
def test_swarm_spec_wire_validates_against_launch_request__f1(path):
    spec = SwarmSpec(
        objective="Summarize the repository.",
        deployment_pins=(PIN,),
        provenance_mode=ProvenanceMode.LIVE,
    )
    errors = sorted(_validator(path).iter_errors(spec.to_wire()), key=str)
    assert not errors, [error.message for error in errors]


def test_default_resource_bindings_are_explicit_empty_lists__f1():
    wire = SwarmSpec(objective="o").to_wire()
    assert wire["resource_bindings"] == {
        "model_file_ids": [],
        "external_repository_ids": [],
        "credential_ref_ids": [],
    }
    required = SPEC["components"]["schemas"]["SmrRunResourceBindingsRequest"]["required"]
    assert set(required) <= set(wire["resource_bindings"])


def test_resource_bindings_carry_selected_inventories__f1():
    spec = SwarmSpec(
        objective="o",
        resource_bindings=RunResourceBindings(model_file_ids=["mf_1"], credential_ref_ids=["cr_1"]),
    )
    assert spec.to_wire()["resource_bindings"] == {
        "model_file_ids": ["mf_1"],
        "external_repository_ids": [],
        "credential_ref_ids": ["cr_1"],
    }


def test_launch_request_requires_every_field_swarm_spec_can_send__f1():
    required = set(SPEC["components"]["schemas"]["SmrProjectTriggerRequest"]["required"])
    wire = SwarmSpec(
        objective="o", deployment_pins=(PIN,), provenance_mode=ProvenanceMode.DRY_RUN
    ).to_wire()
    assert required <= set(wire)
    assert wire["provenance_mode"] == "dry_run"
    assert wire["deployment_pins"] == [
        {
            "repository": "synth-laboratories/backend",
            "commit_sha": PIN.commit_sha,
            "deployment_id": "railway-deploy-1",
            "artifact_digest": None,
            "environment": "staging",
            "resolved_at": "2026-10-06T20:30:00+00:00",
        }
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"deployment_pins": (PIN,)},
        {"provenance_mode": ProvenanceMode.LIVE},
        {"deployment_pins": ({"repository": "r"},), "provenance_mode": ProvenanceMode.LIVE},
        {"resource_bindings": {"model_file_ids": []}},
    ],
)
def test_invalid_provenance_and_bindings_are_unconstructible__f1(kwargs):
    with pytest.raises(ValueError):
        SwarmSpec(objective="o", **kwargs)


def test_provenance_mode_accepts_wire_text__f1():
    spec = SwarmSpec(objective="o", deployment_pins=(PIN,), provenance_mode="live")
    assert spec.provenance_mode is ProvenanceMode.LIVE


def test_deployment_pin_requires_aware_timestamp__f1():
    with pytest.raises(ValueError):
        DeploymentPin(
            repository="r",
            commit_sha="c",
            environment="e",
            resolved_at=datetime(2026, 10, 6),
        )
