"""The retained Swarms identity admits strict launch on the reviewed backend."""

import json
from pathlib import Path

import httpx
import pytest
from synth_ai import SynthClient
from synth_ai.contracts.fingerprint import (
    digest_exported_contract,
    export_contract_json,
    schema_sha256,
)
from synth_ai.options import ClientOptions, MutationOptions
from synth_ai.sdk.swarms.types.resources import ResourceRequest
from synth_ai.types.capabilities import CapabilitySnapshot, snapshot_sha256
from tests.artifact_fixtures import snapshot_body
from tests.http_fixtures import capabilities_body
from tests.test_resource_mnemosyne_trace import PROJECT, declaration
from tests.test_swarms_rate_limits import LIMITS

pytestmark = pytest.mark.unit
BACKEND_RETAINED_SHA = "a1a1c821c60a1c939e79c52bf017d6246b960f1f2d84feab84508fde393d2a43"


def test_observation_operations_are_generated_and_match_retained_reviewed_descriptors():
    root = Path(__file__).parents[1]
    retained = json.loads((root / "contracts/swarms-v1.json").read_text())
    live = json.loads(export_contract_json("swarms.v1"))
    assert retained["operations"] == live["operations"]
    operations = {op["id"]: op for op in retained["operations"]}
    for operation, method, path, mutation in (
        (
            "swarms.observations.recover",
            "POST",
            "/api/v1/swarms/runs/{run_id}/observation-recovery",
            True,
        ),
        (
            "swarms.observations.preview",
            "GET",
            "/api/v1/swarms/runs/{run_id}/observation-recovery/{sequence}",
            False,
        ),
    ):
        assert operations[operation] == {
            "id": operation,
            "method": method,
            "path": path,
            "keyed": True,
            "mutation": mutation,
            "required_scope": "swarms:control",
            "may_charge": False,
            "idempotent": True,
        }
    assert (
        digest_exported_contract(json.dumps(retained))
        == schema_sha256("swarms.v1")
        == BACKEND_RETAINED_SHA
    )


def test_strict_launch_negotiates_reviewed_backend_identity():
    receipt = capabilities_body()
    receipt["contracts"] = [{"contract": "swarms.v1", "schema_sha256": BACKEND_RETAINED_SHA}]
    receipt["operations"] = [
        {
            "operation_id": "swarms.create",
            "contract": "swarms.v1",
            "state": "enabled",
            "surfaces": ["http"],
            "reason": None,
        }
    ]
    snapshot = {k: v for k, v in receipt.items() if k not in {"metadata", "capability_sha256"}}
    receipt["capability_sha256"] = snapshot_sha256(
        CapabilitySnapshot.model_validate_json(json.dumps(snapshot))
    )
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path.endswith("/capabilities"):
            return httpx.Response(200, headers={"X-Synth-Contract": "index.v1"}, json=receipt)
        return httpx.Response(
            200,
            headers={
                "X-Synth-Contract": "swarms.v1",
                "X-Synth-Schema-SHA256": BACKEND_RETAINED_SHA,
                "X-Synth-Capabilities-SHA256": receipt["capability_sha256"],
                "X-Synth-Release-Manifest-SHA256": receipt["release"]["manifest_sha256"],
            },
            json={**snapshot_body(state="queued", result_available=False), "project_id": PROJECT},
        )

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as network,
        SynthClient(
            api_key="fixture",
            base_url="https://backend.test",
            options=ClientOptions(http_client=network, strict_contract=True),
        ) as client,
    ):
        run = client.swarms.create(
            message="m",
            project_id=PROJECT,
            limits=LIMITS,
            resources=(ResourceRequest.model_validate_json(json.dumps(declaration())),),
            mutation=MutationOptions(idempotency_key="strict-launch"),
        )
    assert run.run_id == "run-1"
    assert [r.url.path for r in requests] == ["/api/v1/capabilities", "/api/v1/swarms/runs"]
    assert requests[-1].headers["X-Synth-Schema-SHA256"] == BACKEND_RETAINED_SHA
    assert json.loads(requests[-1].content)["resources"][0]["plugin"] == "mnemosyne.trace"
