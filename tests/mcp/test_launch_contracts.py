"""Launch provenance reaches the SDK adapter; no backend or provider calls.

See backend launch_resource_inventory.md and packages/smr/run_provenance.py.
"""

import jsonschema
import pytest
from synth_ai.mcp.research.server import ResearchMcpServer
from synth_ai.sdk.research.contracts.types import RunResourceBindings

PINS = [
    {
        "repository": "https://github.com/synth-laboratories/backend",
        "commit_sha": "a" * 40,
        "deployment_id": "offline",
        "artifact_digest": "sha256:" + "b" * 64,
        "environment": "local",
        "resolved_at": "2026-10-06T00:00:00Z",
    }
]


@pytest.mark.parametrize(
    "name",
    [
        "research_trigger_run",
        "research_start_run",
        "research_start_one_off_run",
        "research_start_run_in_dev_environment",
    ],
)
def test_schema_valid_launch_forwards_all_required_inputs(monkeypatch, name):
    server = ResearchMcpServer(
        api_key="offline-dummy", backend_base="http://offline.invalid", include_advanced_tools=True
    )
    sent = []

    class Client:
        def __init__(self):
            self.runs = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def launch(self, *args, **kwargs):
            sent.append(kwargs)
            return {"run_id": "r"}

        trigger_run = start_run = trigger_one_off_run = start_run_in_dev_environment = launch

    monkeypatch.setattr(server, "_client_from_args", lambda args: Client())
    arguments = {
        "project_id": "project",
        "objective": "offline",
        "intended_horizon_hours": 1,
        "work_mode": "general",
        "providers": [{"provider": "openai"}],
        "provenance_mode": "dry_run",
        "deployment_pins": PINS,
        "resource_bindings": RunResourceBindings().to_wire(),
    }
    if name == "research_start_one_off_run":
        arguments.pop("project_id")
    if "dev_environment" in name:
        arguments["dev_environment_id"] = "environment"
    else:
        arguments["host_kind"] = "docker"
    definition = server.get_tool_definition(name)
    jsonschema.validate(arguments, definition.input_schema)
    assert server.call_tool(name, arguments) == {"run_id": "r"}
    assert len(sent) == 1
    assert sent[0]["deployment_pins"] == PINS
    assert sent[0]["provenance_mode"] == "dry_run"
    assert sent[0]["resource_bindings"] == RunResourceBindings().to_wire()


def test_bindings_round_trip_model_files_and_empty_inventories():
    wire = {
        "model_file_ids": ["stored-file"],
        "external_repository_ids": [],
        "credential_ref_ids": [],
    }
    assert RunResourceBindings.from_wire(wire).to_wire() == wire


def test_typed_preflight_keeps_holds_and_selected_missing_files():
    from synth_ai.sdk.research.contracts.types import SmrLaunchPreflight

    payload = {
        "project_id": "project",
        "clear_to_trigger": False,
        "resource_readiness": {
            "ready": False,
            "selected_model_file_ids": ["selected"],
            "missing_model_file_ids": ["missing"],
        },
        "hold_requirements": {
            "roles": [
                {"role": role, "hold_micros": 250000}
                for role in ("orchestrator", "reviewer", "worker")
            ],
            "minimum_run_ceiling_micros": 750000,
        },
        "checks": [{"name": "budget", "status": "blocked"}],
    }
    response = SmrLaunchPreflight.from_wire(payload)
    assert response.hold_requirements.minimum_run_ceiling_micros == 750000
    assert response.resource_readiness.selected_model_file_ids == ("selected",)
    assert response.resource_readiness.missing_model_file_ids == ("missing",)
    assert response.checks[0]["name"] == "budget"


@pytest.mark.parametrize(
    "field,code",
    [("deployment_pins", "run_deployment_pins_missing"), ("provenance_mode", "provenance_unbound")],
)
def test_required_provenance_shape_error_keeps_typed_code(field, code):
    import httpx
    from synth_ai.sdk.research.errors import ResearchLaunchRefusalError
    from synth_ai.sdk.research.transport.http import _raise_for_error_response

    response = httpx.Response(
        422,
        request=httpx.Request("POST", "http://offline.invalid/smr/projects/p/trigger"),
        json={"detail": [{"loc": ["body", field], "type": "missing", "msg": "Field required"}]},
    )
    with pytest.raises(ResearchLaunchRefusalError) as refusal:
        _raise_for_error_response(response, operation_id="launch-intent")
    assert refusal.value.error_code == code
    assert refusal.value.retryable is False
    assert refusal.value.operation == "launch-intent"
