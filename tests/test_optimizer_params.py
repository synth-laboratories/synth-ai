"""SYN-5667: the two public swarms_v2 controls and one experimental treatment (contract 1.36.0)."""

import json

import httpx
import pytest
from pydantic import JsonValue, ValidationError
from test_cli import Api, example, invoke

from synth_ai import AsyncSynthClient, OptimizerParams, OptimizerPin, SynthClient, topology_name
from synth_ai._generated.models import FamilyList, LaunchRequest, OptimizerPinTopology
from synth_ai.cli import EXIT_CLIENT_ERROR, EXIT_OK, parse_optimizer
from synth_ai.mcp.server import call_tool
from synth_ai.mcp.tools import TOOLS

PARAMS = {"max_tasks": 8, "max_repairs_per_task": 3}


@pytest.fixture
def api(contract, monkeypatch, tmp_path):
    monkeypatch.setenv("SYNTH_API_KEY", "fixture")
    monkeypatch.setenv("SYNTH_API_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("SYNTH_AUTH_FILE", str(tmp_path / "credentials.json"))
    monkeypatch.setenv("SYNTH_STATE_DIR", str(tmp_path / "state"))
    return Api(contract)


def test_pin_carries_both_controls_and_round_trips():
    pin = OptimizerPin(
        topology=OptimizerPinTopology.SWARMS_V2, version=2, params=OptimizerParams(**PARAMS)
    )
    request = {
        "message": "task",
        "limits": {"maximum_charge": {"amount": "1.00", "currency": "USD"}},
        "optimizer": pin.model_dump(mode="json", exclude_unset=True),
    }
    model = LaunchRequest.model_validate(request)
    assert model.model_dump(mode="json", exclude_unset=True)["optimizer"]["params"] == PARAMS


@pytest.mark.parametrize(
    "params",
    [
        {"max_tasks": 0},
        {"max_tasks": 4097},
        {"max_repairs_per_task": -1},
        {"max_repairs_per_task": 33},
        {"max_tasks": "8"},
        {"max_tasks": 8.0},
        {"width": 3},
    ],
)
def test_out_of_range_noninteger_and_unknown_params_refuse(params):
    with pytest.raises(ValidationError):
        OptimizerPin.model_validate({"topology": "swarms_v2", "version": 2, "params": params})


def test_omitted_params_leave_the_launch_body_unchanged():
    with SynthClient("fixture", base_url="https://example.test") as client:
        request = client.swarms.prepare_launch(
            "task", budget="1.00", optimizer=parse_optimizer("swarms_v2@2")
        )
    assert request.model_dump(mode="json", exclude_unset=True)["optimizer"] == {
        "topology": "swarms_v2",
        "version": 2,
    }


async def test_sdk_sends_params_and_reads_the_echo(contract):
    sent = []
    _, snapshot = example(contract, "getRun")
    snapshot["optimizer"] = {
        "topology": "swarms_v2",
        "version": 2,
        "params": PARAMS,
        "reason": "explicit_pin",
        "definition_digest": "sha256:" + "a" * 64,
    }

    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(202, json=snapshot)

    pin = OptimizerPin(
        topology=OptimizerPinTopology.SWARMS_V2, version=2, params=OptimizerParams(**PARAMS)
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            run = await client.swarms.launch("task", budget="1.00", optimizer=pin)
    assert sent[0]["optimizer"]["params"] == PARAMS
    assert run.optimizer is not None
    assert run.optimizer.params.max_tasks == 8
    assert run.optimizer.params.max_repairs_per_task == 3


def test_cli_flags_send_both_controls_on_a_swarms_pin(api, capsys):
    argv = ["run", "Goal", "--budget", "1.00", "--optimizer", "swarms_v2@2"]
    argv += ["--max-tasks", "8", "--max-repairs-per-task", "3"]
    code, _, err = invoke(api, capsys, *argv)
    assert code == EXIT_OK, err
    launch = json.loads(next(r for oid, r in api.calls if oid == "createRun").content)
    assert launch["optimizer"] == {"topology": "swarms_v2", "version": 2, "params": PARAMS}


@pytest.mark.parametrize(
    "optimizer", [[], ["--optimizer", "auto"], ["--optimizer", "fork_flush@1"]]
)
def test_cli_flags_require_an_explicit_swarms_pin(api, capsys, optimizer):
    argv = ["run", "Goal", "--budget", "1.00", *optimizer, "--max-tasks", "8"]
    code, _, err = invoke(api, capsys, *argv)
    assert code == EXIT_CLIENT_ERROR
    assert "swarms_v2@VERSION" in err
    assert not any(oid == "createRun" for oid, _ in api.calls)


async def test_mcp_launch_schema_and_call_carry_both_controls(contract):
    launch = next(tool for tool in TOOLS if tool["name"] == "swarms_run_launch")
    body_schema = launch["inputSchema"]["properties"]["body"]
    assert isinstance(body_schema, dict)
    body_props = body_schema["properties"]
    assert isinstance(body_props, dict)
    optimizer_schema = body_props["optimizer"]
    assert isinstance(optimizer_schema, dict)
    one_of = optimizer_schema["oneOf"]
    assert isinstance(one_of, list)
    pin = one_of[1]
    assert isinstance(pin, dict)
    pin_props = pin["properties"]
    assert isinstance(pin_props, dict)
    params_schema = pin_props["params"]
    assert isinstance(params_schema, dict)
    params_props = params_schema["properties"]
    assert isinstance(params_props, dict)
    assert set(params_props) == set(PARAMS) | {"branches", "rounds", "exploit_bps"}
    sent = []
    _, snapshot = example(contract, "getRun")

    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(202, json=snapshot)

    body: dict[str, JsonValue] = {
        "message": "task",
        "optimizer": {"topology": "swarms_v2", "version": 2, "params": dict(PARAMS)},
        "limits": {"maximum_charge": {"amount": "1.00", "currency": "USD"}},
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            await call_tool(
                client, "swarms_run_launch", {"body": body, "idempotency_key": "params-once"}
            )
    assert sent[0]["optimizer"]["params"] == PARAMS


def test_public_names_mark_experimental_and_never_default(contract):
    _, raw = example(contract, "listFamilies")
    raw["topologies"] = [
        {
            "topology": "swarms_v2",
            "version": 2,
            "label": "default",
            "auto": True,
            "availability": "served",
        },
        {
            "topology": "fork_flush",
            "version": 1,
            "label": "experimental",
            "auto": False,
            "availability": "unavailable",
        },
    ]
    topologies = FamilyList.model_validate(raw).topologies
    assert topologies is not None
    swarms, fork = topologies
    assert topology_name(swarms) == "Swarms v2 (default)"
    assert topology_name(fork) == "Fork & Flush (experimental, unavailable)"
    # A misconfigured catalog cannot promote an experimental topology to default.
    promoted = fork.model_copy(update={"auto": True, "availability": "served"})
    assert topology_name(promoted) == "Fork & Flush (experimental)"


def test_cli_families_prints_public_names(api, capsys, contract):
    _, raw = example(contract, "listFamilies")
    raw["topologies"] = [
        {
            "topology": "fork_flush",
            "version": 1,
            "label": "experimental",
            "auto": False,
            "availability": "unavailable",
        }
    ]
    raw["families"] = [
        {
            "id": "swarms.general",
            "version": "1",
            "summary": "General-purpose multi-agent swarm",
            "research_binding": "none",
            "required_actor_types": ["planner", "worker", "reviewer"],
        }
    ]
    api.overrides["listFamilies"] = (200, raw)
    code, out, err = invoke(api, capsys, "families")
    assert code == EXIT_OK, err
    assert not out.lstrip().startswith("{")
    assert "TOPOLOGY" in out
    assert "Fork & Flush (experimental, unavailable)" in out
    assert "FAMILY" in out
    assert "swarms.general" in out
    assert "General-purpose multi-agent swarm" in out
    code, out, err = invoke(api, capsys, "families", "--json")
    assert code == EXIT_OK, err
    assert len(out.strip().splitlines()) == 1
    parsed = json.loads(out)
    assert parsed["topologies"][0]["topology"] == "fork_flush"
    assert parsed["families"][0]["id"] == "swarms.general"


def test_cli_fork_flush_flags_and_guidance_reach_launch(api, capsys, tmp_path):
    guidance_doc = [
        {
            "id": "proof",
            "kind": "outcome",
            "text": "Include verifiable evidence.",
            "priority": "must",
        }
    ]
    guidance_path = tmp_path / "guidance.json"
    guidance_path.write_text(json.dumps(guidance_doc))
    argv = [
        "run",
        "Goal",
        "--budget",
        "2.00",
        "--optimizer",
        "fork_flush@1",
        "--branches",
        "6",
        "--rounds",
        "4",
        "--exploit-bps",
        "2500",
        "--guidance",
        str(guidance_path),
    ]
    code, _, err = invoke(api, capsys, *argv)
    assert code == EXIT_OK, err
    launch = json.loads(next(r for oid, r in api.calls if oid == "createRun").content)
    assert launch["optimizer"] == {
        "topology": "fork_flush",
        "version": 1,
        "params": {"branches": 6, "rounds": 4, "exploit_bps": 2500},
    }
    assert launch["guidance"] == guidance_doc


@pytest.mark.parametrize("optimizer", [[], ["--optimizer", "auto"], ["--optimizer", "swarms_v2@2"]])
def test_cli_fork_flush_flags_require_fork_flush_pin(api, capsys, optimizer):
    argv = ["run", "Goal", "--budget", "1.00", *optimizer, "--branches", "4"]
    code, _, err = invoke(api, capsys, *argv)
    assert code == EXIT_CLIENT_ERROR
    assert "fork_flush@VERSION" in err
    assert not any(oid == "createRun" for oid, _ in api.calls)


@pytest.mark.parametrize("document", ["{}", "not json", json.dumps([{"id": "x"}])])
def test_cli_refuses_bad_guidance_before_any_call(api, capsys, tmp_path, document):
    guidance_path = tmp_path / "guidance.json"
    guidance_path.write_text(document)
    upload = tmp_path / "notes.md"
    upload.write_bytes(b"# Notes\n")
    argv = [
        "run",
        "Goal",
        "--budget",
        "1.00",
        "--file",
        str(upload),
        "--guidance",
        str(guidance_path),
    ]
    code, out, err = invoke(api, capsys, *argv)
    assert code == EXIT_CLIENT_ERROR and out == "", err
    assert err.startswith("client_error: --guidance ")
    assert api.calls == []
