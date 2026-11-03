"""SYN-4849: exact program family selection on launch, and the family list, offline."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError
from synth_ai import SynthClient
from synth_ai.errors import ContractError
from synth_ai.mcp.catalog import BY_NAME, create_run, preflight
from synth_ai.mcp.contracts import RunCreate
from synth_ai.sdk.swarms import (
    Limits,
    ProgramChoice,
    ProgramFamilyRef,
    ResearchBinding,
)
from synth_ai.sdk.swarms.client import launch_request
from synth_ai.sdk.swarms.types.launch import LaunchRequest
from synth_ai.sdk.swarms.types.runs import LaunchPreflight
from synth_ai.types.money import Money

pytestmark = pytest.mark.unit
LIMITS = Limits(
    maximum_charge=Money(coefficient=1, scale=0, currency="USD"),
    wall_time_seconds=60,
    maximum_tasks=1,
    maximum_parallel_tasks=1,
)
FORGE_PROJECT = "63000000-0000-4000-8000-000000000001"
RESEARCH = ProgramChoice(
    family=ProgramFamilyRef(id="swarms.research", version=1),
    research=ResearchBinding(forge_project_id=FORGE_PROJECT, record_protocol="forge.records@1"),
)
METADATA = {
    "contract": "swarms.v1",
    "request_id": "families-fixture",
    "operation_id": None,
    "observed_at": "2026-10-29T12:00:00Z",
}
FAMILIES = {
    "metadata": METADATA,
    "families": [
        {
            "id": "swarms.general",
            "version": 1,
            "summary": "General Swarms",
            "research_binding": "forbidden",
            "required_actor_types": ["planner", "worker"],
        },
        {
            "id": "swarms.research",
            "version": 1,
            "summary": "Research Swarms",
            "research_binding": "required",
            "required_actor_types": ["planner", "worker"],
        },
    ],
}


def test_absent_program_is_omitted_from_the_wire():
    request = launch_request(None, "p", LIMITS, None, None, None, message="m")
    assert "program" not in json.loads(request.model_dump_json())


def test_general_family_wire_omits_the_absent_research_binding():
    choice = ProgramChoice(family=ProgramFamilyRef(id="swarms.general", version=1))
    request = launch_request(None, "p", LIMITS, None, None, None, message="m", program=choice)
    assert json.loads(request.model_dump_json())["program"] == {
        "family": {"id": "swarms.general", "version": 1}
    }


def test_research_family_wire_carries_the_exact_binding_and_round_trips():
    request = launch_request(None, "p", LIMITS, None, None, None, message="m", program=RESEARCH)
    body = request.model_dump_json()
    assert json.loads(body)["program"] == {
        "family": {"id": "swarms.research", "version": 1},
        "research": {"forge_project_id": FORGE_PROJECT, "record_protocol": "forge.records@1"},
    }
    assert LaunchRequest.model_validate_json(body).model_dump_json() == body


@pytest.mark.parametrize(
    "family",
    [
        {"id": "swarms.general"},
        {"id": "swarms.general", "version": 0},
        {"id": "swarms general", "version": 1},
        {"id": "swarms.general", "version": "1"},
        {"id": "swarms.general", "version": 1, "latest": True},
    ],
)
def test_family_selection_is_exact(family):
    with pytest.raises(ValidationError):
        ProgramChoice.model_validate_json(json.dumps({"family": family}))


def test_research_binding_needs_a_forge_project_uuid_and_protocol():
    with pytest.raises(ValidationError):
        ResearchBinding(forge_project_id="not-a-uuid", record_protocol="forge.records@1")
    with pytest.raises(ValidationError):
        ResearchBinding.model_validate({"forge_project_id": FORGE_PROJECT})


def sync_client(respond):
    client = SynthClient(api_key="fixture-key", base_url="https://backend.test")
    client._transport.client.close()
    client._transport.client = httpx.Client(transport=httpx.MockTransport(respond))
    client._transport._negotiate = lambda *args: None
    return client


def test_families_reads_the_typed_catalogue():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, headers={"X-Synth-Contract": "swarms.v1"}, json=FAMILIES)

    with sync_client(respond) as client:
        result = client.swarms.families()
    assert [(r.method, r.url.path) for r in requests] == [("GET", "/api/v1/swarms/families")]
    assert [(f.id, f.version, f.research_binding) for f in result.families] == [
        ("swarms.general", 1, "forbidden"),
        ("swarms.research", 1, "required"),
    ]


def test_families_refuses_a_duplicate_family_version():
    duplicate = {**FAMILIES, "families": [FAMILIES["families"][0]] * 2}

    def respond(request):
        return httpx.Response(200, headers={"X-Synth-Contract": "swarms.v1"}, json=duplicate)

    with sync_client(respond) as client, pytest.raises(ContractError):
        client.swarms.families()


def test_mcp_lists_families_and_forwards_the_program_choice():
    assert "swarms_families" in BY_NAME
    body = {
        "message": "Study the selected protocol",
        "project_id": "execution-project",
        "limits": LIMITS.model_dump(mode="json"),
        "program": RESEARCH.model_dump(mode="json"),
    }
    swarm = SimpleNamespace(
        preflight=AsyncMock(
            return_value=LaunchPreflight.model_validate_json(
                json.dumps(
                    {
                        "metadata": METADATA,
                        "ready": False,
                        "blockers": ["program_family_context_unbound"],
                        "configuration": None,
                        "expires_at": None,
                    }
                )
            )
        ),
        create=AsyncMock(return_value=SimpleNamespace(run_id="run-1", rate_limits=())),
    )
    client = SimpleNamespace(swarms=swarm)

    async def run():
        await preflight(client, LaunchRequest.model_validate_json(json.dumps(body)))
        await create_run(
            client, RunCreate.model_validate_json(json.dumps({**body, "idempotency_key": "k-1"}))
        )

    asyncio.run(run())
    for mock in (swarm.preflight, swarm.create):
        assert mock.await_args.kwargs["program"] == RESEARCH
