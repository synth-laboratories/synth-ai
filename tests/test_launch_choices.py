"""Contract 1.26.0: per-role launch choices, role assignments and allocation readback."""

import copy
import json

import httpx
import pytest
from pydantic import ValidationError

from synth_ai import AsyncSynthClient
from synth_ai._cli_view import allocation, launch_choices
from synth_ai._generated.models import LaunchChoices, RoleAssignment, RunAllocation
from synth_ai.cli import parse_role_assignments
from synth_ai.mcp.tools import TOOLS

REVISION = "lcm1_" + "a" * 64
OPTION = "sbo1_0123456789abcdef_" + "b" * 64


def example(contract, path, method="get", status="200"):
    return copy.deepcopy(
        contract["paths"][path][method]["responses"][status]["content"]["application/json"][
            "example"
        ]
    )


async def call(handle, action):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as api:
            return await action(api)


async def test_launch_choices_sends_the_prospective_launch_and_reads_the_menu(contract):
    seen = []

    def handle(request):
        seen.append(request.url)
        return httpx.Response(200, json=example(contract, "/v1/launch-choices"))

    choices = await call(
        handle,
        lambda api: api.swarms.launch_choices(
            parallel_workers=2, wall_time_seconds=600, budget="2.50", domicile_requirement="us_only"
        ),
    )
    assert seen[0].path == "/v1/launch-choices"
    assert dict(seen[0].params) == {
        "domicile_requirement": "us_only",
        "parallel_workers": "2",
        "wall_time_seconds": "600",
        "maximum_charge": "2.50",
    }
    assert choices.menu_revision.startswith("lcm1_")
    assert {r.actor_id for r in choices.roles} >= {"planner", "worker"}
    rendered = launch_choices(choices)
    assert choices.menu_revision in rendered and "planner decides" in rendered


def test_launch_choices_stay_strict(contract):
    raw = example(contract, "/v1/launch-choices")
    raw["inference_profiles"][0]["provider"] = "openai"
    with pytest.raises(ValidationError):
        LaunchChoices.model_validate(raw)
    raw = example(contract, "/v1/launch-choices")
    raw["supplier"] = "x"
    with pytest.raises(ValidationError):
        LaunchChoices.model_validate(raw)


async def test_launch_sends_role_assignments_with_the_menu_revision(contract):
    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(202, json=example(contract, "/v1/runs/{run_id}"))

    assignments = {
        "worker": RoleAssignment(
            inference_profile="luna-high", sandbox_profile="8vcpu-16gib-us", sandbox_option=OPTION
        ),
        "worker_2": RoleAssignment(planner_decides=["sandbox"]),
    }
    await call(
        handle,
        lambda api: api.swarms.launch(
            "task",
            budget="5",
            parallel_workers=2,
            role_assignments=assignments,
            launch_menu_revision=REVISION,
        ),
    )
    assert bodies[0]["launch_menu_revision"] == REVISION
    assert bodies[0]["role_assignments"] == {
        "worker": {
            "inference_profile": "luna-high",
            "sandbox_profile": "8vcpu-16gib-us",
            "sandbox_option": OPTION,
        },
        "worker_2": {"planner_decides": ["sandbox"]},
    }


async def test_role_assignments_without_a_revision_never_reach_the_server():
    def handle(_):
        pytest.fail("request sent")

    with pytest.raises(ValueError, match="launch_menu_revision"):
        await call(
            handle,
            lambda api: api.swarms.launch(
                "task",
                budget="5",
                role_assignments={"worker": RoleAssignment(inference_profile="luna-low")},
            ),
        )


@pytest.mark.parametrize(
    "raw",
    [
        {"provider": "modal"},
        {"sandbox_option": "opt_1"},
        {"planner_decides": []},
        {"planner_decides": ["region"]},
        {"inference_profile": "../x"},
    ],
)
def test_role_assignment_rejects_what_the_contract_rejects(raw):
    with pytest.raises(ValidationError):
        RoleAssignment.model_validate(raw)


def test_cli_parses_assignments_and_planner_decides():
    parsed = parse_role_assignments(
        ["worker.inference=luna-high", "worker.sandbox=2vcpu-2gib-us", "worker.option=" + OPTION],
        ["worker_2.sandbox", "worker_2.inference"],
    )
    assert parsed == {
        "worker": RoleAssignment(
            inference_profile="luna-high", sandbox_profile="2vcpu-2gib-us", sandbox_option=OPTION
        ),
        "worker_2": RoleAssignment(planner_decides=["sandbox", "inference"]),
    }
    assert parse_role_assignments([], []) is None
    for assign, decides in (
        (["worker=luna"], []),
        (["worker.provider=modal"], []),
        (["worker.inference=a", "worker.inference=b"], []),
        ([], ["worker.option"]),
        ([], ["worker.sandbox", "worker.sandbox"]),
    ):
        with pytest.raises(ValueError):
            parse_role_assignments(assign, decides)


async def test_allocation_reads_per_actor_placement_and_charge(contract):
    raw = example(contract, "/v1/runs/{run_id}/allocation")

    def handle(request):
        assert request.url.path.endswith("/allocation")
        return httpx.Response(200, json=raw)

    read = await call(handle, lambda api: api.swarms.allocation(raw["run_id"]))
    assert isinstance(read, RunAllocation) and read.run_id == raw["run_id"]
    assert read.actors and read.actors[0].launch.inference.source.value in {"explicit", "default"}
    assert raw["run_id"] in allocation(read)
    raw["actors"][0]["provider"] = "modal"
    with pytest.raises(ValidationError):
        RunAllocation.model_validate(raw)


def test_mcp_exposes_choices_allocation_and_launch_fields():
    names = {tool["name"]: tool for tool in TOOLS}
    assert names["swarms_launch_choices"]["operationId"] == "getLaunchChoices"
    assert names["swarms_run_allocation"]["operationId"] == "getRunAllocation"
    launch = next(t for t in TOOLS if t["operationId"] == "createRun")
    text = json.dumps(launch)
    assert "role_assignments" in text and "launch_menu_revision" in text
