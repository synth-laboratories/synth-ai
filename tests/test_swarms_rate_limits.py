"""Swarms preflight/create surface the x-ratelimit-* success headers (SYN-4680)."""

from __future__ import annotations

import json

import httpx
import pytest
from synth_ai.cli_auth import parser
from synth_ai.cli_swarms import preflight as cli_preflight
from synth_ai.cli_swarms import read_launch_request, render_preflight
from synth_ai.mcp.catalog import create_run
from synth_ai.mcp.catalog import preflight as mcp_preflight
from synth_ai.mcp.contracts import LaunchPreflightResult, RunCreate
from synth_ai.options import MutationOptions, ReadOptions
from synth_ai.sdk.swarms.types.launch import LaunchRequest, Limits
from synth_ai.sdk.swarms.types.runs import LaunchPreflight
from synth_ai.types.money import Money
from synth_ai.types.rate_limits import RateLimitSnapshot, ResponseMetadata, describe_rate_limit
from tests.artifact_fixtures import drive, meta, snapshot_body

pytestmark = pytest.mark.unit

RESET = "2026-10-29T12:00:00+00:00"
CAPACITY_HEADERS = {
    "x-ratelimit-limit-concurrency": "5",
    "x-ratelimit-remaining-concurrency": "3",
    "x-ratelimit-burst-concurrency": "2",
    "x-ratelimit-reset-concurrency": RESET,
    "x-ratelimit-limit-requests": "60",
    "x-ratelimit-remaining-requests": "59",
    "x-ratelimit-reset-requests": RESET,
}
CONCURRENCY = RateLimitSnapshot("concurrency", 5, 3, RESET, 2)
REQUESTS = RateLimitSnapshot("requests", 60, 59, RESET, None)
EXPECTED = (REQUESTS, CONCURRENCY)
LIMITS = Limits(
    maximum_charge=Money(coefficient=5, scale=0, currency="USD"),
    wall_time_seconds=60,
    maximum_tasks=1,
    maximum_parallel_tasks=1,
)
REQUEST_BODY = {
    "message": "Evaluate the selected workspace",
    "project_id": "project-test",
    "limits": {
        "maximum_charge": {"coefficient": 5, "scale": 0, "currency": "USD"},
        "wall_time_seconds": 60,
        "maximum_tasks": 1,
        "maximum_parallel_tasks": 1,
    },
}
PREFLIGHT_BODY = {
    "metadata": meta("swarms.v1"),
    "ready": False,
    "blockers": ["project_budget_exhausted"],
    "configuration": None,
    "expires_at": None,
}


class Server:
    """Answers preflight/create with capacity headers unless told otherwise."""

    def __init__(self, *, headers: dict[str, str] | None = None) -> None:
        self.headers = CAPACITY_HEADERS if headers is None else headers
        self.paths: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        body = (
            PREFLIGHT_BODY
            if request.url.path.endswith("/swarms/preflight")
            else snapshot_body(state="queued", result_available=False)
        )
        return httpx.Response(
            200, headers={"X-Synth-Contract": "swarms.v1", **self.headers}, json=body
        )


@pytest.mark.parametrize("async_mode", [False, True])
def test_sdk_preflight_and_create_carry_typed_rate_limits(async_mode):
    server = Server()

    async def steps(client, wait):
        preview = await wait(
            client.swarms.preflight(message="m", project_id="project-test", limits=LIMITS)
        )
        run = await wait(
            client.swarms.create(
                message="m",
                project_id="project-test",
                limits=LIMITS,
                mutation=MutationOptions(idempotency_key="key-1"),
            )
        )
        return preview, run

    preview, run = drive(async_mode, server, steps)
    assert server.paths == ["/api/v1/swarms/preflight", "/api/v1/swarms/runs"]
    assert isinstance(preview, LaunchPreflight)
    assert preview.rate_limits == EXPECTED
    assert run.run_id == "run-1"
    assert run.rate_limits == EXPECTED
    # Header capacity never leaks into the wire body or the published schema.
    assert "rate_limits" not in preview.model_dump()
    assert "rate_limits" not in LaunchPreflight.model_json_schema()["properties"]


@pytest.mark.parametrize("async_mode", [False, True])
def test_absent_headers_give_empty_rate_limits(async_mode):
    async def steps(client, wait):
        preview = await wait(
            client.swarms.preflight(message="m", project_id="project-test", limits=LIMITS)
        )
        run = await wait(
            client.swarms.create(message="m", project_id="project-test", limits=LIMITS)
        )
        return preview, run

    preview, run = drive(async_mode, Server(headers={}), steps)
    assert preview.rate_limits == ()
    assert run.rate_limits == ()


def test_caller_supplied_response_metadata_is_filled_and_reused():
    caller = ResponseMetadata()

    async def steps(client, wait):
        return client.swarms.preflight(
            message="m",
            project_id="project-test",
            limits=LIMITS,
            request=ReadOptions(response_metadata=caller),
        )

    preview = drive(False, Server(), steps)
    assert caller.rate_limits == EXPECTED
    assert preview.rate_limits == EXPECTED


def test_describe_rate_limit_lines():
    assert describe_rate_limit(CONCURRENCY) == (
        f"rate limit concurrency: remaining 3/limit 5, resets {RESET}, burst 2"
    )
    assert describe_rate_limit(RateLimitSnapshot("requests", None, 7, None)) == (
        "rate limit requests: remaining 7/limit unknown, resets unknown"
    )


def test_cli_preflight_prints_remaining_per_dimension(tmp_path):
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(REQUEST_BODY))
    args = parser().parse_args(["swarms", "preflight", "--request", str(path)])
    assert (args.command, args.swarms_command, args.json) == ("swarms", "preflight", False)
    request = read_launch_request(args.request)

    async def steps(client, wait):
        return cli_preflight(client, request)

    preview = drive(False, Server(), steps)
    assert render_preflight(preview, as_json=False) == [
        "ready: no",
        'blocker: "project_budget_exhausted"',
        f"rate limit requests: remaining 59/limit 60, resets {RESET}",
        f"rate limit concurrency: remaining 3/limit 5, resets {RESET}, burst 2",
    ]
    machine = json.loads(render_preflight(preview, as_json=True)[0])
    assert machine["rate_limits"] == [
        {"dimension": "requests", "limit": 60, "remaining": 59, "reset_at": RESET, "burst": None},
        {"dimension": "concurrency", "limit": 5, "remaining": 3, "reset_at": RESET, "burst": 2},
    ]

    bare = drive(False, Server(headers={}), steps)
    assert render_preflight(bare, as_json=False)[-1] == "rate limit: not reported"


def test_cli_preflight_refuses_an_unreadable_request(tmp_path):
    with pytest.raises(ValueError, match="unreadable"):
        read_launch_request(str(tmp_path / "missing.json"))


def test_mcp_preflight_and_create_payloads_include_rate_limits():
    preflight_args = LaunchRequest.model_validate_json(json.dumps(REQUEST_BODY))
    create_args = RunCreate.model_validate_json(
        json.dumps({**REQUEST_BODY, "idempotency_key": "key-1"})
    )

    async def steps(client, wait):
        return (
            await mcp_preflight(client, preflight_args),
            await create_run(client, create_args),
        )

    preview, created = drive(True, Server(), steps)
    assert isinstance(preview, LaunchPreflightResult)
    assert set(LaunchPreflight.model_fields) < set(LaunchPreflightResult.model_fields)
    payload = preview.model_dump(mode="json")
    assert payload["blockers"] == ["project_budget_exhausted"] and payload["ready"] is False
    assert payload["rate_limits"][1] == {
        "dimension": "concurrency",
        "limit": 5,
        "remaining": 3,
        "reset_at": RESET,
        "burst": 2,
    }
    created_payload = created.model_dump(mode="json")
    assert created_payload["run_id"] == "run-1"
    assert [item["remaining"] for item in created_payload["rate_limits"]] == [59, 3]
