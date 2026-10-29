"""SYN-4848: launch-wide and per-role model/effort overrides on launch, offline."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError
from synth_ai import SynthClient
from synth_ai.errors import ContractError, OutcomeUncertainError
from synth_ai.mcp.catalog import create_run, preflight
from synth_ai.mcp.contracts import RunCreate
from synth_ai.sdk.swarms import (
    Limits,
    ModelEffortChoice,
    ProfileOverrideReadback,
    ProfileOverrides,
)
from synth_ai.sdk.swarms.client import launch_request, validate_preflight
from synth_ai.sdk.swarms.types.launch import LaunchRequest
from synth_ai.sdk.swarms.types.runs import LaunchPreflight, RunSnapshot
from synth_ai.types.money import Money
from tests.http_fixtures import METADATA

pytestmark = pytest.mark.unit
LIMITS = Limits(
    maximum_charge=Money(coefficient=1, scale=0, currency="USD"),
    wall_time_seconds=60,
    maximum_tasks=1,
    maximum_parallel_tasks=1,
)
OVERRIDES = ProfileOverrides(
    launch=ModelEffortChoice(effort="high"),
    roles={"worker": ModelEffortChoice(canonical_model="gpt-6-luna", effort="low")},
)
BASE = "sha256:" + "a" * 64
DERIVED = "sha256:" + "b" * 64


def _resolution() -> dict:
    return {
        "schema_version": "synth.profile-override-resolution.v1",
        "declaration_digest": "sha256:" + "c" * 64,
        "request_digest": "sha256:" + "d" * 64,
        "actors": [
            {
                "actor_id": "planner-1",
                "actor_type": "planner",
                "base_profile_digest": BASE,
                "profile_digest": DERIVED,
                "canonical_model": "gpt-6-luna",
                "effort": "high",
                "model_source": "template_default",
                "effort_source": "launch",
                "evaluated": False,
            }
        ],
        "evaluated_system_changed": False,
        "digest": "sha256:" + "e" * 64,
    }


def _readback(requested: ProfileOverrides, *, resolved: bool = True) -> dict:
    value = {"requested": requested.model_dump(mode="json")}
    if resolved:
        value["resolution"] = _resolution()
    return value


def _preflight(readback: dict | None, *, ready: bool = True) -> LaunchPreflight:
    configuration = {
        "metadata": METADATA,
        "run_id": None,
        "project_id": "p",
        "configuration_id": "cfg",
        "digest_sha256": "d" * 64,
        "profile_id": "prof",
        "runtime_image_digest": "e" * 64,
        "limits": LIMITS.model_dump(mode="json"),
        "inputs": {"files": []},
        "providers": {"default": "openai"},
        "models": {"default": "gpt-6-luna"},
    }
    if readback is not None:
        configuration["profile_overrides"] = readback
    return LaunchPreflight.model_validate_json(
        json.dumps(
            {
                "metadata": METADATA,
                "ready": ready,
                "blockers": [] if ready else ["profile_override_unsupported"],
                "configuration": configuration,
                "expires_at": "2026-10-20T00:00:00Z" if ready else None,
            }
        )
    )


def test_absent_overrides_keep_the_launch_bytes():
    request = launch_request(None, "p", LIMITS, None, None, None, message="m")
    assert (
        request.model_dump_json()
        == LaunchRequest(message="m", project_id="p", limits=LIMITS).model_dump_json()
    )
    assert "profile_overrides" not in json.loads(request.model_dump_json())


def test_overrides_reach_the_wire_without_absent_fields_and_round_trip():
    request = launch_request(
        None, "p", LIMITS, None, None, None, message="m", profile_overrides=OVERRIDES
    )
    body = request.model_dump_json()
    assert json.loads(body)["profile_overrides"] == {
        "launch": {"effort": "high"},
        "roles": {"worker": {"canonical_model": "gpt-6-luna", "effort": "low"}},
    }
    assert LaunchRequest.model_validate_json(body).model_dump_json() == body
    role_only = ProfileOverrides(roles={"planner": ModelEffortChoice(canonical_model="m-1")})
    assert role_only.model_dump(mode="json") == {"roles": {"planner": {"canonical_model": "m-1"}}}


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"launch": {}},
        {"launch": {"effort": "maximum"}},
        {"roles": {"bad role": {"effort": "low"}}},
        {"roles": {"worker": {"canonical_model": "", "effort": "low"}}},
        {"launch": {"effort": "low"}, "fallback": True},
    ],
)
def test_overrides_are_exact(value):
    with pytest.raises(ValidationError):
        ProfileOverrides.model_validate_json(json.dumps(value))


def test_preflight_must_echo_the_exact_frozen_request_with_a_resolution():
    request = launch_request(
        None, "p", LIMITS, None, None, None, message="m", profile_overrides=OVERRIDES
    )
    result = validate_preflight(_preflight(_readback(OVERRIDES)), request)
    readback = result.configuration.profile_overrides
    assert readback.resolution.actors[0].effort_source == "launch"
    assert readback.resolution.actors[0].profile_digest == DERIVED
    with pytest.raises(ContractError):
        validate_preflight(_preflight(None), request)
    with pytest.raises(ContractError):
        validate_preflight(_preflight(_readback(OVERRIDES, resolved=False)), request)
    other = ProfileOverrides(launch=ModelEffortChoice(effort="low"))
    with pytest.raises(ContractError):
        validate_preflight(_preflight(_readback(other)), request)


def test_blocked_preflight_need_not_carry_a_resolution():
    request = launch_request(
        None, "p", LIMITS, None, None, None, message="m", profile_overrides=OVERRIDES
    )
    blocked = validate_preflight(_preflight(None, ready=False), request)
    assert blocked.blockers == ("profile_override_unsupported",)


def test_preflight_without_overrides_refuses_an_unrequested_echo():
    request = launch_request(None, "p", LIMITS, None, None, None, message="m")
    assert validate_preflight(_preflight(None), request).configuration.profile_overrides is None
    with pytest.raises(ContractError):
        validate_preflight(_preflight(_readback(OVERRIDES)), request)


def test_configuration_without_overrides_serializes_without_the_key():
    configuration = _preflight(None).configuration
    assert "profile_overrides" not in configuration.model_dump(mode="json")


SWARMS_METADATA = {
    "contract": "swarms.v1",
    "request_id": "overrides-fixture",
    "operation_id": None,
    "observed_at": "2026-10-29T12:00:00Z",
}


def _snapshot(readback: dict | None) -> dict:
    value = {
        "metadata": SWARMS_METADATA,
        "run_id": "run-1",
        "project_id": "p",
        "state": "queued",
        "model": "gpt-6-luna",
        "provider": "openai",
        "result_available": False,
        "execution_closed": False,
        "accounting_final": False,
        "cleanup_complete": False,
    }
    if readback is not None:
        value["profile_overrides"] = readback
    return value


def sync_client(respond):
    client = SynthClient(api_key="fixture-key", base_url="https://backend.test")
    client._transport.client.close()
    client._transport.client = httpx.Client(transport=httpx.MockTransport(respond))
    client._transport._negotiate = lambda *args: None
    return client


def test_create_sends_the_overrides_and_requires_the_frozen_readback():
    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"X-Synth-Contract": "swarms.v1"},
            json=_snapshot(_readback(OVERRIDES, resolved=False)),
        )

    with sync_client(respond) as client:
        run = client.swarms.create(
            message="m", project_id="p", limits=LIMITS, profile_overrides=OVERRIDES
        )
    assert sent[0]["profile_overrides"] == OVERRIDES.model_dump(mode="json")
    assert run.run_id == "run-1"

    def ignored(request):
        return httpx.Response(200, headers={"X-Synth-Contract": "swarms.v1"}, json=_snapshot(None))

    # An older server that ignores the field is a mismatch, never a silent default.
    with sync_client(ignored) as client, pytest.raises(OutcomeUncertainError):
        client.swarms.create(
            message="m", project_id="p", limits=LIMITS, profile_overrides=OVERRIDES
        )


def test_snapshot_reads_back_the_resolution_and_omits_an_absent_readback():
    snapshot = RunSnapshot.model_validate_json(json.dumps(_snapshot(_readback(OVERRIDES))))
    assert snapshot.profile_overrides.resolution.digest == "sha256:" + "e" * 64
    assert isinstance(snapshot.profile_overrides, ProfileOverrideReadback)
    bare = RunSnapshot.model_validate_json(json.dumps(_snapshot(None)))
    assert "profile_overrides" not in bare.model_dump(mode="json")


def test_mcp_forwards_the_overrides_on_preflight_and_create():
    body = {
        "message": "Tune the coordinator",
        "project_id": "p",
        "limits": LIMITS.model_dump(mode="json"),
        "profile_overrides": OVERRIDES.model_dump(mode="json"),
    }
    swarm = SimpleNamespace(
        preflight=AsyncMock(return_value=_preflight(_readback(OVERRIDES))),
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
        assert mock.await_args.kwargs["profile_overrides"] == OVERRIDES
