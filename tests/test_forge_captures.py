"""Synthetic owner replies exercise public capture wire identity, never byte custody."""
import asyncio
import hashlib
import json

import httpx
import pytest

from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.contracts.forge import CAPTURE_OPERATIONS
from synth_ai.mcp.catalog import BY_NAME, ALL_BY_NAME
from synth_ai.options import ClientOptions, MutationOptions
from synth_ai.sdk.forge.captures import intent_reference
from synth_ai.sdk.forge.contracts.capture import CaptureAdmissionRequest
from synth_ai.sdk.forge.contracts.contracts import contract_digest
from synth_ai.types.capabilities import CapabilitySnapshot, snapshot_sha256
from tests.test_forge_workspaces import receipt_and_headers

PROJECT = "72000000-0000-4000-8000-000000000002"


def ref(authority, kind, identity="fixture"):
    return dict(authority=authority, kind=kind, record_id=identity, revision="1", digest_sha256="a" * 64)


def fixture():
    request = CaptureAdmissionRequest.model_validate(dict(scope=dict(organization_id="org", project_id=PROJECT),
        trial=ref("forge", "trial"), profile=ref("forge", "capture_profile"), inputs=[],
        expected_objects=1, expected_output_bytes=10, expected_trace_bytes=0,
        retain_until="2026-11-04T00:00:00Z"))
    digest = contract_digest(request)
    admission = dict(capacity_guarantee="logical_budget", retention_status="planned",
        request_digest_sha256=digest, input_grants=[], reused_inputs=[],
        storage_reservation=ref("forge", "capture_budget_reservation"), admitted_storage_bytes=10,
        trace_reservation=None, admitted_trace_bytes=0, retention=ref("forge", "capture_retention_plan"),
        retain_until="2026-11-04T00:00:00Z")
    profile = dict(profile_id="fixture", revision="1", required_roles=["raw_output"],
                   maximum_objects=1, maximum_total_bytes=10)
    capability, headers = receipt_and_headers()
    capability["operations"] = [dict(operation_id=op.id, contract="core.v1", state="enabled",
                                        surfaces=["http", "mcp"], reason=None) for op in sorted(CAPTURE_OPERATIONS, key=lambda op: op.id)]
    capability["capability_sha256"] = snapshot_sha256(CapabilitySnapshot.model_validate_json(json.dumps(
        {k:v for k,v in capability.items() if k not in {"metadata", "capability_sha256"}})))
    headers["X-Synth-Capabilities-SHA256"] = capability["capability_sha256"]
    calls = []

    def respond(wire):
        if wire.url.path.endswith("/capabilities"):
            return httpx.Response(200, headers={"X-Synth-Contract":"index.v1"}, json=capability)
        calls.append(wire)
        metadata = {**capability["metadata"], "contract":"core.v1"}
        retained = dict(request=request.model_dump(mode="json"), admission=admission,
                        released=wire.url.path.endswith("/release"), owner_release_status="not_applicable")
        if wire.method == "POST":
            body = json.loads(wire.content)
            assert wire.headers["Idempotency-Key"] == body["operation_id"] == "stable-capture"
            assert wire.headers["X-Synth-Intent-SHA256"] == hashlib.sha256(wire.content).hexdigest()
        if wire.url.path.endswith("/collect"):
            result = dict(inventory=dict(scope=request.scope.model_dump(mode="json"),
                trial=request.trial.model_dump(mode="json"), profile=request.profile.model_dump(mode="json"),
                execution=json.loads(wire.content)["execution"], execution_inventory=ref("rhodes", "capture_inventory"),
                attempt=dict(workload_id="work", run_id="run", attempt_id="attempt"),
                observed_at="2026-11-03T20:00:00Z", objects=[], gaps=["fixture_has_no_bytes"]))
        elif wire.method == "POST" and wire.url.path.endswith("/captures"):
            result = dict(admission=admission, intent_reference=intent_reference(request).model_dump(mode="json"))
        elif wire.url.path.endswith("/captures"):
            result = dict(captures=[retained], next_cursor=None)
        else:
            result = dict(capture=retained, intent_reference=intent_reference(request).model_dump(mode="json"))
        return httpx.Response(200, headers=headers, json={"metadata":metadata, **result})
    return request, profile, digest, calls, httpx.MockTransport(respond)


@pytest.mark.parametrize("async_mode", [False, True])
def test_admit_read_page_release_preserves_exact_intent_and_retry_body(async_mode):
    request, profile, digest, calls, transport = fixture()
    mutation = MutationOptions(idempotency_key="stable-capture")
    if async_mode:
        async def run():
            async with httpx.AsyncClient(transport=transport) as network, AsyncSynthClient(
                api_key="fixture", base_url="https://backend.test", options=ClientOptions(async_http_client=network)) as client:
                captures = client.hidden("forge").captures
                await captures.admit(PROJECT, request, profile, mutation=mutation)
                await captures.admit(PROJECT, request, profile, mutation=mutation)
                assert (await captures.get(PROJECT, digest)).capture.released is False
                assert len((await captures.retained(PROJECT, limit=1)).captures) == 1
                assert (await captures.release(PROJECT, digest, "trial_abandoned", mutation=mutation)).capture.released
        asyncio.run(run())
    else:
        with httpx.Client(transport=transport) as network, SynthClient(api_key="fixture", base_url="https://backend.test",
                options=ClientOptions(http_client=network)) as client:
            captures = client.hidden("forge").captures
            captures.admit(PROJECT, request, profile, mutation=mutation)
            captures.admit(PROJECT, request, profile, mutation=mutation)
            assert captures.get(PROJECT, digest).capture.released is False
            assert len(captures.retained(PROJECT, limit=1).captures) == 1
            assert captures.release(PROJECT, digest, "trial_abandoned", mutation=mutation).capture.released
    assert calls[0].content == calls[1].content
    assert calls[3].url.params["limit"] == "1"


@pytest.mark.parametrize("name", ["forge_capture_admit", "forge_capture_get", "forge_capture_collect", "forge_capture_retained", "forge_capture_release"])
def test_capture_mcp_arguments_cannot_choose_principal_or_authority(name):
    tool = BY_NAME[name]
    assert tool.request.model_config.get("extra") == "forbid"
    assert not {"principal", "authorization", "bearer", "scorer_identity"} & tool.request.model_fields.keys()


@pytest.mark.parametrize("limit", [True, 0, 129, 1.5])
def test_retained_page_refuses_invalid_bound_before_network(limit):
    from synth_ai.sdk.forge.captures import Captures
    with pytest.raises(ValueError, match="page limit"):
        Captures(None).retained(PROJECT, limit=limit)


def test_async_and_mcp_collect_forward_same_execution_context_and_identity():
    request, profile, digest, calls, transport = fixture()
    execution = ref("rhodes", "execution_attempt")
    async def run():
        async with httpx.AsyncClient(transport=transport) as network, AsyncSynthClient(
            api_key="fixture", base_url="https://backend.test", options=ClientOptions(async_http_client=network)) as client:
            first = await client.hidden("forge").captures.collect(PROJECT, digest, execution,
                orchestra_run_id="run-1", mutation=MutationOptions(idempotency_key="stable-capture"))
            tool = ALL_BY_NAME["forge_capture_collect"]
            second = await tool.execute(client, tool.request.model_validate(dict(project_id=PROJECT,
                digest=digest, execution=execution, orchestra_run_id="run-1", idempotency_key="stable-capture")))
            assert first.inventory == second.inventory
            assert first.inventory.gaps == ("fixture_has_no_bytes",)
    asyncio.run(run())
    assert len(calls) == 2 and calls[0].content == calls[1].content


def test_capture_admission_refuses_other_project_before_network():
    from synth_ai.sdk.forge.captures import Captures
    request, profile, digest, calls, transport = fixture()
    with pytest.raises(ValueError, match="different project"):
        Captures(None).admit("72000000-0000-4000-8000-000000000003", request, profile)
    assert calls == []
