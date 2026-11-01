"""Native project trace SDK uses negotiated scoped metadata and verified bytes."""

import asyncio
import hashlib
import json

import httpx
import pytest
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.contracts.fingerprint import schema_sha256
from synth_ai.errors import ContractError, UnsupportedOperationError
from synth_ai.options import ClientOptions
from synth_ai.types.capabilities import CapabilitySnapshot, snapshot_sha256
from tests.test_forge_workspaces import receipt_and_headers

ORG = "81000000-0000-4000-8000-000000000001"
PROJECT = "81000000-0000-4000-8000-000000000002"
IDENTIFIER = "mna_" + "a" * 64
BODY = b"native zip bytes"
DIGEST = hashlib.sha256(BODY).hexdigest()


def fixture(corruption=None, advertise=True):
    capability, headers = receipt_and_headers()
    capability["contracts"].append(
        {"contract": "index.v1", "schema_sha256": schema_sha256("index.v1")}
    )
    capability["operations"].append(
        {
            "operation_id": "auth.context",
            "contract": "index.v1",
            "state": "enabled",
            "surfaces": ["http"],
            "reason": None,
        }
    )
    if advertise:
        for name in ("list", "get", "content"):
            capability["operations"].append(
                {
                    "operation_id": "mnemosyne.native_traces." + name,
                    "contract": "core.v1",
                    "state": "enabled",
                    "surfaces": ["http"],
                    "reason": None,
                }
            )
    capability["operations"].sort(key=lambda x: x["operation_id"])
    snap = {
        k: v
        for k, v in capability.items()
        if k not in {"metadata", "capability_sha256"}
    }
    capability["capability_sha256"] = headers["X-Synth-Capabilities-SHA256"] = (
        snapshot_sha256(CapabilitySnapshot.model_validate_json(json.dumps(snap)))
    )
    item = {
        "association_id": IDENTIFIER,
        "source_kind": "configured_customer",
        "rollout_id": "rollout_1",
        "execution_epoch": "epoch_1",
        "run_id": None,
        "created_at": "2026-10-31T00:00:00Z",
        "archive_sha256": DIGEST,
        "archive_size_bytes": len(BODY),
    }
    scope = {"organization_id": ORG, "project_id": PROJECT}
    calls = []

    def respond(request):
        path = request.url.path
        if path == "/api/v1/capabilities":
            return httpx.Response(
                200, headers={"X-Synth-Contract": "index.v1"}, json=capability
            )
        if path == "/api/v1/auth/context":
            return httpx.Response(
                200,
                headers={"X-Synth-Contract": "index.v1"},
                json={
                    "metadata": capability["metadata"],
                    "tenant_id": ORG,
                    "subject_id": "user",
                    "credential_id": "key",
                    "credential_kind": "api_key",
                    "oauth_client_id": None,
                    "subject_kind": "user",
                    "scopes": ["forge:read"],
                },
            )
        calls.append(path)
        if path.endswith("/content"):
            result_headers = {
                **headers,
                "Content-Type": "application/zip",
                "X-Content-SHA256": DIGEST,
                "X-Mnemosyne-Organization-Id": ORG,
                "X-Mnemosyne-Project-Id": PROJECT,
                "X-Mnemosyne-Association-Id": IDENTIFIER,
            }
            for name, key in (
                ("org", "X-Mnemosyne-Organization-Id"),
                ("project", "X-Mnemosyne-Project-Id"),
                ("id", "X-Mnemosyne-Association-Id"),
                ("digest", "X-Content-SHA256"),
            ):
                if corruption == name:
                    result_headers[key] = "wrong"
            payload = BODY + (b"wrong" if corruption == "body" else b"")
            return httpx.Response(
                200, headers=result_headers, stream=httpx.ByteStream(payload)
            )
        if path.endswith("/native-traces"):
            data = {
                "schema_version": "synth.mnemosyne.native-trace-page.v1",
                "scope": scope,
                "items": [item],
                "next_cursor": None,
            }
        else:
            data = {
                **item,
                "scope": {
                    **scope,
                    "project_id": "different" if corruption == "metadata" else PROJECT,
                },
                "schema_version": "synth.mnemosyne.native-trace.v1",
            }
        return httpx.Response(
            200,
            headers=headers,
            json={
                "metadata": {**capability["metadata"], "contract": "core.v1"},
                "data": data,
            },
        )

    return httpx.MockTransport(respond), calls


@pytest.mark.parametrize("async_mode", [False, True])
def test_native_trace_list_get_and_verified_download(async_mode):
    transport, calls = fixture()

    async def run():
        async with (
            httpx.AsyncClient(transport=transport) as network,
            AsyncSynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(async_http_client=network),
            ) as client,
        ):
            native = client.mnemosyne.native_traces
            assert (await native.list(PROJECT)).data.items[0].run_id is None
            assert (
                await native.get(PROJECT, IDENTIFIER)
            ).data.association_id == IDENTIFIER
            assert await native.download(PROJECT, IDENTIFIER) == BODY

    if async_mode:
        asyncio.run(run())
    else:
        with (
            httpx.Client(transport=transport) as network,
            SynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(http_client=network),
            ) as client,
        ):
            native = client.mnemosyne.native_traces
            assert native.list(PROJECT).data.items[0].run_id is None
            assert native.get(PROJECT, IDENTIFIER).data.association_id == IDENTIFIER
            assert native.download(PROJECT, IDENTIFIER) == BODY
    assert calls[-1].endswith("/content")


@pytest.mark.parametrize(
    "corruption", ["metadata", "org", "project", "id", "digest", "body"]
)
def test_download_refuses_misbound_or_changed_content(corruption):
    transport, _ = fixture(corruption)
    with (
        httpx.Client(transport=transport) as network,
        SynthClient(
            api_key="fixture",
            base_url="https://backend.test",
            options=ClientOptions(http_client=network),
        ) as client,
        pytest.raises(ContractError),
    ):
        client.mnemosyne.native_traces.download(PROJECT, IDENTIFIER)


def test_missing_capability_refuses_before_owner_call():
    transport, calls = fixture(advertise=False)
    with (
        httpx.Client(transport=transport) as network,
        SynthClient(
            api_key="fixture",
            base_url="https://backend.test",
            options=ClientOptions(http_client=network),
        ) as client,
        pytest.raises(UnsupportedOperationError),
    ):
        client.mnemosyne.native_traces.list(PROJECT)
    assert calls == []


def test_program_source_requires_actual_run_identity():
    from synth_ai.sdk.mnemosyne.native_traces import NativeTraceItem
    from pydantic import ValidationError

    value = {
        "association_id": IDENTIFIER,
        "source_kind": "swarm_program",
        "rollout_id": "rollout_1",
        "execution_epoch": "epoch_1",
        "run_id": "00000000-0000-0000-0000-000000000001",
        "created_at": "2026-10-31T00:00:00Z",
        "archive_sha256": "a" * 64,
        "archive_size_bytes": 1,
    }
    assert NativeTraceItem.model_validate_json(json.dumps(value)).source_kind == "swarm_program"
    with pytest.raises(ValidationError):
        NativeTraceItem.model_validate_json(json.dumps(value | {"run_id": None}))
