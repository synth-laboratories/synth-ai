"""SDK/MCP transport identity against a synthetic private draft wire fixture."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.mcp.catalog import BY_NAME
from synth_ai.mcp.runtime import SDKExecutor
from synth_ai.options import ClientOptions, MutationOptions
from synth_ai.sdk.forge.index_types import ForgeIndexDraftResponse
from synth_ai.types.capabilities import CapabilitySnapshot, snapshot_sha256
from tests.test_forge_workspaces import receipt_and_headers

PROJECT = "72000000-0000-4000-8000-000000000002"
OPERATION = "72000000-0000-4000-8000-000000000006"
PIN = {
    "schema_version": "forge.revision-reference.v1",
    "reference": {
        "authority": "forge",
        "kind": "scientific_export",
        "record_id": "export",
        "revision": "1",
        "digest_sha256": "a" * 64,
    },
    "document_digest_sha256": "b" * 64,
}


def transport_fixture():
    capability, headers = receipt_and_headers()
    for operation in capability["operations"]:
        operation["operation_id"] = (
            operation["operation_id"]
            .replace("forge.projects.apply", "forge.index_drafts.create")
            .replace("forge.projects.get", "forge.index_drafts.get")
        )
    capability["capability_sha256"] = snapshot_sha256(
        CapabilitySnapshot.model_validate_json(
            json.dumps(
                {k: v for k, v in capability.items() if k not in {"metadata", "capability_sha256"}}
            )
        )
    )
    headers["X-Synth-Capabilities-SHA256"] = capability["capability_sha256"]
    calls = []

    def respond(request):
        if request.url.path.endswith("/capabilities"):
            return httpx.Response(200, headers={"X-Synth-Contract": "index.v1"}, json=capability)
        calls.append(request)
        if request.method == "POST":
            assert json.loads(request.content) == {"export_revision": PIN}
            assert request.headers["Idempotency-Key"] == "original-private-draft"
            assert (
                request.headers["X-Synth-Intent-SHA256"]
                == hashlib.sha256(request.content).hexdigest()
            )
        result = {
            "metadata": {**capability["metadata"], "contract": "core.v1", "operation_id": OPERATION},
            "project_id": PROJECT,
            "export_revision": PIN,
            "contribution_id": "allocated-contribution",
            "revision_id": "allocated-revision",
            "collection_id": "allocated-collection",
            "status": "draft",
            "validation_state": "awaiting_upload",
            "manifest_digest_sha256": None,
        }
        validated = ForgeIndexDraftResponse.model_validate_json(json.dumps(result))
        return httpx.Response(200, headers=headers, json=validated.model_dump(mode="json"))

    return httpx.MockTransport(respond), calls


@pytest.mark.parametrize("async_mode", [False, True])
def test_clients_preserve_exact_pin_retry_key_and_original_operation(async_mode):
    transport, calls = transport_fixture()

    async def run():
        async with (
            httpx.AsyncClient(transport=transport) as network,
            AsyncSynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(async_http_client=network),
            ) as client,
        ):
            first = await client.forge.index_drafts.create(
                PROJECT, PIN, mutation=MutationOptions(idempotency_key="original-private-draft")
            )
            again = await client.forge.index_drafts.create(
                PROJECT, PIN, mutation=MutationOptions(idempotency_key="original-private-draft")
            )
            found = await client.forge.index_drafts.get(PROJECT, OPERATION)
            assert (
                first.model_dump(exclude={"metadata"})
                == again.model_dump(exclude={"metadata"})
                == found.model_dump(exclude={"metadata"})
            )
            assert all(
                result.metadata.operation_id == OPERATION for result in (first, again, found)
            )

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
            first = client.forge.index_drafts.create(
                PROJECT, PIN, mutation=MutationOptions(idempotency_key="original-private-draft")
            )
            again = client.forge.index_drafts.create(
                PROJECT, PIN, mutation=MutationOptions(idempotency_key="original-private-draft")
            )
            found = client.forge.index_drafts.get(PROJECT, OPERATION)
            assert (
                first.model_dump(exclude={"metadata"})
                == again.model_dump(exclude={"metadata"})
                == found.model_dump(exclude={"metadata"})
            )
            assert all(
                result.metadata.operation_id == OPERATION for result in (first, again, found)
            )
    assert len(calls) == 3 and calls[0].content == calls[1].content


def test_mcp_uses_both_grants_and_forbids_identity_or_public_rights_arguments():
    tool = BY_NAME["forge_index_draft_create"]
    args = {
        "project_id": PROJECT,
        "export_revision": PIN,
        "idempotency_key": "original-private-draft",
    }
    tool.validate_arguments(args)
    for field in ("credential_id", "subject_id", "rights_attested", "requested_audience"):
        with pytest.raises(ValueError):
            tool.validate_arguments({**args, field: "injected"})
    executor = SDKExecutor.__new__(SDKExecutor)
    executor._enabled = frozenset({tool.name})
    assert not executor._permitted(tool, SimpleNamespace(scopes=("forge:read",)), tool.scope)
    assert executor._permitted(
        tool, SimpleNamespace(scopes=("forge:read", "index:intake")), tool.scope
    )
    transport, _ = transport_fixture()

    async def run():
        async with (
            httpx.AsyncClient(transport=transport) as network,
            AsyncSynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(async_http_client=network),
            ) as client,
        ):
            result = await tool.invoke(client, args)
            assert result["metadata"]["operation_id"] == OPERATION
            assert result["validation_state"] == "awaiting_upload"

    asyncio.run(run())
