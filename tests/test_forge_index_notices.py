"""SDK/MCP notice pages preserve draft identity and bounded current read intent."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.errors import ContractError
from synth_ai.mcp.catalog import BY_NAME
from synth_ai.mcp.runtime import SDKExecutor
from synth_ai.options import ClientOptions
from synth_ai.types.capabilities import CapabilitySnapshot, snapshot_sha256
from tests.test_forge_index_drafts import OPERATION, PIN, PROJECT
from tests.test_forge_workspaces import receipt_and_headers


def wire(changed=None):
    capability, headers = receipt_and_headers()
    capability["operations"].append(
        {
            "operation_id": "forge.index_drafts.notices",
            "contract": "core.v1",
            "state": "enabled",
            "surfaces": ["http", "mcp"],
            "reason": None,
        }
    )
    capability["operations"].sort(key=lambda item: (item["contract"], item["operation_id"]))
    snapshot = {k: v for k, v in capability.items() if k not in {"metadata", "capability_sha256"}}
    digest = snapshot_sha256(CapabilitySnapshot.model_validate_json(json.dumps(snapshot)))
    capability["capability_sha256"] = headers["X-Synth-Capabilities-SHA256"] = digest
    calls = []

    def respond(request):
        if request.url.path.endswith("/capabilities"):
            return httpx.Response(200, headers={"X-Synth-Contract": "index.v1"}, json=capability)
        calls.append(request)
        assert (
            request.url.path
            == f"/api/v1/index/forge/projects/{PROJECT}/drafts/{OPERATION}/source-notices"
        )
        assert dict(request.url.params) == {"after": "7", "limit": "2"}
        assert request.method == "GET" and not request.content
        result = {
            "metadata": {
                **capability["metadata"],
                "contract": "core.v1",
                "operation_id": OPERATION,
            },
            "project_id": PROJECT,
            "draft_operation_id": OPERATION,
            "export_revision": PIN,
            "release": {
                "authority": "index",
                "kind": "contribution",
                "record_id": "owned",
                "revision": "sealed",
                "digest_sha256": "c" * 64,
            },
            "items": [],
            "after": 7,
            "next_cursor": 9,
        }
        if changed == "project":
            result["project_id"] = "72000000-0000-4000-8000-000000000099"
        elif changed == "draft":
            result["draft_operation_id"] = "other"
        elif changed == "operation":
            result["metadata"]["operation_id"] = "other"
        elif changed == "after":
            result["after"] = 6
        return httpx.Response(200, headers=headers, json=result)

    return httpx.MockTransport(respond), calls


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("changed", [None, "project", "draft", "operation", "after"])
def test_private_notices_identity_and_read_only_mcp(async_mode, changed):
    transport, calls = wire(changed)

    async def run():
        async with (
            httpx.AsyncClient(transport=transport) as network,
            AsyncSynthClient(
                api_key="fixture",
                base_url="https://backend.test",
                options=ClientOptions(async_http_client=network),
            ) as client,
        ):
            if changed:
                with pytest.raises(ContractError, match="different private notice page"):
                    await client.forge.index_drafts.notices(PROJECT, OPERATION, after=7, limit=2)
            else:
                tool = BY_NAME["forge_index_draft_notices"]
                result = await tool.invoke(
                    client,
                    {"project_id": PROJECT, "operation_id": OPERATION, "after": 7, "limit": 2},
                )
                assert result["next_cursor"] == 9 and not result["items"]

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
            if changed:
                with pytest.raises(ContractError, match="different private notice page"):
                    client.forge.index_drafts.notices(PROJECT, OPERATION, after=7, limit=2)
            else:
                page = client.forge.index_drafts.notices(PROJECT, OPERATION, after=7, limit=2)
                assert page.after == 7 and page.next_cursor == 9 and not page.items
    assert len(calls) == 1


def test_private_notice_tool_requires_both_original_grants_and_closed_arguments():
    tool = BY_NAME["forge_index_draft_notices"]
    values = {"project_id": PROJECT, "operation_id": OPERATION}
    for extra in (
        {"after": True},
        {"after": -1},
        {"limit": 101},
        {"principal_id": "injected"},
        {"manifest_digest": "a" * 64},
    ):
        with pytest.raises(ValueError):
            tool.validate_arguments({**values, **extra})
    executor = SDKExecutor.__new__(SDKExecutor)
    executor._enabled = frozenset({tool.name})
    assert not executor._permitted(tool, SimpleNamespace(scopes=("forge:read",)), tool.scope)
    assert not executor._permitted(tool, SimpleNamespace(scopes=("index:intake",)), tool.scope)
    assert executor._permitted(
        tool, SimpleNamespace(scopes=("forge:read", "index:intake")), tool.scope
    )
