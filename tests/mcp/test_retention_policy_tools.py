"""MCP retention policy tools reach the existing keyed SDK routes (SYN-4571)."""

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

pytest.importorskip("mcp.types", reason="requires the synth-ai[mcp] extra")

from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsRequest  # noqa: E402
from synth_ai import AsyncSynthClient
from synth_ai.mcp.catalog import BY_NAME
from synth_ai.mcp.server import create_server

pytestmark = pytest.mark.mcp

POLICY = {
    "metadata": {
        "contract": "index.v1",
        "request_id": "retention-policy",
        "operation_id": None,
        "observed_at": "2026-10-19T00:00:00Z",
    },
    "default_mode": "zdr",
    "allowed_modes": ["zdr", "zdr_strict"],
    "enforced": True,
    "revision": 4,
}
UPDATE = {
    "default_mode": "zdr",
    "allowed_modes": ["zdr", "zdr_strict"],
    "enforced": True,
    "expected_revision": 3,
    "idempotency_key": "retention-policy-1",
}


def test_protocol_discovery_and_calls_use_keyed_idempotent_sdk_operations():
    async def run():
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, headers={"X-Synth-Contract": "index.v1"}, json=POLICY)

        async with AsyncSynthClient(api_key="test-key", base_url="https://backend.test") as client:
            await client._transport.client.aclose()
            client._transport.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))

            async def negotiated(operation, deadline):
                return None

            client._transport._negotiate = negotiated

            class Executor:
                async def available_tools(self):
                    return frozenset({"index_retention_get", "index_retention_update"})

                async def invoke(self, tool, arguments):
                    return await tool.invoke(client, arguments)

            server = create_server(Executor())
            listed = (
                await server.request_handlers[ListToolsRequest](
                    ListToolsRequest(method="tools/list")
                )
            ).root
            tools = {tool.name: tool for tool in listed.tools}
            assert tools["index_retention_get"].annotations.readOnlyHint
            assert not tools["index_retention_update"].annotations.readOnlyHint
            for name, arguments in [
                ("index_retention_get", {}),
                ("index_retention_update", UPDATE),
            ]:
                result = (
                    await server.request_handlers[CallToolRequest](
                        CallToolRequest(
                            method="tools/call",
                            params=CallToolRequestParams(name=name, arguments=arguments),
                        )
                    )
                ).root
                assert not result.isError
                assert result.structuredContent["revision"] == 4

        get, put = requests
        assert (get.method, put.method) == ("GET", "PUT")
        assert get.url.path == put.url.path == "/api/v1/index/me/retention"
        assert put.headers["Idempotency-Key"] == UPDATE["idempotency_key"]
        assert json.loads(put.content) == {
            k: v for k, v in UPDATE.items() if k != "idempotency_key"
        }
        assert BY_NAME["index_retention_get"].keyed
        assert BY_NAME["index_retention_update"].keyed
        assert not BY_NAME["index_retention_update"].paid

    asyncio.run(run())


@pytest.mark.parametrize(
    "change",
    [
        {"expected_revision": None},
        {"expected_revision": -1},
        {"idempotency_key": ""},
        {"allowed_modes": ["zdr", "zdr"]},
        {"default_mode": "retained"},
        {"api_key": "forbidden"},
    ],
)
def test_invalid_or_unbound_policy_changes_cannot_dispatch(change):
    with pytest.raises(ValidationError):
        BY_NAME["index_retention_update"].validate_arguments({**UPDATE, **change})
