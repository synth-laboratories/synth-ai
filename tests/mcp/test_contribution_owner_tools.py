"""Private owner MCP tools use existing keyed SDK submission and owner routes."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsRequest
from synth_ai import AsyncSynthClient
from synth_ai.mcp.catalog import BY_NAME
from synth_ai.mcp.runtime import SDKExecutor
from synth_ai.mcp.server import create_server
from tests.test_forge_workspaces import receipt_and_headers

ARGS = {
    "contribution_id": "private-contribution",
    "revision_id": "exact-revision",
    "idempotency_key": "original-submission",
}


def test_protocol_uses_exact_revision_key_and_dedicated_owner_route():
    async def run():
        requests = []
        capability, _ = receipt_and_headers()
        observed = {
            **capability["metadata"],
            "contract": "index.v1",
            "operation_id": "original-operation",
        }

        def respond(request):
            requests.append(request)
            if request.method == "POST":
                assert json.loads(request.content) == {"revision_id": ARGS["revision_id"]}
                assert request.headers["Idempotency-Key"] == ARGS["idempotency_key"]
                assert (
                    request.headers["X-Synth-Intent-SHA256"]
                    == hashlib.sha256(request.content).hexdigest()
                )
                body = {
                    "metadata": observed,
                    "contribution_id": ARGS["contribution_id"],
                    "revision_id": ARGS["revision_id"],
                    "state": "submitted",
                    "reason": None,
                }
            else:
                body = {
                    "metadata": observed,
                    "contribution_id": ARGS["contribution_id"],
                    "revision_id": ARGS["revision_id"],
                    "title": "Private fixture",
                    "status": "submitted",
                    "contents_readable": False,
                }
            return httpx.Response(200, headers={"X-Synth-Contract": "index.v1"}, json=body)

        async with AsyncSynthClient(api_key="fixture", base_url="https://backend.test") as client:
            await client._transport.client.aclose()
            client._transport.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))

            async def negotiated(operation, deadline):
                return None

            client._transport._negotiate = negotiated

            class Executor:
                async def available_tools(self):
                    return frozenset({"index_contribution_submit", "index_contribution_owner_get"})

                async def invoke(self, tool, arguments):
                    return await tool.invoke(client, arguments)

            server = create_server(Executor())
            listed = (
                await server.request_handlers[ListToolsRequest](
                    ListToolsRequest(method="tools/list")
                )
            ).root
            tools = {tool.name: tool for tool in listed.tools}
            assert not tools["index_contribution_submit"].annotations.readOnlyHint
            assert tools["index_contribution_owner_get"].annotations.readOnlyHint
            for name, arguments in [
                ("index_contribution_submit", ARGS),
                ("index_contribution_submit", ARGS),
                ("index_contribution_owner_get", {"contribution_id": ARGS["contribution_id"]}),
            ]:
                result = (
                    await server.request_handlers[CallToolRequest](
                        CallToolRequest(
                            method="tools/call",
                            params=CallToolRequestParams(name=name, arguments=arguments),
                        )
                    )
                ).root
                assert not result.isError, result
                assert result.structuredContent["revision_id"] == ARGS["revision_id"]
        assert requests[0].content == requests[1].content
        assert requests[0].url.path == "/api/v1/index/contributions/private-contribution/submit"
        assert requests[2].url.path == "/api/v1/index/contributions/private-contribution/owner"

    asyncio.run(run())


def test_private_tools_require_intake_and_forbid_caller_or_rights_arguments():
    executor = SDKExecutor.__new__(SDKExecutor)
    executor._enabled = frozenset({"index_contribution_submit", "index_contribution_owner_get"})
    for name in executor._enabled:
        tool = BY_NAME[name]
        assert tool.scope == "index:intake"
        assert not executor._permitted(tool, SimpleNamespace(scopes=("index:read",)), tool.scope)
        assert executor._permitted(tool, SimpleNamespace(scopes=("index:intake",)), tool.scope)
    for field in (
        "subject_id",
        "credential_id",
        "rights_attested",
        "requested_audience",
        "package_id",
    ):
        with pytest.raises(ValueError):
            BY_NAME["index_contribution_submit"].validate_arguments({**ARGS, field: "injected"})
