"""SYN-5723: the stdio MCP server can run over a caller-supplied HTTP client.

A harness that must route the public createRun call through its own transport
(the cached replay grant) launches through the real MCP server and tool
dispatch, instead of a parallel MCP implementation.
"""

from __future__ import annotations

import json

import httpx
import pytest

from synth_ai import AsyncSynthClient

RUN = "run_AAAAAAAAAAAAAAAAAAAAAA"


async def test_server_over_injected_client_launches_queued_run(contract):
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import TextContent

    from synth_ai.mcp.server import build_server

    snapshot = json.loads(
        json.dumps(
            contract["paths"]["/v1/runs/{run_id}"]["get"]["responses"]["200"]["content"][
                "application/json"
            ]["example"]
        )
    )
    snapshot.update(
        run_id=RUN,
        state="queued",
        queue={
            "position": 2,
            "ahead": 1,
            "estimated_start_at": None,
            "expires_at": "2026-11-17T06:00:00Z",
        },
        execution_closed=False,
        accounting_final=False,
        cleanup_complete=False,
    )
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(202, json=snapshot)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    async with AsyncSynthClient(
        "fixture", base_url="http://127.0.0.1:9", http_client=http
    ) as client:
        server = build_server(client, upload_descriptor=None)
        async with create_connected_server_and_client_session(server) as session:
            tools = {tool.name for tool in (await session.list_tools()).tools}
            create = "swarms_run_launch"
            assert create in tools
            body = {
                "message": "fixture task",
                "limits": {
                    "maximum_charge": {"amount": "0.10", "currency": "USD"},
                    "wall_time_seconds": 30,
                },
            }
            result = await session.call_tool(
                create, {"body": body, "idempotency_key": "fixture-mcp-queued"}
            )
    content = result.content[0]
    assert isinstance(content, TextContent)
    assert not result.isError, content.text
    launched = json.loads(content.text)
    assert launched["run_id"] == RUN and launched["state"] == "queued"
    assert launched["queue"]["position"] == 2
    assert [(r.method, r.url.path, r.headers.get("Idempotency-Key")) for r in seen] == [
        ("POST", "/v1/runs", "fixture-mcp-queued")
    ]


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("unknown_tool", {}),
        ("swarms_run_cancel", {"run_id": RUN, "idempotency_key": "   "}),
        ("swarms_run_cancel", {"run_id": RUN, "idempotency_key": "bad\nkey"}),
        ("swarms_run_cancel", {"run_id": RUN, "idempotency_key": "é" * 65}),
        ("swarms_run_get", {"run_id": RUN, "unexpected": "secret-value"}),
    ],
)
async def test_server_invoke_converts_value_error_to_problem(tool_name, arguments):
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import TextContent

    from synth_ai.mcp.server import build_server

    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(500)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    async with AsyncSynthClient(
        "fixture", base_url="http://127.0.0.1:9", http_client=http
    ) as client:
        server = build_server(client, upload_descriptor=None)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(tool_name, arguments)
    assert result.isError
    content = result.content[0]
    assert isinstance(content, TextContent)
    problem = json.loads(content.text)
    assert problem["code"] == "invalid_arguments"
    assert problem["status"] == 422
    assert problem["stage"] == "mcp_input"
    assert problem["outcome"] == "not_accepted"
    assert "\n" not in problem["detail"]
    assert "secret-value" not in problem["detail"]
    assert seen == []


@pytest.mark.parametrize(
    "bad_key",
    [
        "a" * 129,
        "é" * 65,
        "bad\nkey",
        "bad\rkey",
        "bad\0key",
        "   ",
    ],
)
async def test_server_invoke_directory_upload_rejects_bad_idempotency_key(bad_key):
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import TextContent

    from synth_ai.mcp import directory
    from synth_ai.mcp.server import build_server

    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(500)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    async with AsyncSynthClient(
        "fixture", base_url="http://127.0.0.1:9", http_client=http
    ) as client:
        server = build_server(client, upload_descriptor=10)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(directory.NAME, {"idempotency_key": bad_key})
    assert result.isError
    content = result.content[0]
    assert isinstance(content, TextContent)
    problem = json.loads(content.text)
    assert problem["code"] == "tool_arguments_invalid"
    assert problem["status"] == 422
    assert problem["stage"] == "mcp_input"
    assert "128" in problem["detail"]
    assert "\n" not in problem["detail"]
    assert seen == []


async def test_server_invoke_reports_field_level_argument_diagnostics():
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import TextContent

    from synth_ai.mcp.server import build_server

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    async with AsyncSynthClient(
        "fixture", base_url="http://127.0.0.1:9", http_client=http
    ) as client:
        server = build_server(client, upload_descriptor=None)
        async with create_connected_server_and_client_session(server) as session:
            missing_res = await session.call_tool("swarms_run_get", {})
            unexpected_res = await session.call_tool(
                "swarms_run_get", {"run_id": RUN, "bogus_key": "secret-val"}
            )
            nested_res = await session.call_tool(
                "swarms_run_launch",
                {
                    "idempotency_key": "k1",
                    "body": {
                        "message": "task",
                        "limits": {
                            "maximum_charge": {"amount": "1.00", "currency": "USD"},
                            "wall_time_seconds": "not-an-int",
                        },
                    },
                },
            )

    missing_content = missing_res.content[0]
    assert missing_res.isError and isinstance(missing_content, TextContent)
    missing_problem = json.loads(missing_content.text)
    assert "missing required argument(s): run_id" in missing_problem["detail"]

    unexpected_content = unexpected_res.content[0]
    assert unexpected_res.isError and isinstance(unexpected_content, TextContent)
    unexpected_problem = json.loads(unexpected_content.text)
    assert "unexpected argument(s): bogus_key" in unexpected_problem["detail"]
    assert "secret-val" not in unexpected_problem["detail"]

    nested_content = nested_res.content[0]
    assert nested_res.isError and isinstance(nested_content, TextContent)
    nested_problem = json.loads(nested_content.text)
    assert nested_problem["code"] == "tool_arguments_invalid"
    assert "$.body.limits.wall_time_seconds:" in nested_problem["detail"]
