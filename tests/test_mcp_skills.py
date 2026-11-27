"""SYN-6064: Skills over MCP (SEP-2640 / io.modelcontextprotocol/skills) on local stdio MCP."""

from __future__ import annotations

import hashlib
import os
import sys

import anyio
import httpx
import pytest
from pydantic import AnyUrl

from synth_ai import AsyncSynthClient
from synth_ai.mcp import skills


async def _raw_rpc(server, requests):
    """Send raw JSON-RPC requests to a Synth MCP server over in-memory streams."""
    from mcp.shared.memory import create_client_server_memory_streams
    from mcp.shared.message import SessionMessage
    from mcp.types import JSONRPCMessage, JSONRPCRequest

    responses = []
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: server.run(
                    server_read,
                    server_write,
                    server.create_initialization_options(),
                )
            )
            try:
                async with client_read, client_write:
                    for req in requests:
                        msg = SessionMessage(message=JSONRPCMessage(JSONRPCRequest(**req)))
                        await client_write.send(msg)
                        reply = await client_read.receive()
                        assert isinstance(reply, SessionMessage)
                        responses.append(reply.message.root.model_dump(mode="json", by_alias=True))
            finally:
                tg.cancel_scope.cancel()
    return responses


async def test_mcp_skills_capabilities_discovery_and_digest_integrity():
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import TextResourceContents

    from synth_ai.mcp.server import build_server

    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(500)))
    async with AsyncSynthClient(
        "fixture", base_url="http://127.0.0.1:9", http_client=http
    ) as client:
        server = build_server(client, upload_descriptor=None)

        # 1. Verify initialize advertises resources + io.modelcontextprotocol/skills
        init_replies = await _raw_rpc(
            server,
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "test-client", "version": "1.0"},
                    },
                },
                {"jsonrpc": "2.0", "id": 2, "method": "skills/list", "params": {}},
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "skills/get",
                    "params": {"uri": "skill://swarms-launch/SKILL.md"},
                },
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "resources/directory/read",
                    "params": {"uri": "skill://swarms-launch/"},
                },
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "resources/directory/read",
                    "params": {"uri": "skill://swarms-launch/references/"},
                },
            ],
        )

        caps = init_replies[0]["result"]["capabilities"]
        assert "resources" in caps
        assert caps["extensions"]["io.modelcontextprotocol/skills"] == {"directoryRead": True}

        listed_skills = init_replies[1]["result"]["skills"]
        assert {item["name"] for item in listed_skills} == {
            "swarms-billing",
            "swarms-launch",
            "swarms-research",
            "swarms-supervise",
        }
        launch_from_list = next(item for item in listed_skills if item["name"] == "swarms-launch")
        assert init_replies[2]["result"] == launch_from_list
        assert launch_from_list["revision"].startswith("sha256:")
        assert len(launch_from_list["revision"]) == len("sha256:") + 64

        launch_dir = init_replies[3]["result"]["entries"]
        assert [(e["name"], e["isDirectory"]) for e in launch_dir] == [
            ("SKILL.md", False),
            ("references", True),
        ]
        ref_dir = init_replies[4]["result"]["entries"]
        assert [(e["name"], e["isDirectory"], e["uri"]) for e in ref_dir] == [
            (
                "budget-and-choices.md",
                False,
                "skill://swarms-launch/references/budget-and-choices.md",
            )
        ]

        # 2. Verify standard MCP ClientSession list_resources + read_resource match SHA-256 digests
        async with create_connected_server_and_client_session(server) as session:
            resources = (await session.list_resources()).resources
            assert len(resources) == 5
            for sk in listed_skills:
                for res in sk["resources"]:
                    read = await session.read_resource(AnyUrl(res["uri"]))
                    assert len(read.contents) == 1
                    item = read.contents[0]
                    assert isinstance(item, TextResourceContents)
                    assert item.mimeType == "text/markdown"
                    raw = item.text.encode("utf-8")
                    assert len(raw) == res["size"]
                    assert "sha256:" + hashlib.sha256(raw).hexdigest() == res["digest"]


async def test_mcp_skills_fail_closed_on_traversal_and_unknown_uris():
    pytest.importorskip("mcp")
    from synth_ai.mcp.server import build_server

    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(500)))
    async with AsyncSynthClient(
        "fixture", base_url="http://127.0.0.1:9", http_client=http
    ) as client:
        server = build_server(client, upload_descriptor=None)

        bad_file_uris = [
            "",
            "file:///etc/passwd",
            "https://example.com/SKILL.md",
            "skill://swarms-launch/../swarms-billing/SKILL.md",
            "skill://swarms-launch/references/../SKILL.md",
            "skill://swarms-launch/%2e%2e/swarms-billing/SKILL.md",
            "skill://swarms-launch/SKILL.md?x=1",
            "skill://swarms-launch/",
        ]
        reqs = [
            {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0"},
                },
            }
        ]
        next_id = 1
        for method in ("skills/get", "resources/read"):
            for uri in bad_file_uris:
                reqs.append(
                    {"jsonrpc": "2.0", "id": next_id, "method": method, "params": {"uri": uri}}
                )
                next_id += 1
        for bad_dir in (
            "skill://swarms-launch/SKILL.md",
            "skill://swarms-launch",
            "skill://swarms-launch/../",
            "file:///tmp/",
        ):
            reqs.append(
                {
                    "jsonrpc": "2.0",
                    "id": next_id,
                    "method": "resources/directory/read",
                    "params": {"uri": bad_dir},
                }
            )
            next_id += 1

        unknown_checks = [
            ("skills/get", "skill://nonexistent/SKILL.md"),
            ("resources/read", "skill://swarms-launch/missing.md"),
            ("resources/directory/read", "skill://nonexistent/"),
        ]
        for method, uri in unknown_checks:
            reqs.append({"jsonrpc": "2.0", "id": next_id, "method": method, "params": {"uri": uri}})
            next_id += 1

        replies = await _raw_rpc(server, reqs)
        invalid_count = 2 * len(bad_file_uris) + 4
        for reply in replies[1 : 1 + invalid_count]:
            assert reply["error"]["code"] == skills.INVALID_PARAMS, reply
        for reply in replies[1 + invalid_count :]:
            assert reply["error"]["code"] == skills.RESOURCE_NOT_FOUND, reply


async def test_stdio_subprocess_serves_skills_and_capabilities(tmp_path):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import TextResourceContents

    env = dict(os.environ, SYNTH_API_KEY="fixture", SYNTH_AUTH_FILE=str(tmp_path / "absent.json"))
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from synth_ai.mcp.server import main; main()"],
        env=env,
    )
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            init = await session.initialize()
            caps = init.capabilities.model_dump(by_alias=True, mode="json", exclude_none=True)
            assert caps["extensions"]["io.modelcontextprotocol/skills"] == {"directoryRead": True}
            resources = (await session.list_resources()).resources
            uris = {str(r.uri) for r in resources}
            assert "skill://swarms-launch/SKILL.md" in uris
            assert "skill://swarms-launch/references/budget-and-choices.md" in uris
            read = await session.read_resource(AnyUrl("skill://swarms-billing/SKILL.md"))
            assert isinstance(read.contents[0], TextResourceContents)
            assert "account_billing_balance" in read.contents[0].text

    # Verify custom SEP-2640 JSON-RPC requests over real stdio transport
    from mcp.shared.message import SessionMessage
    from mcp.types import JSONRPCMessage, JSONRPCRequest

    async with stdio_client(params) as (reader, writer):
        await writer.send(
            SessionMessage(
                message=JSONRPCMessage(
                    JSONRPCRequest(jsonrpc="2.0", id="stdio-skills", method="skills/list")
                )
            )
        )
        reply = await reader.receive()
        assert isinstance(reply, SessionMessage)
        dumped = reply.message.root.model_dump(mode="json", by_alias=True)
        assert len(dumped["result"]["skills"]) == 4
