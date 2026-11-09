"""Offline smoke run by the isolated wheel interpreter, never the checkout imports."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from synth_ai import AsyncSynthClient, StreamCheckpoint
from synth_ai.models import RunSnapshot


async def sdk_smoke() -> None:
    snapshot = RunSnapshot(
        run_id="run_AAAAAAAAAAAAAAAAAAAAAA",
        metadata={},
        labels={},
        labels_revision=1,
        objectives=[],
        guidance_coverage=[],
        state="completed",
        model="example",
        provider="example",
        result_available=False,
        execution_closed=True,
        accounting_final=True,
        cleanup_complete=True,
        problems=[],
        problems_observed=False,
        created_at="2026-11-04T18:00:00Z",
        started_at=None,
        closed_at=None,
    )
    calls: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.headers["accept"] == "text/event-stream":
            calls.append("subscribe")
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=b"")
        calls.append("read")
        return httpx.Response(200, content=snapshot.model_dump_json())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        async with AsyncSynthClient(
            "fixture", base_url="https://example.test", http_client=http
        ) as client:
            observed = await client.swarms.wait(snapshot.run_id, timeout=1)
            if not observed.finished or calls != ["subscribe", "read"]:
                raise ValueError("installed observation smoke failed")
    checkpoint = StreamCheckpoint(run_id=snapshot.run_id, last_event_id="7")
    if StreamCheckpoint.from_dict(checkpoint.to_dict()) != checkpoint:
        raise ValueError("installed checkpoint smoke failed")


async def mcp_smoke() -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from synth_ai.mcp.tools import TOOLS

    params = StdioServerParameters(
        command=sys.executable,
        args=["-I", "-c", "from synth_ai.mcp.server import main; main()"],
        env={
            "PATH": os.environ.get("PATH", ""),
            "SYNTH_API_KEY": "fixture",
            "SYNTH_AUTH_FILE": str(Path.cwd() / "absent-auth.json"),
        },
    )
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            catalog = await session.list_tools()
            if {tool.name: tool.inputSchema for tool in catalog.tools} != {
                tool["name"]: tool["inputSchema"] for tool in TOOLS
            }:
                raise ValueError("installed MCP catalog drift")
            if not (await session.call_tool("unknown_tool", {})).isError:
                raise ValueError("installed MCP error signal missing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcp", action="store_true")
    args = parser.parse_args()
    asyncio.run(mcp_smoke() if args.mcp else sdk_smoke())
    print(json.dumps({"installed_mcp" if args.mcp else "installed_sdk": "passed"}))


if __name__ == "__main__":
    main()
