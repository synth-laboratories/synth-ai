import os
import sys

import pytest

from synth_ai.mcp.tools import TOOLS


async def test_stdio_catalog_and_error_signal(tmp_path):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(os.environ, SYNTH_API_KEY="fixture", SYNTH_AUTH_FILE=str(tmp_path / "absent.json"))
    params = StdioServerParameters(
        command=sys.executable, args=["-c", "from synth_ai.mcp.server import main; main()"], env=env
    )
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            catalog = await session.list_tools()
            assert {t.name: t.inputSchema for t in catalog.tools} == {
                t["name"]: t["inputSchema"] for t in TOOLS
            }
            result = await session.call_tool("unknown_tool", {})
            assert result.isError
