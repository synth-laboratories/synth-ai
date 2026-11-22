import os
import sys

import pytest

from synth_ai.mcp.tools import TOOLS


@pytest.mark.parametrize("grant", [False, True])
async def test_stdio_catalog_and_error_signal(tmp_path, grant):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(os.environ, SYNTH_API_KEY="fixture", SYNTH_AUTH_FILE=str(tmp_path / "absent.json"))
    args = ["-c", "from synth_ai.mcp.server import main; main()"]
    if grant:
        args += ["--upload-root", str(tmp_path)]
    params = StdioServerParameters(command=sys.executable, args=args, env=env)
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            catalog = await session.list_tools()
            expected = {t["name"]: t["inputSchema"] for t in TOOLS}
            if grant:
                from synth_ai.mcp.directory import NAME, SCHEMA

                expected[NAME] = SCHEMA
                refused = await session.call_tool(NAME, {"idempotency_key": ""})
                assert refused.isError
                assert "tool_arguments_invalid" in refused.content[0].text
            assert {t.name: t.inputSchema for t in catalog.tools} == expected
            result = await session.call_tool("unknown_tool", {})
            assert result.isError
