"""A hidden tool refuses with its own typed code, cause and request id (SYN-4354)."""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.mcp

pytest.importorskip("mcp.types", reason="requires the synth-ai[mcp] extra")

from mcp.types import CallToolRequest, CallToolRequestParams  # noqa: E402
from synth_ai.mcp.catalog import BY_NAME, ToolSpec  # noqa: E402
from synth_ai.mcp.server import create_server  # noqa: E402


class _NoTools:
    async def available_tools(self) -> frozenset[str]:
        return frozenset()

    async def unavailable_cause(self, tool: ToolSpec) -> str:
        return "capability:schema_mismatch"

    async def invoke(self, tool, arguments):  # pragma: no cover - never reached
        raise AssertionError("an unavailable tool must not dispatch")


def test_unavailable_tool_names_its_code_cause_and_request_id():
    server = create_server(_NoTools())
    handler = server.request_handlers[CallToolRequest]
    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name="index_search_list", arguments={"limit": 1}),
    )
    assert "index_search_list" in BY_NAME
    result = asyncio.run(handler(request)).root
    error = result.structuredContent["error"]
    assert result.isError
    assert error["code"] == "tool_unavailable"
    assert "capability:schema_mismatch" in error["message"]
    assert len(error["request_id"]) == 32
    assert error["outcome"] == "not_accepted"
