"""MCP discovery uses JSON contracts for every tool, without running local effects."""

import json

from synth_ai.mcp.catalog import TOOLS


def test_all_tool_definitions_are_json_contracts():
    names = [tool.name for tool in TOOLS]
    assert len(set(names)) == len(names), "Duplicate MCP tool names"
    assert names, "Missing MCP registry"
    for tool in TOOLS:
        schema = tool.input_schema()
        assert schema.get("type") == "object", tool.name
        json.dumps(schema)
        json.dumps(tool.output_schema())
