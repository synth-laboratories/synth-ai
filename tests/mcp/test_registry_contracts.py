"""MCP discovery uses JSON contracts for every tool, without running local effects."""

import json

from synth_ai.mcp.research.server import ResearchMcpServer


def test_all_tool_definitions_are_json_contracts():
    server = ResearchMcpServer(
        api_key="offline-dummy", backend_base="http://offline.invalid", include_advanced_tools=True
    )
    definitions = server.tool_definitions()
    names = [definition.name for definition in definitions]
    assert len(set(names)) == len(names), "Duplicate MCP tool names"
    assert definitions, "Missing MCP registry"
    for definition in definitions:
        assert definition.input_schema.get("type") == "object", definition.name
        json.dumps(definition.input_schema)
