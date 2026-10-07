"""MCP launch inputs must express the mandatory launch contract."""

import pytest
from synth_ai.mcp.research.server import ResearchMcpServer


@pytest.mark.parametrize(
    "name",
    [
        "research_trigger_run",
        "research_start_run",
        "research_start_one_off_run",
        "research_start_run_in_dev_environment",
    ],
)
@pytest.mark.parametrize("field", ["deployment_pins", "provenance_mode"])
def test_launch_schema_accepts_provenance__RW01_PR12(name, field):
    server = ResearchMcpServer(
        api_key="offline-dummy", backend_base="http://offline.invalid", include_advanced_tools=True
    )
    tool = server.get_tool_definition(name)
    assert tool is not None, f"RW-01: missing launch tool {name}"
    assert field in tool.input_schema["properties"], f"RW-01/PR-12: {name} forbids required {field}"
