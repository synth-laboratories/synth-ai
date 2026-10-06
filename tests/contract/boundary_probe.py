"""Exercise the real public MCP gate over one effect-counting tool definition.

See testing/specifications/sdk/core_research_migration.md, closed MCP arguments.
"""

from synth_ai.mcp.research.registry import call_tool
from synth_ai.mcp.research.server import ResearchMcpServer


def invoke_tool(tool, arguments, entrypoint):
    if entrypoint == "registry":
        return call_tool({tool.name: tool}, tool.name, arguments)
    server = ResearchMcpServer.__new__(ResearchMcpServer)
    server._advertised_tools = lambda: {tool.name: tool}
    return server.call_tool(tool.name, arguments)
