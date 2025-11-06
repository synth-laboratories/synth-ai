import asyncio
from typing import Any

from synth_ai.mcp.setup import poll_handshake, start_handshake

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import ServerCapabilities, TextContent, Tool, ToolsCapability

server = Server("synth-ai")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="setup_start",
            description=(
                "Begin the Synth device-code authentication flow. Returns JSON containing "
                "`origin`, `verification_uri`, `device_code`, `expires_at`, `expires_in`, and "
                "`poll_interval` so the caller can instruct the user and schedule polling."
            ),
            inputSchema={"type": "object"}
        ),
        Tool(
            name="setup_poll",
            description=(
                "Perform a single poll of the Synth authentication handshake using the provided "
                "`device_code`. Responds with JSON: `status` ('pending' | 'success' | 'error'); "
                "`keys` on success (masked by default); `message` and `code` for errors."
            ),
            inputSchema={
                "type": "object",
                "required": ["device_code"],
                "properties": {"device_code": {"type": "string"}}
            }
        )
    ]


@server.call_tool()
async def call_tool(
    name: str,
    arguments: dict[str, Any] | None
) -> list[TextContent]:
    args = arguments or {}
    if name == "setup_start":
        return [TextContent(
            type="text",
            text=start_handshake()
        )]
    if name == "setup_poll":
        device_code = args.get("device_code", '')
        return [TextContent(
            type="text",
            text=poll_handshake(device_code)
        )]
    return [TextContent(
        type="text",
        text=f"Unknown tool '{name}'"
    )]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        init_options = InitializationOptions(
            server_name="synth-ai",
            server_version="0.1.0",
            capabilities=ServerCapabilities(tools=ToolsCapability())
        )
        await server.run(read_stream, write_stream, init_options)


if __name__ == "__main__":
    asyncio.run(main())
