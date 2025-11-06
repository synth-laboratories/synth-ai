import asyncio
import os
from typing import Any

from synth_ai.cfgs import LocalDeployCfg
from synth_ai.mcp.setup import mcp_init_auth_session, mcp_poll_handshake
from synth_ai.uvicorn import deploy_app_uvicorn

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import ServerCapabilities, TextContent, Tool, ToolsCapability

server = Server("synth-ai")

os.environ["CTX"] = "mcp"

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
        ),
        Tool(
            name="deploy_local",
            description="Deploy a task app to a local server",
            inputSchema={
                "type": "object",
                "required": ["task_app_path"],
                "properties": {
                    "task_app_path": {
                        "type": "string",
                        "description": "Asbolute path to the task app Python file"
                    },
                    "env_api_key": {
                        "type": "string",
                        "description": "Use the ENVIRONMENT_API_KEY fetched via the setup function or supplied by user"
                    },
                    "trace": {
                        "type": "boolean",
                        "default": True
                    },
                    "host": {
                        "type": "string",
                        "default": "127.0.0.1"
                    },
                    "port": {
                        "type": "integer",
                        "default": 8000
                    }
                }
            }
        ),
    ]


@server.call_tool()
async def call_tool(
    name: str,
    arguments: dict[str, Any] | None
) -> list[TextContent]:
    args = arguments or {}
    match name:
        case "setup_start":
            return [TextContent(
                type="text",
                text=mcp_init_auth_session()
            )]
        case "setup_poll":
            device_code = args.get("device_code", '')
            return [TextContent(
                type="text",
                text=mcp_poll_handshake(device_code)
            )]
        case "deploy_local":
            try:
                cfg = LocalDeployCfg.create_from_dict(args)
            except Exception as exc:
                return [TextContent(
                    type="text",
                    text=f"{name} invalid configuration: {exc}"
                )]
            try:
                msg = deploy_app_uvicorn(cfg)
            except Exception as exc:
                msg = f"{name} failed: {exc}"
            return [TextContent(
                type="text",
                text=msg or f"{name} task app deployed"
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
