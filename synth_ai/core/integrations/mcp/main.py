import asyncio
import os
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import ServerCapabilities, TextContent, Tool, ToolsCapability

from synth_ai.core.cfgs import LocalDeployCfg, ModalDeployCfg
from synth_ai.core.integrations.mcp.setup import setup_fetch, setup_start
from synth_ai.core.integrations.modal import deploy_app_modal
from synth_ai.core.uvicorn import deploy_app_uvicorn

server = Server("synth-ai")

os.environ["CTX"] = "mcp"


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="deploy_local",
            description="Deploy a task app to a local server",
            inputSchema={
                "type": "object",
                "required": [
                    "task_app_path",
                    "env_api_key"
                ],
                "properties": {
                    "task_app_path": {
                        "type": "string",
                        "description": "Absolute path to the task app Python file"
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
        Tool(
            name="deploy_modal",
            description="Deploy a task app to Modal",
            inputSchema={
                "type": "object",
                "required": [
                    "task_app_path",
                    "modal_app_path",
                    "synth_api_key",
                    "env_api_key"
                ],
                "properties": {
                    "task_app_path": {
                        "type": "string",
                        "description": "Absolute path to the task app Python file"
                    },
                    "modal_app_path": {
                        "type": "string",
                        "description": "Absolute path to the Modal app Python file"
                    },
                    "synth_api_key": {
                        "type": "string",
                        "description": "SYNTH_API_KEY for authentication"
                    },
                    "env_api_key": {
                        "type": "string",
                        "description": "ENVIRONMENT_API_KEY used to access the task app"
                    },
                    "modal_bin_path": {
                        "type": "string",
                        "description": "Optional path to the Modal CLI binary"
                    },
                    "cmd_arg": {
                        "type": "string",
                        "enum": ["deploy", "serve"],
                        "default": "deploy",
                        "description": "Modal command to run"
                    },
                    "task_app_name": {
                        "type": "string",
                        "description": "Optional Modal app name override"
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Print the Modal command without executing it"
                    }
                }
            }
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
            name="setup_start",
            description=(
                "Begin the Synth device-code authentication flow. Returns JSON containing "
                "`origin`, `verification_uri`, `device_code`, `expires_at`, `expires_in`, and "
                "`poll_interval` so the caller can instruct the user and schedule polling."
            ),
            inputSchema={"type": "object"}
        ),
        Tool(
            name="create_rl_task_app",
            description="Instructions on how to create a reinforcement learning task app, required for all deployments",
            inputSchema={},
        ),
        Tool(
            name="create_sft_task_app",
            description="Instructions on how to create a supervised fine-tuning task app, required for all deployments",
            inputSchema={},
        )
    ]


@server.call_tool()
async def call_tool(
    name: str,
    arguments: dict[str, Any] | None
) -> list[TextContent]:
    args = arguments or {}
    match name:
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
        case "setup_poll":
            device_code = args.get("device_code", '')
            return [TextContent(
                type="text",
                text=setup_fetch(device_code)
            )]
        case "setup_start":
            return [TextContent(
                type="text",
                text=setup_start()
            )]
        case "deploy_modal":
            missing: list[str] = []
            synth_api_key = str(args.get("synth_api_key")).strip()
            if not synth_api_key:
                missing.append("synth_api_key")
            env_api_key = str(args.get("env_api_key")).strip()
            if not env_api_key:
                missing.append("env_api_key")
            task_app_path_raw = args.get("task_app_path")
            if not task_app_path_raw:
                missing.append("task_app_path")
            modal_app_path_raw = args.get("modal_app_path")
            if not modal_app_path_raw:
                missing.append("modal_app_path")
            if len(missing) > 0:
                return [TextContent(
                    type="text",
                    text=f"{name} missing args: {missing}"
                )]
            assert synth_api_key is not None
            assert env_api_key is not None
            assert task_app_path_raw is not None
            assert modal_app_path_raw is not None
            cfg_kwargs: dict[str, Any] = {
                "modal_app_path": Path(modal_app_path_raw),
                "cmd_arg": args.get("cmd_arg"),
                "task_app_name": args.get("task_app_name"),
                "dry_run": bool(args.get("dry_run", False)),
            }
            if args.get("modal_bin_path"):
                cfg_kwargs["modal_bin_path"] = Path(args["modal_bin_path"])
            try:
                cfg = ModalDeployCfg.create_from_kwargs(
                    task_app_path=Path(task_app_path_raw),
                    synth_api_key=synth_api_key,
                    env_api_key=env_api_key,
                    **cfg_kwargs
                )
            except Exception as exc:
                return [TextContent(
                    type="text",
                    text=f"{name} invalid configuration: {exc}"
                )]
            try:
                msg = deploy_app_modal(cfg)
            except Exception as exc:
                msg = f"{name} failed: {exc}"
            return [TextContent(
                type="text",
                text=msg or f"{name} task app deployed"
            )]
        case "create_rl_task_app":
            return [TextContent(
                type="text",
                text="Instructions for creating an RL task app are at Synth AI's official docs: https://docs.usesynth.ai/task-app/task-app-rl",
            )]
        case "create_sft_task_app":
            return [TextContent(
                type="text",
                text="Instructions for creating an SFT task app are at Synth AI's official docs: https://docs.usesynth.ai/task-app/task-app-sft",
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
