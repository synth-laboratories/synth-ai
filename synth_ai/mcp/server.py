"""Optional stdio MCP transport, sharing the hosted server's generated catalog."""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

from pydantic import BaseModel

from .._generated.operations import OPERATIONS
from ..aclient import AsyncSynthClient
from ..errors import SwarmsAPIError
from .tools import TOOLS


async def call_tool(client: AsyncSynthClient, name: str, arguments: dict[str, Any]) -> str:
    tool = next((tool for tool in TOOLS if tool["name"] == name), None)
    if tool is None:
        raise ValueError("Unknown tool")
    schema = tool["inputSchema"]
    if set(arguments) - set(schema["properties"]) or set(schema["required"]) - set(arguments):
        raise ValueError("Tool arguments do not match the contract")
    op = OPERATIONS[tool["operationId"]]
    result = await client._transport.invoke(
        tool["operationId"],
        path={key: arguments[key] for key in op["path_params"]},
        query={key: arguments[key] for key in op["query_params"] if key in arguments},
        body=arguments.get("body"),
        idempotency_key=arguments.get("idempotency_key"),
    )
    if isinstance(result, BaseModel):
        return result.model_dump_json(exclude_unset=True)
    raise ValueError("Unexpected tool response")


async def serve() -> None:
    try:
        server_module = importlib.import_module("mcp.server")
        stdio = importlib.import_module("mcp.server.stdio")
        types = importlib.import_module("mcp.types")
    except ImportError as error:
        raise RuntimeError("Install synth-ai[mcp] to run the MCP server") from error
    server = server_module.Server("synth")
    async with AsyncSynthClient() as client:

        async def list_tools() -> list[Any]:
            return [
                types.Tool(
                    name=t["name"], description=t["description"], inputSchema=t["inputSchema"]
                )
                for t in TOOLS
            ]

        async def invoke(name: str, arguments: dict[str, Any]) -> Any:
            try:
                content = await call_tool(client, name, arguments)
            except SwarmsAPIError as error:
                return types.CallToolResult(
                    isError=True,
                    content=[types.TextContent(type="text", text=error.problem.model_dump_json())],
                )
            return [types.TextContent(type="text", text=content)]

        server.list_tools()(list_tools)
        server.call_tool()(invoke)
        async with stdio.stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())


def main() -> None:
    asyncio.run(serve())
