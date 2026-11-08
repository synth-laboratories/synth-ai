"""Optional stdio MCP transport, sharing the hosted server's generated catalog."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.types import CallToolResult, TextContent, Tool

from pydantic import BaseModel, JsonValue, TypeAdapter

from .._generated.operations import OPERATIONS
from ..aclient import AsyncSynthClient
from ..errors import SwarmsAPIError
from .tools import TOOLS


async def call_tool(client: AsyncSynthClient, name: str, arguments: dict[str, JsonValue]) -> str:
    tool = next((tool for tool in TOOLS if tool["name"] == name), None)
    if tool is None:
        raise ValueError("Unknown tool")
    schema = tool["inputSchema"]
    if set(arguments) - set(schema["properties"]) or set(schema["required"]) - set(arguments):
        raise ValueError("Tool arguments do not match the contract")
    op = OPERATIONS[tool["operationId"]]
    path = TypeAdapter(dict[str, str]).validate_python(
        {key: arguments[key] for key in op["path_params"]}, strict=True
    )
    query = TypeAdapter(dict[str, str | int | float | bool | None]).validate_python(
        {key: arguments[key] for key in op["query_params"] if key in arguments}, strict=True
    )
    body = TypeAdapter(dict[str, JsonValue] | None).validate_python(
        arguments.get("body"), strict=True
    )
    idempotency_key = TypeAdapter(str | None).validate_python(
        arguments.get("idempotency_key"), strict=True
    )
    result = await client._transport.invoke(
        tool["operationId"],
        path=path,
        query=query,
        body=body,
        idempotency_key=idempotency_key,
    )
    if isinstance(result, BaseModel):
        return result.model_dump_json(exclude_unset=True)
    raise ValueError("Unexpected tool response")


async def serve() -> None:
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import CallToolResult, TextContent, Tool
    except ImportError as error:
        raise RuntimeError("Install synth-ai[mcp] to run the MCP server") from error
    server = Server("synth")
    async with AsyncSynthClient() as client:

        async def list_tools() -> list[Tool]:
            return [
                Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
                for t in TOOLS
            ]

        async def invoke(
            name: str, arguments: dict[str, JsonValue]
        ) -> list[TextContent] | CallToolResult:
            try:
                content = await call_tool(client, name, arguments)
            except SwarmsAPIError as error:
                return CallToolResult(
                    isError=True,
                    content=[TextContent(type="text", text=error.problem.model_dump_json())],
                )
            return [TextContent(type="text", text=content)]

        server.list_tools()(list_tools)
        server.call_tool()(invoke)
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())


def main() -> None:
    asyncio.run(serve())
