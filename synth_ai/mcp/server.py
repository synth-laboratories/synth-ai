"""Optional stdio MCP transport, sharing the hosted server's generated catalog."""

from __future__ import annotations

import argparse
import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.types import CallToolResult, TextContent, Tool

from pydantic import BaseModel, JsonValue

from ..aclient import AsyncSynthClient
from ..environments import Environment
from ..errors import InvalidRequestError, SwarmsAPIError, check_upload_format, local_problem
from .dispatch import dispatch
from .tools import TOOLS, ToolSchema

UPLOAD_OPERATIONS = frozenset({"uploadFile", "createUpload"})


async def call_tool(client: AsyncSynthClient, name: str, arguments: dict[str, JsonValue]) -> str:
    tool = next((tool for tool in TOOLS if tool["name"] == name), None)
    if tool is None:
        raise ValueError("Unknown tool")
    schema = tool["inputSchema"]
    if set(arguments) - set(schema["properties"]) or set(schema["required"]) - set(arguments):
        raise ValueError("Tool arguments do not match the contract")
    validate_arguments(tool["operationId"], schema, arguments)
    result = await dispatch(client, tool["operationId"], arguments)
    if result is None:
        return "null"  # The public operation has no representation or a nullable result.
    if isinstance(result, BaseModel):
        return result.model_dump_json(exclude_unset=True)
    raise ValueError("Unexpected tool response")


def validate_arguments(
    operation_id: str, schema: ToolSchema, arguments: dict[str, JsonValue]
) -> None:
    """Refuse invalid tool input with a typed Problem instead of a schema message.

    An upload in a format Core does not accept is `upload_format_unsupported`
    (422), exactly as the server would answer; any other schema violation is
    `tool_arguments_invalid` (422).
    """
    from jsonschema import Draft202012Validator

    body = arguments.get("body")
    if operation_id in UPLOAD_OPERATIONS and isinstance(body, dict):
        media_type = body.get("media_type")
        check_upload_format(media_type if isinstance(media_type, str) else None)
    error = next(iter(Draft202012Validator(schema).iter_errors(arguments)), None)
    if error is not None:
        raise InvalidRequestError(
            local_problem("tool_arguments_invalid", "mcp_input", status=422, detail=error.message)
        )


async def serve(*, environment: Environment | None = None) -> None:
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import CallToolResult, TextContent, Tool
    except ImportError as error:
        raise RuntimeError("Install synth-ai[mcp] to run the MCP server") from error
    server = Server("synth")
    async with AsyncSynthClient(environment=environment) as client:

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
        # call_tool validates input itself so refusals are Problems, not schema text.
        server.call_tool(validate_input=False)(invoke)
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser(prog="synth-ai-mcp")
    parser.add_argument("--env", type=Environment, choices=tuple(Environment))
    options = parser.parse_args()
    asyncio.run(serve(environment=options.env))
