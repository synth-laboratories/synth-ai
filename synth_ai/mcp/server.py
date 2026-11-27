"""Optional stdio MCP transport, sharing the hosted server's generated catalog."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx
    from mcp.server import Server
    from mcp.types import CallToolResult, TextContent, Tool

    from synth_ai._retry import RetryPolicy

from pydantic import BaseModel, JsonValue, ValidationError

from synth_ai._validation import safe_validation_detail
from synth_ai.aclient import AsyncSynthClient
from synth_ai.environments import Environment
from synth_ai.errors import (
    DirectoryUploadCancelledError,
    DirectoryUploadError,
    InvalidRequestError,
    SwarmsAPIError,
    SynthError,
    check_upload_format,
    local_problem,
)

from . import directory
from .dispatch import dispatch
from .tools import TOOLS, ToolSchema

__all__ = [
    "UPLOAD_OPERATIONS",
    "build_server",
    "call_tool",
    "main",
    "refusal",
    "serve",
    "validate_arguments",
]

UPLOAD_OPERATIONS = frozenset({"uploadFile", "createUpload"})


async def call_tool(
    client: AsyncSynthClient,
    name: str,
    arguments: dict[str, JsonValue],
    *,
    upload_descriptor: int | None = None,
) -> str:
    if name == directory.NAME and upload_descriptor is not None:
        return await directory.upload(client, upload_descriptor, arguments)
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


async def serve(
    *,
    environment: Environment | None = None,
    upload_root: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    retry_policy: RetryPolicy | None = None,
) -> None:
    """Serve the catalog on stdio.

    `http_client` lets a harness route the public calls through its own
    transport (for example a cached replay grant) while the stdio framing, tool
    validation and dispatch stay this server's own. `retry_policy` lets a harness
    that owns retries replace the SDK's per-operation loop; `None` keeps the default.
    """
    with directory.grant_root(upload_root) as descriptor:
        await _serve(
            environment=environment,
            upload_descriptor=descriptor,
            http_client=http_client,
            retry_policy=retry_policy,
        )


async def _serve(
    *,
    environment: Environment | None,
    upload_descriptor: int | None,
    http_client: httpx.AsyncClient | None = None,
    retry_policy: RetryPolicy | None = None,
) -> None:
    try:
        from mcp.server.stdio import stdio_server
    except ImportError as error:
        raise RuntimeError("Install synth-ai[mcp] to run the MCP server") from error
    async with AsyncSynthClient(
        environment=environment, http_client=http_client, retry_policy=retry_policy
    ) as client:
        server = build_server(client, upload_descriptor=upload_descriptor)
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())


def _invalid_arguments_detail(error: ValueError) -> str:
    raw = safe_validation_detail(error) if isinstance(error, ValidationError) else str(error)
    detail = " ".join(raw.split())
    return detail[:256] if detail else "invalid arguments"


def refusal(error: SwarmsAPIError) -> str:
    """The tool error text: the Problem, plus `retry_after_seconds` like the CLI's JSON.

    MCP has no response headers, so the server's Retry-After rides beside the
    Problem's own members instead of being dropped (SYN-5723).
    """
    if error.retry_after_seconds is None:
        return error.problem.model_dump_json()
    document = error.problem.model_dump(mode="json")
    document["retry_after_seconds"] = error.retry_after_seconds
    return json.dumps(document)


def build_server(client: AsyncSynthClient, *, upload_descriptor: int | None) -> Server:
    """The MCP server bound to an open client; the caller owns the client's lifetime."""
    try:
        from mcp.server import Server
        from mcp.types import CallToolResult, TextContent, Tool
    except ImportError as error:
        raise RuntimeError("Install synth-ai[mcp] to run the MCP server") from error
    server = Server("synth")

    async def list_tools() -> list[Tool]:
        catalog = [
            Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
            for t in TOOLS
        ]
        if upload_descriptor is not None:
            catalog.append(
                Tool(
                    name=directory.NAME,
                    description=directory.DESCRIPTION,
                    inputSchema=directory.SCHEMA,
                )
            )
        return catalog

    async def invoke(
        name: str, arguments: dict[str, JsonValue]
    ) -> list[TextContent] | CallToolResult:
        try:
            content = await call_tool(client, name, arguments, upload_descriptor=upload_descriptor)
        except (DirectoryUploadError, DirectoryUploadCancelledError) as error:
            return CallToolResult(
                isError=True, content=[TextContent(type="text", text=directory.failure(error))]
            )
        except SwarmsAPIError as error:
            return CallToolResult(
                isError=True, content=[TextContent(type="text", text=refusal(error))]
            )
        except SynthError as error:
            detail = " ".join(str(error).split())
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps({"code": error.code, "detail": detail}),
                    )
                ],
            )
        except ValueError as error:
            problem = local_problem(
                "invalid_arguments",
                "mcp_input",
                status=422,
                detail=_invalid_arguments_detail(error),
            )
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=problem.model_dump_json())],
            )
        return [TextContent(type="text", text=content)]

    server.list_tools()(list_tools)
    # call_tool validates input itself so refusals are Problems, not schema text.
    server.call_tool(validate_input=False)(invoke)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(prog="synth-ai-mcp")
    parser.add_argument("--env", type=Environment, choices=tuple(Environment))
    parser.add_argument(
        "--upload-root", help="Grant upload of exactly this absolute local directory"
    )
    options = parser.parse_args()
    asyncio.run(serve(environment=options.env, upload_root=options.upload_root))
