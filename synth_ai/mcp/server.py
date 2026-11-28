"""Optional stdio MCP transport, sharing the hosted server's generated catalog."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import TYPE_CHECKING

from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterable

    import httpx
    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
    from mcp.server import NotificationOptions, Server
    from mcp.server.lowlevel.helper_types import ReadResourceContents
    from mcp.server.models import InitializationOptions
    from mcp.shared.message import SessionMessage
    from mcp.types import (
        CallToolResult,
        CreateTaskResult,
        LoggingLevel,
        Resource,
        TaskMetadata,
        TextContent,
        Tool,
    )
    from pydantic import AnyUrl

    from synth_ai._retry import RetryPolicy

from pydantic import BaseModel, JsonValue, ValidationError

from synth_ai._limits import SERVER_WAIT_SECONDS_DEFAULT
from synth_ai._validation import safe_validation_detail
from synth_ai._version import __version__
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

from . import directory, events, skills
from ._arguments import required
from .dispatch import dispatch
from .elicitation import client_supports_elicitation
from .progress import wait_run_with_progress
from .tasks import (
    TASK_SUPPORTED_TOOLS,
    call_tool_as_task,
    cancel_task,
    get_task,
    get_task_result,
)
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


def resolve_and_validate(
    name: str,
    arguments: dict[str, JsonValue],
) -> tuple[str, ToolSchema]:
    tool = next((tool for tool in TOOLS if tool["name"] == name), None)
    if tool is None:
        raise ValueError("Unknown tool")
    schema = tool["inputSchema"]
    unexpected = sorted(set(arguments) - set(schema["properties"]))
    missing = sorted(set(schema["required"]) - set(arguments))
    if unexpected or missing:
        parts: list[str] = []
        if missing:
            parts.append("missing required argument(s): " + ", ".join(missing))
        if unexpected:
            parts.append("unexpected argument(s): " + ", ".join(unexpected))
        raise ValueError("; ".join(parts))
    validate_arguments(tool["operationId"], schema, arguments)
    return tool["operationId"], schema


async def call_tool(
    client: AsyncSynthClient,
    name: str,
    arguments: dict[str, JsonValue],
    *,
    upload_descriptor: int | None = None,
) -> str:
    if name == directory.NAME and upload_descriptor is not None:
        return await directory.upload(client, upload_descriptor, arguments)
    operation_id, _ = resolve_and_validate(name, arguments)
    result = await dispatch(client, operation_id, arguments)
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
        detail = f"{error.json_path}: {error.message}" if error.json_path != "$" else error.message
        raise InvalidRequestError(
            local_problem("tool_arguments_invalid", "mcp_input", status=422, detail=detail)
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
        import anyio
        from mcp.server import Server
        from mcp.server.lowlevel.helper_types import ReadResourceContents
        from mcp.server.models import InitializationOptions
        from mcp.shared.exceptions import McpError
        from mcp.shared.message import SessionMessage
        from mcp.types import (
            CallToolRequest,
            CallToolResult,
            CancelTaskRequest,
            CreateTaskResult,
            ErrorData,
            GetTaskPayloadRequest,
            GetTaskRequest,
            JSONRPCError,
            JSONRPCMessage,
            JSONRPCNotification,
            JSONRPCRequest,
            JSONRPCResponse,
            Resource,
            ServerCapabilities,
            ServerResult,
            ServerTasksCapability,
            ServerTasksRequestsCapability,
            TasksCallCapability,
            TasksCancelCapability,
            TasksToolsCapability,
            TextContent,
            Tool,
            ToolExecution,
        )
        from pydantic import AnyUrl
    except ImportError as error:
        raise RuntimeError("Install synth-ai[mcp] to run the MCP server") from error

    class SynthMCPServer(Server):
        @override
        def get_capabilities(
            self,
            notification_options: NotificationOptions,
            experimental_capabilities: dict[str, dict[str, object]],
        ) -> ServerCapabilities:
            capabilities = super().get_capabilities(notification_options, experimental_capabilities)
            capabilities.tasks = ServerTasksCapability(
                cancel=TasksCancelCapability(),
                requests=ServerTasksRequestsCapability(
                    tools=TasksToolsCapability(call=TasksCallCapability())
                ),
            )
            return capabilities

        @override
        def create_initialization_options(
            self,
            notification_options: NotificationOptions | None = None,
            experimental_capabilities: dict[str, dict[str, object]] | None = None,
        ) -> InitializationOptions:
            base = super().create_initialization_options(
                notification_options=notification_options,
                experimental_capabilities=experimental_capabilities,
            )
            dumped = base.capabilities.model_dump(by_alias=True, mode="json", exclude_none=True)
            dumped["extensions"] = {
                **skills.EXTENSION_CAPABILITIES,
                **events.EXTENSION_CAPABILITIES,
            }
            return InitializationOptions(
                server_name=base.server_name,
                server_version=base.server_version,
                capabilities=ServerCapabilities.model_validate(dumped),
                instructions=base.instructions,
                website_url=base.website_url,
                icons=base.icons,
            )

        @override
        async def run(
            self,
            read_stream: MemoryObjectReceiveStream[SessionMessage | Exception],
            write_stream: MemoryObjectSendStream[SessionMessage],
            initialization_options: InitializationOptions,
            raise_exceptions: bool = False,
            stateless: bool = False,
        ) -> None:
            forwarded_send, forwarded_recv = anyio.create_memory_object_stream[
                SessionMessage | Exception
            ](64)

            async def _handle_events_poll(req: JSONRPCRequest) -> None:
                raw_params = dict(req.params) if isinstance(req.params, dict) else None
                reply: JSONRPCResponse | JSONRPCError
                try:
                    payload = await events.poll_events(client, raw_params)
                    reply = JSONRPCResponse(jsonrpc="2.0", id=req.id, result=payload)
                except events.EventError as error:
                    reply = JSONRPCError(
                        jsonrpc="2.0",
                        id=req.id,
                        error=ErrorData(code=error.code, message=error.message, data=error.data),
                    )
                await write_stream.send(SessionMessage(message=JSONRPCMessage(reply)))

            async def _handle_events_stream(req: JSONRPCRequest) -> None:
                raw_params = dict(req.params) if isinstance(req.params, dict) else None

                async def _emit(event_params: dict[str, JsonValue]) -> None:
                    note = JSONRPCNotification(
                        jsonrpc="2.0",
                        method="notifications/events/event",
                        params=event_params,
                    )
                    await write_stream.send(SessionMessage(message=JSONRPCMessage(note)))

                reply: JSONRPCResponse | JSONRPCError
                try:
                    payload = await events.stream_events(client, raw_params, _emit)
                    reply = JSONRPCResponse(jsonrpc="2.0", id=req.id, result=payload)
                except events.EventError as error:
                    reply = JSONRPCError(
                        jsonrpc="2.0",
                        id=req.id,
                        error=ErrorData(code=error.code, message=error.message, data=error.data),
                    )
                await write_stream.send(SessionMessage(message=JSONRPCMessage(reply)))

            async def _route_incoming() -> None:
                async with read_stream, forwarded_send:
                    async for message in read_stream:
                        if not isinstance(message, SessionMessage):
                            await forwarded_send.send(message)
                            continue
                        root = message.message.root
                        if not isinstance(root, JSONRPCRequest):
                            await forwarded_send.send(message)
                            continue
                        if root.method in skills.EXTENSION_METHODS:
                            raw_params = (
                                dict(root.params) if isinstance(root.params, dict) else None
                            )
                            payload, err = skills.dispatch_extension_request(
                                root.method, raw_params
                            )
                            reply: JSONRPCResponse | JSONRPCError
                            if err is not None:
                                reply = JSONRPCError(
                                    jsonrpc="2.0",
                                    id=root.id,
                                    error=ErrorData(
                                        code=err.code, message=err.message, data=err.data
                                    ),
                                )
                            else:
                                assert payload is not None
                                reply = JSONRPCResponse(jsonrpc="2.0", id=root.id, result=payload)
                            await write_stream.send(SessionMessage(message=JSONRPCMessage(reply)))
                            continue
                        if root.method in events.EXTENSION_METHODS:
                            if root.method == "events/list":
                                list_reply = JSONRPCResponse(
                                    jsonrpc="2.0",
                                    id=root.id,
                                    result=events.list_events_payload(),
                                )
                                await write_stream.send(
                                    SessionMessage(message=JSONRPCMessage(list_reply))
                                )
                            elif root.method == "events/poll":
                                task_group.start_soon(_handle_events_poll, root)
                            else:
                                task_group.start_soon(_handle_events_stream, root)
                            continue
                        if root.method == "resources/read":
                            raw_uri = (
                                root.params.get("uri") if isinstance(root.params, dict) else None
                            )
                            if (
                                not isinstance(raw_uri, str)
                                or skills.parse_skill_uri(raw_uri, want_dir=False) is None
                            ):
                                err_reply = JSONRPCError(
                                    jsonrpc="2.0",
                                    id=root.id,
                                    error=ErrorData(
                                        code=skills.INVALID_PARAMS,
                                        message="Invalid params: malformed or disallowed resource URI.",
                                        data={"uri": raw_uri} if isinstance(raw_uri, str) else None,
                                    ),
                                )
                                await write_stream.send(
                                    SessionMessage(message=JSONRPCMessage(err_reply))
                                )
                                continue
                        await forwarded_send.send(message)

            async with anyio.create_task_group() as task_group:
                task_group.start_soon(_route_incoming)
                await super().run(
                    forwarded_recv,
                    write_stream,
                    initialization_options,
                    raise_exceptions=raise_exceptions,
                    stateless=stateless,
                )
                task_group.cancel_scope.cancel()

    server = SynthMCPServer("synth", version=__version__)
    catalog = skills.load_skills()
    log_level_state: dict[str, LoggingLevel | None] = {"minimum": None}

    async def set_logging_level(level: LoggingLevel) -> None:
        log_level_state["minimum"] = level

    async def list_tools() -> list[Tool]:
        catalog_tools = [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
                execution=ToolExecution(taskSupport="optional")
                if t["name"] in TASK_SUPPORTED_TOOLS
                else None,
            )
            for t in TOOLS
        ]
        if upload_descriptor is not None:
            catalog_tools.append(
                Tool(
                    name=directory.NAME,
                    description=directory.DESCRIPTION,
                    inputSchema=directory.SCHEMA,
                )
            )
        return catalog_tools

    async def list_resources() -> list[Resource]:
        return [
            Resource(
                uri=AnyUrl(entry.uri),
                name=entry.name,
                description=entry.description,
                mimeType=entry.mime_type,
                size=entry.size,
            )
            for entry in catalog.resources
        ]

    async def read_resource(uri: AnyUrl) -> Iterable[ReadResourceContents]:
        try:
            entry = skills.read_resource_entry(str(uri))
        except skills.SkillError as error:
            raise McpError(
                ErrorData(code=error.code, message=error.message, data=error.data)
            ) from None
        return [ReadResourceContents(content=entry.content, mime_type=entry.mime_type)]

    async def invoke(
        name: str,
        arguments: dict[str, JsonValue],
        *,
        task_meta: TaskMetadata | None = None,
    ) -> list[TextContent] | CallToolResult | CreateTaskResult:
        try:
            ctx = server.request_context
            if task_meta is not None:
                resolve_and_validate(name, arguments)
                return await call_tool_as_task(client, name, arguments, task_meta)

            progress_token = ctx.meta.progressToken if ctx.meta is not None else None
            if name == "swarms_run_wait" and (
                progress_token is not None or client_supports_elicitation(ctx.session)
            ):
                resolve_and_validate(name, arguments)
                wait_seconds = (
                    required(arguments, "timeout_seconds", int)
                    if "timeout_seconds" in arguments
                    else SERVER_WAIT_SECONDS_DEFAULT
                )
                waited = await wait_run_with_progress(
                    client,
                    ctx.session,
                    required(arguments, "run_id", str),
                    timeout_seconds=wait_seconds,
                    progress_token=progress_token,
                    related_request_id=ctx.request_id,
                    minimum_log_level=log_level_state["minimum"],
                )
                content = waited.model_dump_json(exclude_unset=True)
            else:
                content = await call_tool(
                    client, name, arguments, upload_descriptor=upload_descriptor
                )
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

    async def handle_call_tool(req: CallToolRequest) -> ServerResult:
        arguments: dict[str, JsonValue] = (
            dict(req.params.arguments) if req.params.arguments is not None else {}
        )
        result = await invoke(req.params.name, arguments, task_meta=req.params.task)
        if isinstance(result, (CallToolResult, CreateTaskResult)):
            return ServerResult(result)
        return ServerResult(CallToolResult(content=list(result), isError=False))

    async def handle_get_task(req: GetTaskRequest) -> ServerResult:
        return ServerResult(await get_task(client, req.params.taskId))

    async def handle_get_task_result(req: GetTaskPayloadRequest) -> ServerResult:
        return ServerResult(await get_task_result(client, req.params.taskId))

    async def handle_cancel_task(req: CancelTaskRequest) -> ServerResult:
        return ServerResult(await cancel_task(client, req.params.taskId))

    server.set_logging_level()(set_logging_level)
    server.list_tools()(list_tools)
    server.list_resources()(list_resources)
    server.read_resource()(read_resource)
    # call_tool validates input itself so refusals are Problems, not schema text.
    server.request_handlers[CallToolRequest] = handle_call_tool
    server.request_handlers[GetTaskRequest] = handle_get_task
    server.request_handlers[GetTaskPayloadRequest] = handle_get_task_result
    server.request_handlers[CancelTaskRequest] = handle_cancel_task
    return server


def main() -> None:
    parser = argparse.ArgumentParser(prog="synth-ai-mcp")
    parser.add_argument("--env", type=Environment, choices=tuple(Environment))
    parser.add_argument(
        "--upload-root", help="Grant upload of exactly this absolute local directory"
    )
    options = parser.parse_args()
    asyncio.run(serve(environment=options.env, upload_root=options.upload_root))
