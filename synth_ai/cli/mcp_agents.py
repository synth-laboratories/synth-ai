from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

import click

from synth_ai.types import MODEL_NAMES, ModelName
from synth_ai.utils.agent_launchers import (
    AgentPreparationError,
    prepare_agent_invocation,
)
from synth_ai.utils.agents import (
    AgentGuide,
    get_agent_guides,
    render_agents_markdown,
)


CATALOG_RESOURCE_URI = "synth://agents/catalog"
AGENT_RESOURCE_URI_TEMPLATE = "synth://agents/{agent_id}"

JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INTERNAL_ERROR = -32603


@dataclass(frozen=True, slots=True)
class ToolSchema:
    description: str
    input_schema: Mapping[str, Any]


def _tool_schema_for_agent(guide: AgentGuide) -> ToolSchema:
    description = guide.summary
    properties: dict[str, Any] = {
        "force": {
            "type": "boolean",
            "description": "Prompt for credentials even if cached values exist.",
            "default": False,
        }
    }
    required: list[str] = []

    if guide.id in {"claude", "codex", "opencode"}:
        properties["model_name"] = {
            "type": "string",
            "description": "Synth model slug (see `synth-ai models list`).",
            "enum": sorted(MODEL_NAMES),
        }
    properties["override_url"] = {
        "type": "string",
        "description": "Optional override endpoint in place of Synth defaults.",
    }

    return ToolSchema(
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    )


class SimpleMCPServer:
    """Minimal JSON-RPC 2.0 server using MCP method names over stdio."""

    def __init__(self) -> None:
        self._request_handlers: Dict[str, Callable[[dict[str, Any]], Any]] = {}
        self._notification_handlers: Dict[str, Callable[[dict[str, Any]], None]] = {}

    def request(self, method: str) -> Callable[[Callable[[dict[str, Any]], Any]], Callable[[dict[str, Any]], Any]]:
        def decorator(fn: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
            self._request_handlers[method] = fn
            return fn

        return decorator

    def notification(self, method: str) -> Callable[[Callable[[dict[str, Any]], None]], Callable[[dict[str, Any]], None]]:
        def decorator(fn: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], None]:
            self._notification_handlers[method] = fn
            return fn

        return decorator

    def serve_stdio(self) -> None:
        stdin = sys.stdin.buffer
        while True:
            message = self._read_message(stdin)
            if message is None:
                break
            self._handle_message(message)

    def _handle_message(self, message: Any) -> None:
        if not isinstance(message, dict):
            self._write_error(None, JSONRPC_INVALID_REQUEST, "Invalid request payload")
            return

        method = message.get("method")
        params = message.get("params") or {}
        message_id = message.get("id")

        if method is None:
            # Ignore responses or malformed payloads
            return

        if message_id is None:
            handler = self._notification_handlers.get(method)
            if handler:
                try:
                    handler(params)
                except Exception:
                    # Notifications do not receive error responses
                    pass
            return

        handler = self._request_handlers.get(method)
        if handler is None:
            self._write_error(message_id, JSONRPC_METHOD_NOT_FOUND, f"Method not found: {method}")
            return

        try:
            result = handler(params)
        except MCPError as exc:
            self._write_error(message_id, exc.code, exc.message, exc.data)
        except Exception as exc:
            self._write_error(message_id, JSONRPC_INTERNAL_ERROR, str(exc))
        else:
            self._write_response(message_id, result)

    def _read_message(self, stdin: Any) -> Any | None:
        headers: dict[str, str] = {}
        while True:
            line = stdin.readline()
            if not line:
                return None
            stripped = line.strip()
            if not stripped:
                break
            try:
                key, value = stripped.decode("ascii").split(":", 1)
            except ValueError:
                continue
            headers[key.lower()] = value.strip()

        content_length = int(headers.get("content-length", "0") or "0")
        if content_length <= 0:
            return None
        body = stdin.read(content_length)
        if not body:
            return None
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._write_error(None, JSONRPC_PARSE_ERROR, "Failed to decode JSON payload")
            return None

    def _write_response(self, message_id: Any, result: Any) -> None:
        payload = json.dumps({"jsonrpc": "2.0", "id": message_id, "result": result})
        self._write_stdout(payload)

    def _write_error(
        self,
        message_id: Any,
        code: int,
        message: str,
        data: Any | None = None,
    ) -> None:
        error_payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": message_id,
            "error": {
                "code": code,
                "message": message,
            },
        }
        if data is not None:
            error_payload["error"]["data"] = data
        self._write_stdout(json.dumps(error_payload))

    @staticmethod
    def _write_stdout(payload: str) -> None:
        encoded = payload.encode("utf-8")
        sys.stdout.buffer.write(f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii"))
        sys.stdout.buffer.write(encoded)
        sys.stdout.buffer.flush()


class MCPError(RuntimeError):
    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def _run_process(command: list[str], env: Mapping[str, str], cwd: str | None) -> tuple[str, int]:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(env),
        cwd=cwd,
        check=False,
    )
    output = result.stdout or ""
    return output, result.returncode


def _parse_arguments(arguments: Mapping[str, Any]) -> tuple[ModelName | None, bool, str | None]:
    model_arg = arguments.get("model_name")
    model_name: ModelName | None = model_arg if model_arg else None
    force = bool(arguments.get("force", False))
    override_url = arguments.get("override_url")
    if override_url is not None and not isinstance(override_url, str):
        raise MCPError(JSONRPC_INVALID_REQUEST, "override_url must be a string when provided")
    return model_name, force, override_url


def build_server() -> SimpleMCPServer:
    server = SimpleMCPServer()
    guide_map = {guide.id: guide for guide in get_agent_guides()}
    tool_schemas = {gid: _tool_schema_for_agent(guide) for gid, guide in guide_map.items()}

    @server.request("initialize")
    def handle_initialize(params: dict[str, Any]) -> dict[str, Any]:
        _ = params  # unused
        return {
            "protocolVersion": "1.0",
            "capabilities": {
                "roots": False,
                "tools": True,
                "resources": True,
                "prompts": False,
                "sampling": False,
            },
        }

    @server.notification("initialized")
    def handle_initialized(params: dict[str, Any]) -> None:  # pragma: no cover - no-op
        _ = params

    @server.request("shutdown")
    def handle_shutdown(params: dict[str, Any]) -> dict[str, Any]:
        _ = params
        return {}

    @server.request("ping")
    def handle_ping(params: dict[str, Any]) -> dict[str, Any]:
        _ = params
        return {}

    @server.request("tools/list")
    def handle_list_tools(params: dict[str, Any]) -> dict[str, Any]:
        _ = params
        tools: list[dict[str, Any]] = []
        for agent_id, guide in guide_map.items():
            schema = tool_schemas[agent_id]
            tools.append(
                {
                    "name": agent_id,
                    "description": schema.description,
                    "inputSchema": schema.input_schema,
                }
            )
        return {"tools": tools}

    @server.request("tools/call")
    def handle_call_tool(params: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise MCPError(JSONRPC_INVALID_REQUEST, "Invalid tool parameters")
        name = params.get("name")
        if name not in guide_map:
            raise MCPError(
                JSONRPC_INVALID_REQUEST,
                f"Unknown agent '{name}'. Available: {', '.join(sorted(guide_map))}",
            )
        arguments = params.get("arguments") or {}
        try:
            model_name, force, override_url = _parse_arguments(arguments)
            invocation = prepare_agent_invocation(
                name,
                model_name=model_name,
                force=force,
                override_url=override_url,
            )
        except MCPError:
            raise
        except (ValueError, AgentPreparationError) as exc:
            raise MCPError(JSONRPC_INVALID_REQUEST, str(exc)) from exc

        output, returncode = _run_process(invocation.command, invocation.env, None if invocation.cwd is None else str(invocation.cwd))
        is_error = returncode != 0
        summary = f"{invocation.display_name} exited with code {returncode}"
        payload = output or summary
        content = [{"type": "text", "text": payload}]
        if output:
            content.append({"type": "text", "text": summary})
        return {
            "content": content,
            "isError": is_error,
        }

    @server.request("resources/list")
    def handle_list_resources(params: dict[str, Any]) -> dict[str, Any]:
        _ = params
        resources: list[dict[str, Any]] = [
            {
                "uri": CATALOG_RESOURCE_URI,
                "name": "Synth Agents Catalog",
                "description": "Markdown summary of supported IDE/CLI agents.",
                "mimeType": "text/markdown",
            }
        ]
        for guide in guide_map.values():
            resources.append(
                {
                    "uri": AGENT_RESOURCE_URI_TEMPLATE.format(agent_id=guide.id),
                    "name": guide.name,
                    "description": guide.summary,
                    "mimeType": "application/json",
                }
            )
        return {"resources": resources}

    @server.request("resources/read")
    def handle_read_resource(params: dict[str, Any]) -> dict[str, Any]:
        uri = params.get("uri")
        if not isinstance(uri, str):
            raise MCPError(JSONRPC_INVALID_REQUEST, "resources/read requires a string uri")

        if uri == CATALOG_RESOURCE_URI:
            markdown = render_agents_markdown()
            return {
                "contents": [
                    {
                        "type": "text",
                        "text": markdown,
                    }
                ]
            }

        if uri.startswith("synth://agents/"):
            agent_id = uri.rsplit("/", 1)[-1]
            guide = guide_map.get(agent_id)
            if guide:
                payload = json.dumps(guide.to_resource(), indent=2)
                return {
                    "contents": [
                        {
                            "type": "json",
                            "text": payload,
                        }
                    ]
                }

        raise MCPError(JSONRPC_INVALID_REQUEST, f"Unsupported resource: {uri}")

    return server


def _run_server(transport: str) -> None:
    if transport != "stdio":
        raise click.ClickException(f"Unsupported transport '{transport}'")
    server = build_server()
    server.serve_stdio()


@click.command("mcp-agents")
@click.option(
    "--transport",
    default="stdio",
    type=click.Choice(["stdio"]),
    show_default=True,
    help="Transport to use when serving MCP.",
)
def serve_mcp_agents(transport: str) -> None:
    """Launch the Synth Agents MCP server."""
    try:
        _run_server(transport)
    except KeyboardInterrupt:  # pragma: no cover - manual interrupt
        click.echo("Shutting down Synth Agents MCP server.")
