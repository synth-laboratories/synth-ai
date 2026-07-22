"""Synth Tag MCP tool definitions."""

from __future__ import annotations

from typing import Any

from synth_ai.core.research._legacy.models.tag import TagSessionCreateRequest
from synth_ai.mcp.research.registry import (
    READ_SCOPES,
    WRITE_SCOPES,
    ToolDefinition,
    tool_schema,
)


def _client(server: Any, args: dict[str, Any]):
    return server._client_from_args(args)


def _tool_body(args: dict[str, Any], *, exclude: set[str]) -> dict[str, Any]:
    return {
        key: value
        for key, value in args.items()
        if key not in {*exclude, "api_key", "backend_base"} and value is not None
    }


def build_tag_tools(server: Any) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="tag_create_session",
            description="Create a Synth Tag v1 research session and launch its bound SMR run.",
            input_schema=tool_schema(
                {
                    "request": {"type": "string", "description": "Delegated task text."},
                    "definition_of_done": {
                        "type": "string",
                        "description": "Optional definition of done passed into the run objective.",
                    },
                    "scope_id": {
                        "type": "string",
                        "description": "Optional Tag scope id. Defaults to the org scope.",
                    },
                    "factory_id": {
                        "type": "string",
                        "description": "Owning Factory id (required by backend).",
                    },
                    "effort_id": {
                        "type": "string",
                        "description": "Owning canonical-project Effort id.",
                    },
                    "experiment_id": {
                        "type": "string",
                        "description": "Optional owner experiment binding.",
                    },
                    "candidate_id": {
                        "type": "string",
                        "description": "Optional candidate within experiment_id.",
                    },
                    "timebox_seconds": {
                        "type": "integer",
                        "description": "Optional run timebox in seconds.",
                    },
                    "runbook_preset": {
                        "type": "string",
                        "description": "Optional runbook preset. Defaults to tag_steward.",
                    },
                    "metadata": {"type": "object", "description": "Optional metadata."},
                    "api_key": {"type": "string"},
                    "backend_base": {"type": "string"},
                },
                required=["request", "factory_id", "effort_id"],
            ),
            handler=lambda args: _client(server, args).tag.create_session(
                TagSessionCreateRequest(**_tool_body(args, exclude=set()))
            ),
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="tag_list_sessions",
            description="List org-owned Factory Tag sessions with optional owner filters.",
            input_schema=tool_schema(
                {
                    "factory_id": {"type": "string"},
                    "effort_id": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
                    "api_key": {"type": "string"},
                    "backend_base": {"type": "string"},
                },
                required=[],
            ),
            handler=lambda args: _client(server, args).tag.list_sessions(
                factory_id=args.get("factory_id"),
                effort_id=args.get("effort_id"),
                limit=int(args.get("limit") or 50),
            ),
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="tag_get_session",
            description="Get a Synth Tag session receipt and current run status.",
            input_schema=tool_schema(
                {
                    "session_id": {"type": "string", "description": "Tag session id."},
                    "api_key": {"type": "string"},
                    "backend_base": {"type": "string"},
                },
                required=["session_id"],
            ),
            handler=lambda args: _client(server, args).tag.get_session(str(args["session_id"])),
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="tag_watch_session",
            description="Reconnect to a Tag session and read its ordered message ledger.",
            input_schema=tool_schema(
                {
                    "session_id": {"type": "string"},
                    "api_key": {"type": "string"},
                    "backend_base": {"type": "string"},
                },
                required=["session_id"],
            ),
            handler=lambda args: _client(server, args).tag.watch_session(str(args["session_id"])),
            required_scopes=READ_SCOPES,
        ),
        ToolDefinition(
            name="tag_send_message",
            description="Steer the active run or persist direction for the next cycle.",
            input_schema=tool_schema(
                {
                    "session_id": {"type": "string", "description": "Tag session id."},
                    "message": {"type": "string", "description": "Steering message."},
                    "steering_target": {
                        "type": "string",
                        "enum": ["active_run", "next_cycle"],
                        "default": "active_run",
                    },
                    "metadata": {"type": "object", "description": "Optional metadata."},
                    "idempotency_key": {
                        "type": "string",
                        "description": "Optional client idempotency key.",
                    },
                    "api_key": {"type": "string"},
                    "backend_base": {"type": "string"},
                },
                required=["session_id", "message"],
            ),
            handler=lambda args: _client(server, args).tag.send_message(
                str(args["session_id"]),
                _tool_body(args, exclude={"session_id"}),
            ),
            required_scopes=WRITE_SCOPES,
        ),
        ToolDefinition(
            name="tag_control_session",
            description="Pause, stop, archive, or reconnect a Factory Tag session.",
            input_schema=tool_schema(
                {
                    "session_id": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["pause", "stop", "archive", "reconnect"],
                    },
                    "api_key": {"type": "string"},
                    "backend_base": {"type": "string"},
                },
                required=["session_id", "action"],
            ),
            handler=lambda args: _client(server, args).tag.control_session(
                str(args["session_id"]), str(args["action"])
            ),
            required_scopes=WRITE_SCOPES,
        ),
    ]


__all__ = ["build_tag_tools"]
