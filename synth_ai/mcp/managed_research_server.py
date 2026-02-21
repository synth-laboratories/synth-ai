"""Managed Research MCP server (stdio transport).

This server exposes managed-research control operations as MCP tools so external
agents can control SMR projects and runs through synth-ai.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Callable

from synth_ai import __version__
from synth_ai.sdk.managed_research import SmrControlClient

JSONDict = dict[str, Any]
ToolHandler = Callable[[JSONDict], Any]

SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2024-11-05")
DEFAULT_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]


class RpcError(Exception):
    """JSON-RPC error wrapper."""

    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


@dataclass(frozen=True)
class ToolDefinition:
    """MCP tool metadata and handler."""

    name: str
    description: str
    input_schema: JSONDict
    handler: ToolHandler


_CONNECTION_PROPERTIES: JSONDict = {
    "api_key": {
        "type": "string",
        "description": "Optional API key override (defaults to SYNTH_API_KEY).",
    },
    "backend_base": {
        "type": "string",
        "description": "Optional backend URL override (defaults to SYNTH_BACKEND_URL).",
    },
}


def _tool_schema(properties: JSONDict, required: list[str]) -> JSONDict:
    merged: JSONDict = dict(properties)
    merged.update(_CONNECTION_PROPERTIES)
    return {
        "type": "object",
        "properties": merged,
        "required": required,
        "additionalProperties": False,
    }


def _require_string(payload: JSONDict, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' is required and must be a non-empty string")
    return value.strip()


def _optional_string(payload: JSONDict, key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{key}' must be a string when provided")
    stripped = value.strip()
    return stripped or None


def _optional_bool(payload: JSONDict, key: str, default: bool = False) -> bool:
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"'{key}' must be a boolean when provided")


def _optional_int(payload: JSONDict, key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'{key}' must be an integer when provided")
    return value


class ManagedResearchMcpServer:
    """Minimal MCP server for managed research control."""

    def __init__(self) -> None:
        self._tools = {tool.name: tool for tool in self._build_tools()}

    def _client_from_args(self, args: JSONDict) -> SmrControlClient:
        return SmrControlClient(
            api_key=_optional_string(args, "api_key"),
            backend_base=_optional_string(args, "backend_base"),
        )

    def _build_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="smr_list_projects",
                description="List managed research projects.",
                input_schema=_tool_schema(
                    {
                        "include_archived": {
                            "type": "boolean",
                            "description": "Include archived projects in results.",
                        }
                    },
                    required=[],
                ),
                handler=self._tool_list_projects,
            ),
            ToolDefinition(
                name="smr_get_project",
                description="Fetch a managed research project.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        }
                    },
                    required=["project_id"],
                ),
                handler=self._tool_get_project,
            ),
            ToolDefinition(
                name="smr_get_project_status",
                description="Fetch status for a managed research project.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        }
                    },
                    required=["project_id"],
                ),
                handler=self._tool_get_project_status,
            ),
            ToolDefinition(
                name="smr_get_starting_data_upload_urls",
                description="Request presigned upload URLs for starting-data files.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "dataset_ref": {
                            "type": "string",
                            "description": "Optional dataset ref override (for example starting-data/banking77).",
                        },
                        "files": {
                            "type": "array",
                            "description": "File metadata entries to upload.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "content_type": {"type": "string"},
                                },
                                "required": ["path"],
                                "additionalProperties": False,
                            },
                            "minItems": 1,
                        },
                    },
                    required=["project_id", "files"],
                ),
                handler=self._tool_get_starting_data_upload_urls,
            ),
            ToolDefinition(
                name="smr_upload_starting_data",
                description="Upload starting-data file contents (text) via presigned URLs.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "dataset_ref": {
                            "type": "string",
                            "description": "Optional dataset ref override (for example starting-data/banking77).",
                        },
                        "files": {
                            "type": "array",
                            "description": "Files to upload (UTF-8 text content).",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "content": {"type": "string"},
                                    "content_type": {"type": "string"},
                                },
                                "required": ["path", "content"],
                                "additionalProperties": False,
                            },
                            "minItems": 1,
                        },
                    },
                    required=["project_id", "files"],
                ),
                handler=self._tool_upload_starting_data,
            ),
            ToolDefinition(
                name="smr_trigger_run",
                description="Trigger a run for a managed research project.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "timebox_seconds": {
                            "type": "integer",
                            "description": "Optional run timebox in seconds.",
                        },
                        "agent_model": {
                            "type": "string",
                            "description": (
                                "Override agent model for this run only, "
                                "e.g. 'claude-opus-4-5' or 'gpt-4o'. "
                                "Does not affect the project's default model."
                            ),
                        },
                        "agent_kind": {
                            "type": "string",
                            "enum": ["codex", "claude", "opencode"],
                            "description": (
                                "Override agent runtime for this run only: "
                                "'codex' (default), 'claude' (Claude Code), or 'opencode'."
                            ),
                        },
                    },
                    required=["project_id"],
                ),
                handler=self._tool_trigger_run,
            ),
            ToolDefinition(
                name="smr_set_agent_config",
                description=(
                    "Set the default agent model and/or kind for all future runs of a project. "
                    "Writes into project.execution.agent_model / agent_kind. "
                    "Use smr_trigger_run agent_model/agent_kind params for one-off overrides."
                ),
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "model": {
                            "type": "string",
                            "description": (
                                "Model string, e.g. 'claude-opus-4-5', 'gpt-4o', "
                                "'claude-haiku-4-5-20251001'."
                            ),
                        },
                        "agent_kind": {
                            "type": "string",
                            "enum": ["codex", "claude", "opencode"],
                            "description": "Agent runtime: 'codex' (default), 'claude', or 'opencode'.",
                        },
                    },
                    required=["project_id"],
                ),
                handler=self._tool_set_agent_config,
            ),
            ToolDefinition(
                name="smr_list_runs",
                description="List runs for a managed research project.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "active_only": {
                            "type": "boolean",
                            "description": "Return only active runs.",
                        },
                    },
                    required=["project_id"],
                ),
                handler=self._tool_list_runs,
            ),
            ToolDefinition(
                name="smr_get_run",
                description="Fetch a run by id.",
                input_schema=_tool_schema(
                    {
                        "run_id": {"type": "string", "description": "Run id."},
                        "project_id": {
                            "type": "string",
                            "description": "Optional project id for project-scoped fallback route.",
                        },
                    },
                    required=["run_id"],
                ),
                handler=self._tool_get_run,
            ),
            ToolDefinition(
                name="smr_pause_run",
                description="Pause a run.",
                input_schema=_tool_schema(
                    {"run_id": {"type": "string", "description": "Run id."}},
                    required=["run_id"],
                ),
                handler=self._tool_pause_run,
            ),
            ToolDefinition(
                name="smr_resume_run",
                description="Resume a paused run.",
                input_schema=_tool_schema(
                    {"run_id": {"type": "string", "description": "Run id."}},
                    required=["run_id"],
                ),
                handler=self._tool_resume_run,
            ),
            ToolDefinition(
                name="smr_stop_run",
                description="Stop a run.",
                input_schema=_tool_schema(
                    {"run_id": {"type": "string", "description": "Run id."}},
                    required=["run_id"],
                ),
                handler=self._tool_stop_run,
            ),
            ToolDefinition(
                name="smr_list_project_questions",
                description="List project-level pending questions.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "status_filter": {
                            "type": "string",
                            "description": "Question status filter.",
                        },
                    },
                    required=["project_id"],
                ),
                handler=self._tool_list_project_questions,
            ),
            ToolDefinition(
                name="smr_respond_question",
                description="Respond to a run question.",
                input_schema=_tool_schema(
                    {
                        "run_id": {"type": "string", "description": "Run id."},
                        "question_id": {"type": "string", "description": "Question id."},
                        "response_text": {"type": "string", "description": "Response text."},
                        "project_id": {
                            "type": "string",
                            "description": "Optional project id for project-scoped fallback route.",
                        },
                    },
                    required=["run_id", "question_id", "response_text"],
                ),
                handler=self._tool_respond_question,
            ),
            ToolDefinition(
                name="smr_list_project_approvals",
                description="List project-level approvals.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "status_filter": {
                            "type": "string",
                            "description": "Approval status filter.",
                        },
                    },
                    required=["project_id"],
                ),
                handler=self._tool_list_project_approvals,
            ),
            ToolDefinition(
                name="smr_resolve_approval",
                description="Approve or deny an approval request.",
                input_schema=_tool_schema(
                    {
                        "decision": {
                            "type": "string",
                            "enum": ["approve", "deny"],
                            "description": "Decision to apply.",
                        },
                        "run_id": {"type": "string", "description": "Run id."},
                        "approval_id": {"type": "string", "description": "Approval id."},
                        "comment": {
                            "type": "string",
                            "description": "Optional decision comment.",
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Optional project id for project-scoped fallback route.",
                        },
                    },
                    required=["decision", "run_id", "approval_id"],
                ),
                handler=self._tool_resolve_approval,
            ),
            ToolDefinition(
                name="smr_get_usage",
                description="Fetch project usage metrics.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        }
                    },
                    required=["project_id"],
                ),
                handler=self._tool_get_usage,
            ),
            ToolDefinition(
                name="smr_get_ops_status",
                description="Fetch ops/task status for a project.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "include_done_tasks": {
                            "type": "boolean",
                            "description": "Include completed tasks in response.",
                        },
                    },
                    required=["project_id"],
                ),
                handler=self._tool_get_ops_status,
            ),
            # --- Project lifecycle mutations ---
            ToolDefinition(
                name="smr_create_project",
                description="Create a new managed research project.",
                input_schema=_tool_schema(
                    {
                        "name": {
                            "type": "string",
                            "description": "Human-readable project name.",
                        },
                        "config": {
                            "type": "object",
                            "description": "Project configuration payload.",
                        },
                    },
                    required=["name"],
                ),
                handler=self._tool_create_project,
            ),
            ToolDefinition(
                name="smr_pause_project",
                description="Pause a managed research project (prevents new runs from starting).",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        }
                    },
                    required=["project_id"],
                ),
                handler=self._tool_pause_project,
            ),
            ToolDefinition(
                name="smr_resume_project",
                description="Resume a paused managed research project.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        }
                    },
                    required=["project_id"],
                ),
                handler=self._tool_resume_project,
            ),
            ToolDefinition(
                name="smr_archive_project",
                description="Archive a managed research project.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        }
                    },
                    required=["project_id"],
                ),
                handler=self._tool_archive_project,
            ),
            ToolDefinition(
                name="smr_unarchive_project",
                description="Unarchive a managed research project.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        }
                    },
                    required=["project_id"],
                ),
                handler=self._tool_unarchive_project,
            ),
            # --- Logs ---
            ToolDefinition(
                name="smr_get_run_logs",
                description=(
                    "Query VictoriaLogs for a specific run. "
                    "Returns structured log records with optional task/component filters."
                ),
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "run_id": {
                            "type": "string",
                            "description": "Run id.",
                        },
                        "task_key": {
                            "type": "string",
                            "description": "Optional filter by task key.",
                        },
                        "component": {
                            "type": "string",
                            "description": "Optional filter by component (e.g. 'orchestrator', 'worker').",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of log records to return (default 200, max 1000).",
                        },
                        "start": {
                            "type": "string",
                            "description": "Optional RFC3339 start time filter.",
                        },
                        "end": {
                            "type": "string",
                            "description": "Optional RFC3339 end time filter.",
                        },
                    },
                    required=["project_id", "run_id"],
                ),
                handler=self._tool_get_run_logs,
            ),
            ToolDefinition(
                name="smr_search_project_logs",
                description=(
                    "Free-text LogSQL search across VictoriaLogs for a project. "
                    "Use smr_get_run_logs for structured run-scoped queries."
                ),
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "q": {
                            "type": "string",
                            "description": "Optional free-text LogSQL query string.",
                        },
                        "run_id": {
                            "type": "string",
                            "description": "Optional filter by run id.",
                        },
                        "service": {
                            "type": "string",
                            "description": "Optional filter by service name.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of records (default 200).",
                        },
                        "start": {
                            "type": "string",
                            "description": "Optional RFC3339 start time filter.",
                        },
                        "end": {
                            "type": "string",
                            "description": "Optional RFC3339 end time filter.",
                        },
                    },
                    required=["project_id"],
                ),
                handler=self._tool_search_project_logs,
            ),
            # --- Artifacts + results ---
            ToolDefinition(
                name="smr_list_run_artifacts",
                description="List artifacts produced by a run.",
                input_schema=_tool_schema(
                    {
                        "run_id": {
                            "type": "string",
                            "description": "Run id.",
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Optional project id for project-scoped lookup.",
                        },
                    },
                    required=["run_id"],
                ),
                handler=self._tool_list_run_artifacts,
            ),
            ToolDefinition(
                name="smr_get_artifact",
                description="Fetch artifact metadata (title, uri, type) by artifact id.",
                input_schema=_tool_schema(
                    {
                        "artifact_id": {
                            "type": "string",
                            "description": "Artifact id.",
                        }
                    },
                    required=["artifact_id"],
                ),
                handler=self._tool_get_artifact,
            ),
            ToolDefinition(
                name="smr_get_run_results",
                description=(
                    "Get a run result summary: outcome, artifacts grouped by type, "
                    "and a pre-built log query hint for debugging."
                ),
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "run_id": {
                            "type": "string",
                            "description": "Run id.",
                        },
                    },
                    required=["project_id", "run_id"],
                ),
                handler=self._tool_get_run_results,
            ),
            ToolDefinition(
                name="smr_get_project_git_status",
                description=(
                    "Get read-only workspace git status for a project: "
                    "commit SHA, last push timestamp, default branch, and optional "
                    "remote repo metadata. Does not expose storage internals or allow download."
                ),
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        }
                    },
                    required=["project_id"],
                ),
                handler=self._tool_get_project_git_status,
            ),
            ToolDefinition(
                name="smr_get_orchestrator_status",
                description=(
                    "Get orchestrator status for a run: current phase, heartbeat, "
                    "turn count, turn history (phase + outcome + timing for each turn), "
                    "and a log query hint scoped to the orchestrator component."
                ),
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "run_id": {
                            "type": "string",
                            "description": "Run id.",
                        },
                    },
                    required=["project_id", "run_id"],
                ),
                handler=self._tool_get_orchestrator_status,
            ),
        ]

    # Tool handlers -----------------------------------------------------

    def _tool_list_projects(self, args: JSONDict) -> Any:
        include_archived = _optional_bool(args, "include_archived", default=False)
        with self._client_from_args(args) as client:
            return client.list_projects(include_archived=include_archived)

    def _tool_get_project(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_project(project_id)

    def _tool_get_project_status(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_project_status(project_id)

    def _tool_get_starting_data_upload_urls(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        dataset_ref = _optional_string(args, "dataset_ref")
        files = args.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("'files' must be a non-empty array")

        normalized_files: list[JSONDict] = []
        for entry in files:
            if not isinstance(entry, dict):
                raise ValueError("each file entry must be an object")
            path = _require_string(entry, "path")
            normalized: JSONDict = {"path": path}
            content_type = _optional_string(entry, "content_type")
            if content_type:
                normalized["content_type"] = content_type
            normalized_files.append(normalized)

        with self._client_from_args(args) as client:
            return client.get_starting_data_upload_urls(
                project_id,
                files=normalized_files,
                dataset_ref=dataset_ref,
            )

    def _tool_upload_starting_data(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        dataset_ref = _optional_string(args, "dataset_ref")
        files = args.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("'files' must be a non-empty array")

        normalized_files: list[JSONDict] = []
        for entry in files:
            if not isinstance(entry, dict):
                raise ValueError("each file entry must be an object")
            path = _require_string(entry, "path")
            content = entry.get("content")
            if not isinstance(content, str):
                raise ValueError("each file entry requires string 'content'")
            normalized: JSONDict = {"path": path, "content": content}
            content_type = _optional_string(entry, "content_type")
            if content_type:
                normalized["content_type"] = content_type
            normalized_files.append(normalized)

        with self._client_from_args(args) as client:
            return client.upload_starting_data_files(
                project_id,
                files=normalized_files,
                dataset_ref=dataset_ref,
            )

    def _tool_trigger_run(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        timebox_seconds = _optional_int(args, "timebox_seconds")
        agent_model = _optional_string(args, "agent_model")
        agent_kind = _optional_string(args, "agent_kind")
        with self._client_from_args(args) as client:
            return client.trigger_run(
                project_id,
                timebox_seconds=timebox_seconds,
                agent_model=agent_model,
                agent_kind=agent_kind,
            )

    def _tool_set_agent_config(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        model = _optional_string(args, "model")
        agent_kind = _optional_string(args, "agent_kind")
        if model is None and agent_kind is None:
            raise ValueError("at least one of 'model' or 'agent_kind' is required")
        with self._client_from_args(args) as client:
            return client.set_agent_config(project_id, model=model, agent_kind=agent_kind)

    def _tool_list_runs(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        active_only = _optional_bool(args, "active_only", default=False)
        with self._client_from_args(args) as client:
            if active_only:
                return client.list_active_runs(project_id)
            return client.list_runs(project_id)

    def _tool_get_run(self, args: JSONDict) -> Any:
        run_id = _require_string(args, "run_id")
        project_id = _optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_run(run_id, project_id=project_id)

    def _tool_pause_run(self, args: JSONDict) -> Any:
        run_id = _require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.pause_run(run_id)

    def _tool_resume_run(self, args: JSONDict) -> Any:
        run_id = _require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.resume_run(run_id)

    def _tool_stop_run(self, args: JSONDict) -> Any:
        run_id = _require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.stop_run(run_id)

    def _tool_list_project_questions(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        status_filter = _optional_string(args, "status_filter") or "pending"
        with self._client_from_args(args) as client:
            return client.list_project_questions(project_id, status_filter=status_filter)

    def _tool_respond_question(self, args: JSONDict) -> Any:
        run_id = _require_string(args, "run_id")
        question_id = _require_string(args, "question_id")
        response_text = _require_string(args, "response_text")
        project_id = _optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.respond_question(
                run_id,
                question_id,
                response_text=response_text,
                project_id=project_id,
            )

    def _tool_list_project_approvals(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        status_filter = _optional_string(args, "status_filter") or "pending"
        with self._client_from_args(args) as client:
            return client.list_project_approvals(project_id, status_filter=status_filter)

    def _tool_resolve_approval(self, args: JSONDict) -> Any:
        decision = _require_string(args, "decision")
        run_id = _require_string(args, "run_id")
        approval_id = _require_string(args, "approval_id")
        comment = _optional_string(args, "comment")
        project_id = _optional_string(args, "project_id")

        with self._client_from_args(args) as client:
            if decision == "approve":
                return client.approve(run_id, approval_id, comment=comment, project_id=project_id)
            if decision == "deny":
                return client.deny(run_id, approval_id, comment=comment, project_id=project_id)
        raise ValueError("'decision' must be 'approve' or 'deny'")

    def _tool_get_usage(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_usage(project_id)

    def _tool_get_ops_status(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        include_done_tasks = args.get("include_done_tasks")
        if include_done_tasks is not None and not isinstance(include_done_tasks, bool):
            raise ValueError("'include_done_tasks' must be a boolean when provided")
        with self._client_from_args(args) as client:
            return client.get_ops_status(project_id, include_done_tasks=include_done_tasks)

    # Project lifecycle mutations ---------------------------------------

    def _tool_create_project(self, args: JSONDict) -> Any:
        name = _require_string(args, "name")
        config = args.get("config") or {}
        if not isinstance(config, dict):
            raise ValueError("'config' must be a JSON object when provided")
        payload: JSONDict = {"name": name, **config}
        with self._client_from_args(args) as client:
            return client.create_project(payload)

    def _tool_pause_project(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.pause_project(project_id)

    def _tool_resume_project(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.resume_project(project_id)

    def _tool_archive_project(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.archive_project(project_id)

    def _tool_unarchive_project(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.unarchive_project(project_id)

    # Logs --------------------------------------------------------------

    def _tool_get_run_logs(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        run_id = _require_string(args, "run_id")
        task_key = _optional_string(args, "task_key")
        component = _optional_string(args, "component")
        limit_raw = _optional_int(args, "limit")
        limit = limit_raw if limit_raw is not None else 200
        start = _optional_string(args, "start")
        end = _optional_string(args, "end")
        with self._client_from_args(args) as client:
            return client.get_run_logs(
                project_id,
                run_id,
                task_key=task_key,
                component=component,
                limit=limit,
                start=start,
                end=end,
            )

    def _tool_search_project_logs(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        q = _optional_string(args, "q")
        run_id = _optional_string(args, "run_id")
        service = _optional_string(args, "service")
        limit_raw = _optional_int(args, "limit")
        limit = limit_raw if limit_raw is not None else 200
        start = _optional_string(args, "start")
        end = _optional_string(args, "end")
        with self._client_from_args(args) as client:
            return client.search_victoria_logs(
                project_id,
                q=q,
                run_id=run_id,
                service=service,
                limit=limit,
                start=start,
                end=end,
            )

    # Artifacts + results -----------------------------------------------

    def _tool_list_run_artifacts(self, args: JSONDict) -> Any:
        run_id = _require_string(args, "run_id")
        project_id = _optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.list_run_artifacts(run_id, project_id=project_id)

    def _tool_get_artifact(self, args: JSONDict) -> Any:
        artifact_id = _require_string(args, "artifact_id")
        with self._client_from_args(args) as client:
            return client.get_artifact(artifact_id)

    def _tool_get_run_results(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        run_id = _require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.get_run_results(project_id, run_id)

    def _tool_get_project_git_status(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_project_git_status(project_id)

    def _tool_get_orchestrator_status(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        run_id = _require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.get_run_orchestrator_status(project_id, run_id)

    # Protocol ----------------------------------------------------------

    def list_tools(self) -> list[JSONDict]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    def _handle_initialize(self, params: JSONDict | None) -> JSONDict:
        requested_version = None
        if isinstance(params, dict):
            requested_version = params.get("protocolVersion")

        protocol_version = DEFAULT_PROTOCOL_VERSION
        if isinstance(requested_version, str) and requested_version in SUPPORTED_PROTOCOL_VERSIONS:
            protocol_version = requested_version

        return {
            "protocolVersion": protocol_version,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {},
                "prompts": {},
            },
            "serverInfo": {
                "name": "synth-ai-managed-research",
                "version": __version__,
            },
            "instructions": (
                "Use tools to control managed research projects and runs. "
                "All tool outputs are JSON-encoded in text content blocks."
            ),
        }

    def _handle_tools_call(self, params: JSONDict | None) -> JSONDict:
        if not isinstance(params, dict):
            return self._tool_error("Invalid params for tools/call")

        name = params.get("name")
        if not isinstance(name, str) or not name.strip():
            return self._tool_error("Tool call is missing a valid 'name'")

        tool = self._tools.get(name)
        if tool is None:
            return self._tool_error(f"Unknown tool: {name}")

        arguments = params.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return self._tool_error("Tool 'arguments' must be a JSON object")

        try:
            result = tool.handler(arguments)
        except Exception as exc:
            return self._tool_error(f"{type(exc).__name__}: {exc}")

        text = json.dumps(result, indent=2, default=str)
        return {"content": [{"type": "text", "text": text}]}

    def _tool_error(self, message: str) -> JSONDict:
        return {
            "content": [{"type": "text", "text": message}],
            "isError": True,
        }

    def dispatch(self, method: str, params: JSONDict | None) -> JSONDict:
        if method == "initialize":
            return self._handle_initialize(params)
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": self.list_tools()}
        if method == "tools/call":
            return self._handle_tools_call(params)
        if method == "resources/list":
            return {"resources": []}
        if method == "prompts/list":
            return {"prompts": []}
        raise RpcError(-32601, f"Method not found: {method}")

    def handle_notification(self, method: str, _params: JSONDict | None) -> None:
        # No-op, but keep known notifications explicit.
        if method == "notifications/initialized":
            return

    def handle_request(self, message: JSONDict) -> JSONDict:
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params")

        if not isinstance(method, str):
            return _jsonrpc_error(request_id, -32600, "Invalid request: missing method")

        try:
            result = self.dispatch(method, params if isinstance(params, dict) else None)
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except RpcError as exc:
            return _jsonrpc_error(request_id, exc.code, exc.message, data=exc.data)
        except Exception as exc:
            return _jsonrpc_error(request_id, -32603, f"Internal error: {exc}")


def _jsonrpc_error(request_id: Any, code: int, message: str, *, data: Any | None = None) -> JSONDict:
    error: JSONDict = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def _read_message(stdin: Any) -> JSONDict | None:
    headers: dict[str, str] = {}

    while True:
        line = stdin.readline()
        if line == b"":
            return None
        if line in (b"\r\n", b"\n"):
            break

        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Invalid header encoding: {exc}") from exc

        if ":" not in text:
            continue
        key, value = text.split(":", 1)
        headers[key.strip().lower()] = value.strip()

    raw_length = headers.get("content-length")
    if raw_length is None:
        raise ValueError("Missing Content-Length header")

    try:
        content_length = int(raw_length)
    except ValueError as exc:
        raise ValueError(f"Invalid Content-Length: {raw_length}") from exc

    if content_length < 0:
        raise ValueError("Content-Length must be non-negative")

    payload = stdin.read(content_length)
    if len(payload) != content_length:
        raise ValueError("Unexpected EOF while reading message body")

    try:
        message = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc

    if not isinstance(message, dict):
        raise ValueError("JSON-RPC message must be a JSON object")
    return message


def _write_message(stdout: Any, message: JSONDict) -> None:
    payload = json.dumps(message, separators=(",", ":"), default=str).encode("utf-8")
    header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
    stdout.write(header)
    stdout.write(payload)
    stdout.flush()


def run_stdio_server() -> None:
    """Run the MCP server using stdio transport."""
    server = ManagedResearchMcpServer()
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        try:
            message = _read_message(stdin)
        except Exception as exc:
            _write_message(stdout, _jsonrpc_error(None, -32700, f"Parse error: {exc}"))
            continue

        if message is None:
            break

        method = message.get("method")
        request_id = message.get("id")

        if isinstance(method, str) and request_id is None:
            server.handle_notification(method, message.get("params") if isinstance(message.get("params"), dict) else None)
            continue

        if isinstance(method, str):
            _write_message(stdout, server.handle_request(message))
            continue

        _write_message(stdout, _jsonrpc_error(message.get("id"), -32600, "Invalid request"))


def main() -> None:
    """CLI entrypoint for the managed-research MCP server."""
    run_stdio_server()


__all__ = ["ManagedResearchMcpServer", "main", "run_stdio_server"]
