"""Managed Research MCP server (stdio transport).

This server exposes managed-research control operations as MCP tools so external
agents can control SMR projects and runs through synth-ai.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

from synth_ai import __version__
from synth_ai.core.utils.env import get_api_key
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


def _optional_object(payload: JSONDict, key: str) -> JSONDict | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be an object when provided")
    return value


class ManagedResearchMcpServer:
    """Minimal MCP server for managed research control."""

    def __init__(self) -> None:
        self._tools = {tool.name: tool for tool in self._build_tools()}

    def available_tool_names(self) -> list[str]:
        """Return sorted MCP tool names exposed by this server."""
        return sorted(self._tools.keys())

    def _client_from_args(self, args: JSONDict) -> SmrControlClient:
        return SmrControlClient(
            api_key=_optional_string(args, "api_key"),
            backend_base=_optional_string(args, "backend_base"),
        )

    def _build_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="smr_health_check",
                description="Return a setup and connectivity health report for the managed research MCP server.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Optional project id to validate project status access.",
                        }
                    },
                    required=[],
                ),
                handler=self._tool_health_check,
            ),
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
                name="smr_get_binding",
                description="Fetch the active project binding (pool lineage and runtime/environment resolution).",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "run_id": {
                            "type": "string",
                            "description": "Optional expected published_by_run_id for handoff verification.",
                        },
                    },
                    required=["project_id"],
                ),
                handler=self._tool_get_binding,
            ),
            ToolDefinition(
                name="smr_promote_binding",
                description="Promote/update active binding with expected-revision CAS semantics.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "pool_id": {
                            "type": "string",
                            "description": "Target pool id to bind.",
                        },
                        "dataset_revision": {
                            "type": "string",
                            "description": "Dataset revision id to bind.",
                        },
                        "expected_revision": {
                            "type": "integer",
                            "description": "Current binding revision expected by caller (CAS).",
                        },
                        "runtime_kind": {
                            "type": "string",
                            "description": "Optional runtime kind override.",
                        },
                        "environment_kind": {
                            "type": "string",
                            "description": "Optional environment kind override.",
                        },
                        "published_by_run_id": {
                            "type": "string",
                            "description": "Optional run id publishing this binding.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Optional reason for audit trail.",
                        },
                        "idempotency_key": {
                            "type": "string",
                            "description": "Optional idempotency key.",
                        },
                    },
                    required=["project_id", "pool_id", "dataset_revision", "expected_revision"],
                ),
                handler=self._tool_promote_binding,
            ),
            ToolDefinition(
                name="smr_get_pool_context",
                description=(
                    "Fetch project/run pool context for worker coordination: active binding, "
                    "run-level pool ledger summary, recommended target (if any), and fallback policy."
                ),
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "run_id": {
                            "type": "string",
                            "description": "Optional run id used to read run-scoped pool metadata.",
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Optional task id for task-level assignment lookup.",
                        },
                    },
                    required=["project_id"],
                ),
                handler=self._tool_get_pool_context,
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
                        "idempotency_key_upload": {
                            "type": "string",
                            "description": "Canonical idempotency key for upload retries.",
                        },
                        "idempotency_key": {
                            "type": "string",
                            "description": "Deprecated alias for idempotency_key_upload.",
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
                        "idempotency_key_upload": {
                            "type": "string",
                            "description": "Canonical idempotency key for upload retries.",
                        },
                        "idempotency_key": {
                            "type": "string",
                            "description": "Deprecated alias for idempotency_key_upload.",
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
                        "workflow": {
                            "type": "object",
                            "description": (
                                "Optional workflow payload for rails such as data_factory_v1. "
                                "When omitted, behavior is unchanged."
                            ),
                            "properties": {
                                "kind": {"type": "string"},
                                "profile": {"type": "string"},
                                "source_mode": {
                                    "type": "string",
                                    "enum": [
                                        "mcp_local",
                                        "oneshot_mcp_local",
                                        "synth_mcp_local",
                                        "frontend_interactive",
                                    ],
                                },
                                "targets": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": [
                                            "harbor",
                                            "openenv",
                                            "archipelago",
                                            "custom_container",
                                            "synth_container",
                                        ],
                                    },
                                    "minItems": 1,
                                },
                                "preferred_target": {
                                    "type": "string",
                                    "enum": [
                                        "harbor",
                                        "openenv",
                                        "archipelago",
                                        "custom_container",
                                        "synth_container",
                                    ],
                                },
                                "runtime_kind": {
                                    "type": "string",
                                    "enum": ["react_mcp", "react", "horizons", "sandbox_agent"],
                                },
                                "environment_kind": {
                                    "type": "string",
                                    "enum": [
                                        "harbor",
                                        "openenv",
                                        "archipelago",
                                        "custom_container",
                                        "synth_container",
                                    ],
                                },
                                "template": {
                                    "type": "string",
                                    "enum": ["harbor_hardening_v1"],
                                },
                                "input": {
                                    "type": "object",
                                    "properties": {
                                        "dataset_ref": {"type": "string"},
                                        "bundle_manifest_path": {"type": "string"},
                                        "session_id": {"type": "string"},
                                        "session_state": {"type": "string"},
                                        "session_title": {"type": "string"},
                                        "session_notes": {"type": "string"},
                                    },
                                    "required": ["dataset_ref", "bundle_manifest_path"],
                                    "additionalProperties": False,
                                },
                                "options": {
                                    "type": "object",
                                    "properties": {
                                        "strictness_mode": {
                                            "type": "string",
                                            "enum": ["warn", "strict"],
                                        }
                                    },
                                    "additionalProperties": False,
                                },
                            },
                            "required": ["kind", "source_mode", "targets", "input"],
                            "additionalProperties": False,
                        },
                        "idempotency_key_run_create": {
                            "type": "string",
                            "description": "Canonical idempotency key for run-create retries.",
                        },
                        "idempotency_key": {
                            "type": "string",
                            "description": "Deprecated alias for idempotency_key_run_create.",
                        },
                    },
                    required=["project_id"],
                ),
                handler=self._tool_trigger_run,
            ),
            ToolDefinition(
                name="smr_trigger_data_factory",
                description=(
                    "Trigger a standardized Data Factory run "
                    "(syntactic sugar over smr_trigger_run with workflow.kind=data_factory_v1)."
                ),
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "dataset_ref": {
                            "type": "string",
                            "description": "S3-prefix style dataset ref containing capture bundle files.",
                        },
                        "bundle_manifest_path": {
                            "type": "string",
                            "description": "Path under dataset_ref to capture_bundle.json.",
                        },
                        "profile": {
                            "type": "string",
                            "enum": ["founder_default", "researcher_strict"],
                            "description": "Data Factory profile rail (default founder_default).",
                        },
                        "source_mode": {
                            "type": "string",
                            "enum": [
                                "mcp_local",
                                "oneshot_mcp_local",
                                "synth_mcp_local",
                                "frontend_interactive",
                            ],
                            "description": "Capture source mode (default synth_mcp_local).",
                        },
                        "template": {
                            "type": "string",
                            "enum": ["harbor_hardening_v1"],
                            "description": "Optional workflow template preset.",
                        },
                        "targets": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "harbor",
                                    "openenv",
                                    "archipelago",
                                    "custom_container",
                                    "synth_container",
                                ],
                            },
                            "minItems": 1,
                            "description": "Execution targets in priority set.",
                        },
                        "preferred_target": {
                            "type": "string",
                            "enum": [
                                "harbor",
                                "openenv",
                                "archipelago",
                                "custom_container",
                                "synth_container",
                            ],
                            "description": "Preferred target (default harbor).",
                        },
                        "runtime_kind": {
                            "type": "string",
                            "enum": ["react_mcp", "react", "horizons", "sandbox_agent"],
                            "description": "Optional runtime kind for compatibility gating.",
                        },
                        "environment_kind": {
                            "type": "string",
                            "enum": [
                                "harbor",
                                "openenv",
                                "archipelago",
                                "custom_container",
                                "synth_container",
                            ],
                            "description": "Optional environment kind for compatibility gating.",
                        },
                        "strictness_mode": {
                            "type": "string",
                            "enum": ["warn", "strict"],
                            "description": "Validation strictness mode (default warn).",
                        },
                        "timebox_seconds": {
                            "type": "integer",
                            "description": "Optional run timebox in seconds.",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Interactive session identifier when source_mode=frontend_interactive.",
                        },
                        "session_state": {
                            "type": "string",
                            "enum": [
                                "empty",
                                "active",
                                "completed",
                                "uploaded",
                                "finalizing",
                                "finalized",
                                "publish-ready",
                                "blocked",
                                "recoverable-fail",
                            ],
                            "description": "Interactive session lifecycle state.",
                        },
                        "session_title": {
                            "type": "string",
                            "description": "Optional interactive session title.",
                        },
                        "session_notes": {
                            "type": "string",
                            "description": "Optional interactive session notes/context.",
                        },
                        "idempotency_key_run_create": {
                            "type": "string",
                            "description": "Canonical idempotency key for run-create retries.",
                        },
                        "idempotency_key": {
                            "type": "string",
                            "description": "Deprecated alias for idempotency_key_run_create.",
                        },
                    },
                    required=["project_id", "dataset_ref", "bundle_manifest_path"],
                ),
                handler=self._tool_trigger_data_factory,
            ),
            ToolDefinition(
                name="smr_data_factory_finalize",
                description=("Submit a Data Factory finalization run via the dedicated endpoint."),
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "dataset_ref": {
                            "type": "string",
                            "description": "Dataset ref containing capture bundle files.",
                        },
                        "bundle_manifest_path": {
                            "type": "string",
                            "description": "Path under dataset_ref to capture_bundle.json.",
                        },
                        "template": {
                            "type": "string",
                            "enum": ["harbor_hardening_v1"],
                            "description": "Optional workflow template preset.",
                        },
                        "target_formats": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "harbor",
                                    "openenv",
                                    "archipelago",
                                    "custom_container",
                                    "synth_container",
                                ],
                            },
                            "minItems": 1,
                            "description": "Execution target formats.",
                        },
                        "preferred_target": {
                            "type": "string",
                            "enum": [
                                "harbor",
                                "openenv",
                                "archipelago",
                                "custom_container",
                                "synth_container",
                            ],
                            "description": "Preferred target (default harbor).",
                        },
                        "finalizer_profile": {
                            "type": "string",
                            "enum": ["founder_default", "researcher_strict"],
                            "description": "Data Factory profile rail (default founder_default).",
                        },
                        "source_mode": {
                            "type": "string",
                            "enum": [
                                "mcp_local",
                                "oneshot_mcp_local",
                                "synth_mcp_local",
                                "frontend_interactive",
                            ],
                            "description": "Capture source mode (default synth_mcp_local).",
                        },
                        "runtime_kind": {
                            "type": "string",
                            "enum": ["react_mcp", "react", "horizons", "sandbox_agent"],
                            "description": "Optional runtime kind for compatibility gating.",
                        },
                        "environment_kind": {
                            "type": "string",
                            "enum": [
                                "harbor",
                                "openenv",
                                "archipelago",
                                "custom_container",
                                "synth_container",
                            ],
                            "description": "Optional environment kind for compatibility gating.",
                        },
                        "strictness_mode": {
                            "type": "string",
                            "enum": ["warn", "strict"],
                            "description": "Validation strictness mode (default warn).",
                        },
                        "timebox_seconds": {
                            "type": "integer",
                            "description": "Optional run timebox in seconds.",
                        },
                        "idempotency_key_run_create": {
                            "type": "string",
                            "description": "Canonical idempotency key for run-create retries.",
                        },
                        "idempotency_key": {
                            "type": "string",
                            "description": "Deprecated alias for idempotency_key_run_create.",
                        },
                    },
                    required=["project_id"],
                ),
                handler=self._tool_data_factory_finalize,
            ),
            ToolDefinition(
                name="smr_data_factory_finalize_status",
                description="Fetch Data Factory finalization status by job id.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "job_id": {
                            "type": "string",
                            "description": "Finalization job id (run id).",
                        },
                    },
                    required=["project_id", "job_id"],
                ),
                handler=self._tool_data_factory_finalize_status,
            ),
            ToolDefinition(
                name="smr_data_factory_publish",
                description="Publish finalized Data Factory artifacts.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "job_id": {
                            "type": "string",
                            "description": "Finalization job id (run id).",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Publish reason (default manual_publish).",
                        },
                        "idempotency_key_publish": {
                            "type": "string",
                            "description": "Canonical idempotency key for publish retries.",
                        },
                        "idempotency_key": {
                            "type": "string",
                            "description": "Deprecated alias for idempotency_key_publish.",
                        },
                    },
                    required=["project_id", "job_id"],
                ),
                handler=self._tool_data_factory_publish,
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
                name="smr_list_jobs",
                description=(
                    "List org-level SMR jobs feed (runs), optionally filtered by project/state."
                ),
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Optional managed research project id filter.",
                        },
                        "state": {
                            "type": "string",
                            "description": "Optional run state filter (single or comma-separated).",
                        },
                        "active_only": {
                            "type": "boolean",
                            "description": "Return only active runs.",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 200,
                            "description": "Maximum rows to return (default 50).",
                        },
                    },
                    required=[],
                ),
                handler=self._tool_list_jobs,
            ),
            ToolDefinition(
                name="smr_get_run",
                description="Fetch a run by id.",
                input_schema=_tool_schema(
                    {
                        "run_id": {"type": "string", "description": "Run id."},
                        "project_id": {
                            "type": "string",
                            "description": "Optional project id for project-scoped strict route.",
                        },
                    },
                    required=["run_id"],
                ),
                handler=self._tool_get_run,
            ),
            ToolDefinition(
                name="smr_get_run_usage",
                description="Fetch run-level usage with charged-spend totals and ledger entries.",
                input_schema=_tool_schema(
                    {
                        "run_id": {"type": "string", "description": "Run id."},
                        "project_id": {
                            "type": "string",
                            "description": "Optional project id for project-scoped strict route.",
                        },
                    },
                    required=["run_id"],
                ),
                handler=self._tool_get_run_usage,
            ),
            ToolDefinition(
                name="smr_get_actor_status",
                description="Fetch unified actor status (orchestrator + workers) for a project.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "run_id": {
                            "type": "string",
                            "description": "Optional run id filter.",
                        },
                    },
                    required=["project_id"],
                ),
                handler=self._tool_get_actor_status,
            ),
            ToolDefinition(
                name="smr_control_actor",
                description="Pause or resume an orchestrator/worker actor within a run.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "run_id": {"type": "string", "description": "Run id."},
                        "actor_id": {
                            "type": "string",
                            "description": "Actor id (orchestrator or worker id).",
                        },
                        "action": {
                            "type": "string",
                            "enum": ["pause", "resume"],
                            "description": "Control action.",
                        },
                        "reason": {"type": "string", "description": "Optional operator reason."},
                        "idempotency_key": {
                            "type": "string",
                            "description": "Optional idempotency key for retries.",
                        },
                    },
                    required=["project_id", "run_id", "actor_id", "action"],
                ),
                handler=self._tool_control_actor,
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
                            "description": "Optional project id for project-scoped strict route.",
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
                            "description": "Optional project id for project-scoped strict route.",
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
            ToolDefinition(
                name="smr_codex_subscription_status",
                description="Get global Codex subscription connection status.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Optional project id to read project-bound state.",
                        }
                    },
                    required=[],
                ),
                handler=self._tool_codex_subscription_status,
            ),
            ToolDefinition(
                name="smr_codex_subscription_connect_start",
                description="Start Codex subscription login (returns authorize_url and instructions).",
                input_schema=_tool_schema(
                    {
                        "sandbox_agent_url": {
                            "type": "string",
                            "description": "Optional connector URL override.",
                        },
                        "provider_id": {
                            "type": "string",
                            "description": "Optional connector provider id override (for example openai or codex).",
                        },
                        "external_account_hint": {
                            "type": "string",
                            "description": "Optional account hint to store with the connection.",
                        },
                    },
                    required=[],
                ),
                handler=self._tool_codex_subscription_connect_start,
            ),
            ToolDefinition(
                name="smr_codex_subscription_connect_complete",
                description="Complete Codex subscription login after browser consent.",
                input_schema=_tool_schema(
                    {
                        "code": {
                            "type": "string",
                            "description": "Optional OAuth code for code-based flows.",
                        },
                        "sandbox_agent_url": {
                            "type": "string",
                            "description": "Optional connector URL override.",
                        },
                    },
                    required=[],
                ),
                handler=self._tool_codex_subscription_connect_complete,
            ),
            ToolDefinition(
                name="smr_codex_subscription_disconnect",
                description="Disconnect the global Codex subscription from SMR.",
                input_schema=_tool_schema({}, required=[]),
                handler=self._tool_codex_subscription_disconnect,
            ),
            ToolDefinition(
                name="smr_github_org_status",
                description="Get org-level GitHub integration status.",
                input_schema=_tool_schema({}, required=[]),
                handler=self._tool_github_org_status,
            ),
            ToolDefinition(
                name="smr_github_org_oauth_start",
                description="Start org-level GitHub OAuth flow.",
                input_schema=_tool_schema(
                    {
                        "redirect_uri": {
                            "type": "string",
                            "description": "Optional callback URL override.",
                        }
                    },
                    required=[],
                ),
                handler=self._tool_github_org_oauth_start,
            ),
            ToolDefinition(
                name="smr_github_org_oauth_callback",
                description="Complete org-level GitHub OAuth callback.",
                input_schema=_tool_schema(
                    {
                        "code": {
                            "type": "string",
                            "description": "OAuth callback code.",
                        },
                        "state": {
                            "type": "string",
                            "description": "Optional OAuth state value.",
                        },
                        "redirect_uri": {
                            "type": "string",
                            "description": "Optional callback URL override.",
                        },
                    },
                    required=["code"],
                ),
                handler=self._tool_github_org_oauth_callback,
            ),
            ToolDefinition(
                name="smr_github_org_disconnect",
                description="Disconnect org-level GitHub integration.",
                input_schema=_tool_schema({}, required=[]),
                handler=self._tool_github_org_disconnect,
            ),
            ToolDefinition(
                name="smr_github_org_pat_connect",
                description="Connect org-level GitHub PAT credential.",
                input_schema=_tool_schema(
                    {
                        "pat": {
                            "type": "string",
                            "description": "GitHub PAT value.",
                        }
                    },
                    required=["pat"],
                ),
                handler=self._tool_github_org_pat_connect,
            ),
            ToolDefinition(
                name="smr_github_project_pat_connect",
                description="Connect project-level GitHub PAT credential.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "pat": {
                            "type": "string",
                            "description": "GitHub PAT value.",
                        },
                        "repo": {
                            "type": "string",
                            "description": "Optional default repo in owner/name format.",
                        },
                        "pr_write_enabled": {
                            "type": "boolean",
                            "description": "Require push permission for selected repo.",
                        },
                    },
                    required=["project_id", "pat"],
                ),
                handler=self._tool_github_project_pat_connect,
            ),
            ToolDefinition(
                name="smr_linear_status",
                description="Get project-level Linear integration status.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        }
                    },
                    required=["project_id"],
                ),
                handler=self._tool_linear_status,
            ),
            ToolDefinition(
                name="smr_linear_oauth_start",
                description="Start project-level Linear OAuth flow.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "redirect_uri": {
                            "type": "string",
                            "description": "Optional callback URL override.",
                        },
                    },
                    required=["project_id"],
                ),
                handler=self._tool_linear_oauth_start,
            ),
            ToolDefinition(
                name="smr_linear_oauth_callback",
                description="Complete project-level Linear OAuth callback.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "code": {
                            "type": "string",
                            "description": "OAuth callback code.",
                        },
                        "state": {
                            "type": "string",
                            "description": "Optional OAuth state value.",
                        },
                        "redirect_uri": {
                            "type": "string",
                            "description": "Optional callback URL override.",
                        },
                    },
                    required=["project_id", "code"],
                ),
                handler=self._tool_linear_oauth_callback,
            ),
            ToolDefinition(
                name="smr_linear_disconnect",
                description="Disconnect project-level Linear integration.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        }
                    },
                    required=["project_id"],
                ),
                handler=self._tool_linear_disconnect,
            ),
            ToolDefinition(
                name="smr_linear_list_teams",
                description="List Linear teams available to the project integration.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        }
                    },
                    required=["project_id"],
                ),
                handler=self._tool_linear_list_teams,
            ),
            ToolDefinition(
                name="smr_set_execution_preferences",
                description="Set execution lane preferences for a project.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "preferred_lane": {
                            "type": "string",
                            "enum": ["auto", "synth_hosted", "user_connected"],
                            "description": "Preferred execution lane.",
                        },
                        "allow_fallback_to_synth": {
                            "type": "boolean",
                            "description": "Allow fallback to synth-hosted lane.",
                        },
                        "free_tier_eligible": {
                            "type": "boolean",
                            "description": "Mark project eligible for free-tier synth hosted lane.",
                        },
                        "monthly_soft_limit_tokens": {
                            "type": "integer",
                            "description": "Optional monthly soft token limit.",
                        },
                    },
                    required=["project_id", "preferred_lane"],
                ),
                handler=self._tool_set_execution_preferences,
            ),
            ToolDefinition(
                name="smr_get_capacity_lane_preview",
                description="Preview resolved execution lane for a project.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        }
                    },
                    required=["project_id"],
                ),
                handler=self._tool_get_capacity_lane_preview,
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
                name="smr_get_project_repos",
                description="List project-scoped GitHub repos configured in integrations.github.repos.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        }
                    },
                    required=["project_id"],
                ),
                handler=self._tool_get_project_repos,
            ),
            ToolDefinition(
                name="smr_link_org_github",
                description="Link a project to the org-level GitHub credential.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        }
                    },
                    required=["project_id"],
                ),
                handler=self._tool_link_org_github,
            ),
            ToolDefinition(
                name="smr_add_project_repo",
                description="Add a GitHub repo to a project with optional PR write enablement.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "repo": {
                            "type": "string",
                            "description": "Repo in owner/name format.",
                        },
                        "pr_write_enabled": {
                            "type": "boolean",
                            "description": "Whether PR creation should be enabled for this repo.",
                        },
                    },
                    required=["project_id", "repo"],
                ),
                handler=self._tool_add_project_repo,
            ),
            ToolDefinition(
                name="smr_remove_project_repo",
                description="Remove a GitHub repo from a project.",
                input_schema=_tool_schema(
                    {
                        "project_id": {
                            "type": "string",
                            "description": "Managed research project id.",
                        },
                        "repo": {
                            "type": "string",
                            "description": "Repo in owner/name format.",
                        },
                    },
                    required=["project_id", "repo"],
                ),
                handler=self._tool_remove_project_repo,
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
                name="smr_get_artifact_content",
                description=(
                    "Download artifact content by artifact id. Returns UTF-8 text when possible, "
                    "otherwise base64-encoded bytes."
                ),
                input_schema=_tool_schema(
                    {
                        "artifact_id": {
                            "type": "string",
                            "description": "Artifact id.",
                        },
                        "disposition": {
                            "type": "string",
                            "description": "Either 'inline' or 'attachment'.",
                        },
                        "max_bytes": {
                            "type": "integer",
                            "description": "Maximum bytes to return in the response (default 200000).",
                        },
                    },
                    required=["artifact_id"],
                ),
                handler=self._tool_get_artifact_content,
            ),
            ToolDefinition(
                name="smr_list_run_pull_requests",
                description="List pull requests created for a run via github_pr artifacts.",
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
                        "limit": {
                            "type": "integer",
                            "description": "Maximum PR artifacts to inspect (default 100).",
                        },
                    },
                    required=["run_id"],
                ),
                handler=self._tool_list_run_pull_requests,
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

    def _tool_get_binding(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        run_id = _optional_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.get_binding(project_id, run_id=run_id)

    def _tool_promote_binding(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        pool_id = _require_string(args, "pool_id")
        dataset_revision = _require_string(args, "dataset_revision")
        expected_revision = _optional_int(args, "expected_revision")
        if expected_revision is None:
            raise ValueError("'expected_revision' is required")
        runtime_kind = _optional_string(args, "runtime_kind")
        environment_kind = _optional_string(args, "environment_kind")
        published_by_run_id = _optional_string(args, "published_by_run_id")
        reason = _optional_string(args, "reason")
        idempotency_key = _optional_string(args, "idempotency_key")
        with self._client_from_args(args) as client:
            return client.promote_binding(
                project_id,
                pool_id=pool_id,
                dataset_revision=dataset_revision,
                expected_revision=expected_revision,
                runtime_kind=runtime_kind,
                environment_kind=environment_kind,
                published_by_run_id=published_by_run_id,
                reason=reason,
                idempotency_key=idempotency_key,
            )

    def _tool_get_pool_context(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        run_id = _optional_string(args, "run_id")
        task_id = _optional_string(args, "task_id")
        with self._client_from_args(args) as client:
            return client.get_pool_context(project_id, run_id=run_id, task_id=task_id)

    def _tool_get_starting_data_upload_urls(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        dataset_ref = _optional_string(args, "dataset_ref")
        idempotency_key_upload = _optional_string(
            args, "idempotency_key_upload"
        ) or _optional_string(args, "idempotency_key")
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
                idempotency_key_upload=idempotency_key_upload,
            )

    def _tool_upload_starting_data(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        dataset_ref = _optional_string(args, "dataset_ref")
        idempotency_key_upload = _optional_string(
            args, "idempotency_key_upload"
        ) or _optional_string(args, "idempotency_key")
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
                idempotency_key_upload=idempotency_key_upload,
            )

    def _tool_trigger_run(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        timebox_seconds = _optional_int(args, "timebox_seconds")
        agent_model = _optional_string(args, "agent_model")
        agent_kind = _optional_string(args, "agent_kind")
        workflow = _optional_object(args, "workflow")
        idempotency_key_run_create = _optional_string(
            args, "idempotency_key_run_create"
        ) or _optional_string(args, "idempotency_key")
        with self._client_from_args(args) as client:
            return client.trigger_run(
                project_id,
                timebox_seconds=timebox_seconds,
                agent_model=agent_model,
                agent_kind=agent_kind,
                workflow=workflow,
                idempotency_key_run_create=idempotency_key_run_create,
            )

    def _tool_trigger_data_factory(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        dataset_ref = _require_string(args, "dataset_ref")
        bundle_manifest_path = _require_string(args, "bundle_manifest_path")
        profile = _optional_string(args, "profile") or "founder_default"
        source_mode = _optional_string(args, "source_mode") or "synth_mcp_local"
        template = _optional_string(args, "template")
        preferred_target = _optional_string(args, "preferred_target") or "harbor"
        strictness_mode = _optional_string(args, "strictness_mode") or "warn"
        timebox_seconds = _optional_int(args, "timebox_seconds")

        raw_targets = args.get("targets")
        targets: list[str] | None = None
        if raw_targets is not None:
            if not isinstance(raw_targets, list) or not raw_targets:
                raise ValueError("'targets' must be a non-empty array when provided")
            parsed_targets: list[str] = []
            for value in raw_targets:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("each targets entry must be a non-empty string")
                parsed_targets.append(value.strip())
            targets = parsed_targets

        runtime_kind = _optional_string(args, "runtime_kind")
        environment_kind = _optional_string(args, "environment_kind")
        session_id = _optional_string(args, "session_id")
        session_state = _optional_string(args, "session_state")
        session_title = _optional_string(args, "session_title")
        session_notes = _optional_string(args, "session_notes")
        idempotency_key_run_create = _optional_string(
            args, "idempotency_key_run_create"
        ) or _optional_string(args, "idempotency_key")

        with self._client_from_args(args) as client:
            return client.trigger_data_factory_run(
                project_id,
                dataset_ref=dataset_ref,
                bundle_manifest_path=bundle_manifest_path,
                template=template,
                profile=profile,
                source_mode=source_mode,
                targets=targets,
                preferred_target=preferred_target,
                runtime_kind=runtime_kind,
                environment_kind=environment_kind,
                session_id=session_id,
                session_state=session_state,
                session_title=session_title,
                session_notes=session_notes,
                strictness_mode=strictness_mode,
                timebox_seconds=timebox_seconds,
                idempotency_key_run_create=idempotency_key_run_create,
            )

    def _tool_data_factory_finalize(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        dataset_ref = _optional_string(args, "dataset_ref") or "starting-data"
        bundle_manifest_path = (
            _optional_string(args, "bundle_manifest_path") or "capture_bundle.json"
        )
        template = _optional_string(args, "template")
        finalizer_profile = _optional_string(args, "finalizer_profile") or "founder_default"
        source_mode = _optional_string(args, "source_mode") or "synth_mcp_local"
        preferred_target = _optional_string(args, "preferred_target") or "harbor"
        strictness_mode = _optional_string(args, "strictness_mode") or "warn"
        runtime_kind = _optional_string(args, "runtime_kind")
        environment_kind = _optional_string(args, "environment_kind")
        timebox_seconds = _optional_int(args, "timebox_seconds")
        idempotency_key_run_create = _optional_string(
            args, "idempotency_key_run_create"
        ) or _optional_string(args, "idempotency_key")

        raw_target_formats = args.get("target_formats")
        target_formats: list[str] | None = None
        if raw_target_formats is not None:
            if not isinstance(raw_target_formats, list) or not raw_target_formats:
                raise ValueError("'target_formats' must be a non-empty array when provided")
            parsed_target_formats: list[str] = []
            for value in raw_target_formats:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("each target_formats entry must be a non-empty string")
                parsed_target_formats.append(value.strip())
            target_formats = parsed_target_formats

        with self._client_from_args(args) as client:
            return client.data_factory_finalize(
                project_id,
                dataset_ref=dataset_ref,
                bundle_manifest_path=bundle_manifest_path,
                template=template,
                target_formats=target_formats,
                preferred_target=preferred_target,
                finalizer_profile=finalizer_profile,
                source_mode=source_mode,
                runtime_kind=runtime_kind,
                environment_kind=environment_kind,
                strictness_mode=strictness_mode,
                timebox_seconds=timebox_seconds,
                idempotency_key_run_create=idempotency_key_run_create,
            )

    def _tool_data_factory_finalize_status(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        job_id = _require_string(args, "job_id")
        with self._client_from_args(args) as client:
            return client.data_factory_finalize_status(project_id, job_id)

    def _tool_data_factory_publish(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        job_id = _require_string(args, "job_id")
        reason = _optional_string(args, "reason") or "manual_publish"
        idempotency_key_publish = _optional_string(
            args, "idempotency_key_publish"
        ) or _optional_string(args, "idempotency_key")
        with self._client_from_args(args) as client:
            return client.data_factory_publish(
                project_id,
                job_id,
                reason=reason,
                idempotency_key_publish=idempotency_key_publish,
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

    def _tool_list_jobs(self, args: JSONDict) -> Any:
        project_id = _optional_string(args, "project_id")
        state = _optional_string(args, "state")
        active_only = _optional_bool(args, "active_only", default=False)
        limit = _optional_int(args, "limit") or 50
        with self._client_from_args(args) as client:
            return client.list_jobs(
                project_id=project_id,
                state=state,
                active_only=active_only,
                limit=limit,
            )

    def _tool_get_run(self, args: JSONDict) -> Any:
        run_id = _require_string(args, "run_id")
        project_id = _optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_run(run_id, project_id=project_id)

    def _tool_get_run_usage(self, args: JSONDict) -> Any:
        run_id = _require_string(args, "run_id")
        project_id = _optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_run_usage(run_id, project_id=project_id)

    def _tool_get_actor_status(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        run_id = _optional_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.get_actor_status(project_id, run_id=run_id)

    def _tool_control_actor(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        run_id = _require_string(args, "run_id")
        actor_id = _require_string(args, "actor_id")
        action = _require_string(args, "action")
        if action not in {"pause", "resume"}:
            raise ValueError("'action' must be 'pause' or 'resume'")
        reason = _optional_string(args, "reason")
        idempotency_key = _optional_string(args, "idempotency_key")
        with self._client_from_args(args) as client:
            return client.control_actor(
                project_id,
                run_id,
                actor_id,
                action=action,  # type: ignore[arg-type]
                reason=reason,
                idempotency_key=idempotency_key,
            )

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

    def _tool_codex_subscription_status(self, args: JSONDict) -> Any:
        project_id = _optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.chatgpt_connection_status(project_id=project_id)

    def _tool_codex_subscription_connect_start(self, args: JSONDict) -> Any:
        sandbox_agent_url = _optional_string(args, "sandbox_agent_url")
        provider_id = _optional_string(args, "provider_id")
        external_account_hint = _optional_string(args, "external_account_hint")
        with self._client_from_args(args) as client:
            return client.chatgpt_connect_start(
                sandbox_agent_url=sandbox_agent_url,
                provider_id=provider_id,
                external_account_hint=external_account_hint,
            )

    def _tool_codex_subscription_connect_complete(self, args: JSONDict) -> Any:
        code = _optional_string(args, "code")
        sandbox_agent_url = _optional_string(args, "sandbox_agent_url")
        with self._client_from_args(args) as client:
            return client.chatgpt_connect_complete(code=code, sandbox_agent_url=sandbox_agent_url)

    def _tool_codex_subscription_disconnect(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return client.chatgpt_disconnect()

    def _tool_github_org_status(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return client.github_org_status()

    def _tool_github_org_oauth_start(self, args: JSONDict) -> Any:
        redirect_uri = _optional_string(args, "redirect_uri")
        with self._client_from_args(args) as client:
            return client.github_org_oauth_start(redirect_uri=redirect_uri)

    def _tool_github_org_oauth_callback(self, args: JSONDict) -> Any:
        code = _require_string(args, "code")
        state = _optional_string(args, "state")
        redirect_uri = _optional_string(args, "redirect_uri")
        with self._client_from_args(args) as client:
            return client.github_org_oauth_callback(
                code=code,
                state=state,
                redirect_uri=redirect_uri,
            )

    def _tool_github_org_disconnect(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return client.github_org_disconnect()

    def _tool_github_org_pat_connect(self, args: JSONDict) -> Any:
        pat = _require_string(args, "pat")
        with self._client_from_args(args) as client:
            return client.github_org_pat_connect(pat=pat)

    def _tool_github_project_pat_connect(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        pat = _require_string(args, "pat")
        repo = _optional_string(args, "repo")
        pr_write_enabled = _optional_bool(args, "pr_write_enabled", default=False)
        with self._client_from_args(args) as client:
            return client.github_pat_connect(
                project_id=project_id,
                pat=pat,
                repo=repo,
                pr_write_enabled=pr_write_enabled,
            )

    def _tool_linear_status(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.linear_status(project_id)

    def _tool_linear_oauth_start(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        redirect_uri = _optional_string(args, "redirect_uri")
        with self._client_from_args(args) as client:
            return client.linear_oauth_start(project_id=project_id, redirect_uri=redirect_uri)

    def _tool_linear_oauth_callback(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        code = _require_string(args, "code")
        state = _optional_string(args, "state")
        redirect_uri = _optional_string(args, "redirect_uri")
        with self._client_from_args(args) as client:
            return client.linear_oauth_callback(
                project_id=project_id,
                code=code,
                state=state,
                redirect_uri=redirect_uri,
            )

    def _tool_linear_disconnect(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.linear_disconnect(project_id)

    def _tool_linear_list_teams(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.linear_list_teams(project_id)

    def _tool_set_execution_preferences(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        preferred_lane = _require_string(args, "preferred_lane")
        if preferred_lane not in {"auto", "synth_hosted", "user_connected"}:
            raise ValueError("'preferred_lane' must be one of: auto, synth_hosted, user_connected")
        allow_fallback_to_synth = args.get("allow_fallback_to_synth")
        if allow_fallback_to_synth is not None and not isinstance(allow_fallback_to_synth, bool):
            raise ValueError("'allow_fallback_to_synth' must be a boolean when provided")
        free_tier_eligible = args.get("free_tier_eligible")
        if free_tier_eligible is not None and not isinstance(free_tier_eligible, bool):
            raise ValueError("'free_tier_eligible' must be a boolean when provided")
        monthly_soft_limit_tokens = args.get("monthly_soft_limit_tokens")
        if monthly_soft_limit_tokens is not None and not isinstance(monthly_soft_limit_tokens, int):
            raise ValueError("'monthly_soft_limit_tokens' must be an integer when provided")
        with self._client_from_args(args) as client:
            return client.set_execution_preferences(
                project_id,
                preferred_lane=preferred_lane,  # type: ignore[arg-type]
                allow_fallback_to_synth=allow_fallback_to_synth,
                free_tier_eligible=free_tier_eligible,
                monthly_soft_limit_tokens=monthly_soft_limit_tokens,
            )

    def _tool_get_capacity_lane_preview(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_capacity_lane_preview(project_id)

    # Project lifecycle mutations ---------------------------------------

    def _tool_create_project(self, args: JSONDict) -> Any:
        name = _require_string(args, "name")
        config = args.get("config") or {}
        if not isinstance(config, dict):
            raise ValueError("'config' must be a JSON object when provided")
        payload: JSONDict = {"name": name, **config}
        with self._client_from_args(args) as client:
            return client.create_project(payload)

    def _tool_get_project_repos(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_project_repos(project_id)

    def _tool_link_org_github(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.link_org_github(project_id)

    def _tool_add_project_repo(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        repo = _require_string(args, "repo")
        pr_write_enabled = _optional_bool(args, "pr_write_enabled", default=False)
        with self._client_from_args(args) as client:
            return client.add_project_repo(
                project_id,
                repo=repo,
                pr_write_enabled=pr_write_enabled,
            )

    def _tool_remove_project_repo(self, args: JSONDict) -> Any:
        project_id = _require_string(args, "project_id")
        repo = _require_string(args, "repo")
        with self._client_from_args(args) as client:
            return client.remove_project_repo(project_id, repo=repo)

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

    def _tool_get_artifact_content(self, args: JSONDict) -> Any:
        artifact_id = _require_string(args, "artifact_id")
        disposition = _optional_string(args, "disposition") or "inline"
        if disposition not in {"inline", "attachment"}:
            raise ValueError("'disposition' must be 'inline' or 'attachment'")
        max_bytes_raw = _optional_int(args, "max_bytes")
        max_bytes = max_bytes_raw if max_bytes_raw is not None else 200_000
        if max_bytes <= 0:
            raise ValueError("'max_bytes' must be a positive integer")

        with self._client_from_args(args) as client:
            artifact = client.get_artifact(artifact_id)
            response = client.get_artifact_content_response(
                artifact_id,
                disposition=disposition,
                follow_redirects=True,
            )
            content_bytes = response.content or b""

        full_size = len(content_bytes)
        truncated = full_size > max_bytes
        payload_bytes = content_bytes[:max_bytes] if truncated else content_bytes

        try:
            text_content = payload_bytes.decode("utf-8")
            encoding = "utf-8"
            content: str = text_content
        except UnicodeDecodeError:
            encoding = "base64"
            content = base64.b64encode(payload_bytes).decode("ascii")

        return {
            "artifact_id": artifact_id,
            "artifact_type": artifact.get("artifact_type"),
            "title": artifact.get("title"),
            "uri": artifact.get("uri"),
            "content_type": response.headers.get("content-type"),
            "encoding": encoding,
            "content": content,
            "content_bytes_returned": len(payload_bytes),
            "content_bytes_total": full_size,
            "truncated": truncated,
            "max_bytes": max_bytes,
        }

    def _tool_list_run_pull_requests(self, args: JSONDict) -> Any:
        run_id = _require_string(args, "run_id")
        project_id = _optional_string(args, "project_id")
        limit_raw = _optional_int(args, "limit")
        limit = limit_raw if limit_raw is not None else 100
        if limit <= 0:
            raise ValueError("'limit' must be a positive integer")
        with self._client_from_args(args) as client:
            return client.list_run_pull_requests(
                run_id,
                project_id=project_id,
                limit=limit,
            )

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

    def _tool_health_check(self, args: JSONDict) -> Any:
        project_id = _optional_string(args, "project_id")
        backend_base = _optional_string(args, "backend_base")
        api_key_override = _optional_string(args, "api_key")
        resolved_backend_base = (
            backend_base
            or os.environ.get("SYNTH_BACKEND_URL", "").strip()
            or "https://api.usesynth.ai"
        )
        try:
            resolved_api_key = api_key_override or get_api_key("SYNTH_API_KEY", required=False)
        except Exception:
            resolved_api_key = api_key_override or os.environ.get("SYNTH_API_KEY", "").strip()
        api_key_present = bool(resolved_api_key)
        tools = self.available_tool_names()
        checks: JSONDict = {
            "api_key": {
                "status": "pass" if api_key_present else "fail",
                "message": (
                    "Synth API key available."
                    if api_key_present
                    else "No Synth API key was resolved from args, environment, or Synth config."
                ),
                "hint": (
                    None
                    if api_key_present
                    else "Run `synth-ai setup` or export SYNTH_API_KEY before launching Codex."
                ),
            },
            "mcp_server": {
                "status": "pass",
                "server_name": "synth-ai-managed-research",
                "server_version": __version__,
                "protocol_version": DEFAULT_PROTOCOL_VERSION,
                "supported_protocol_versions": list(SUPPORTED_PROTOCOL_VERSIONS),
                "tool_count": len(tools),
            },
        }
        ok = api_key_present

        try:
            with self._client_from_args(args) as client:
                capabilities = client.get_capabilities()
                backend_check: JSONDict = {
                    "status": "pass",
                    "backend_url": resolved_backend_base,
                    "capability_keys": sorted(capabilities.keys())
                    if isinstance(capabilities, dict)
                    else [],
                }
                if isinstance(capabilities, dict):
                    for key in ("version", "backend_version", "api_version", "build_sha"):
                        value = capabilities.get(key)
                        if isinstance(value, str) and value.strip():
                            backend_check["backend_version"] = value.strip()
                            break
                checks["backend_ping"] = backend_check

                projects = client.list_projects(limit=1)
                checks["project_access"] = {
                    "status": "pass",
                    "project_count_sampled": len(projects),
                }

                if project_id:
                    project_status = client.get_project_status(project_id)
                    checks["project_status"] = {
                        "status": "pass",
                        "project_id": project_id,
                        "project_status_keys": sorted(project_status.keys())
                        if isinstance(project_status, dict)
                        else [],
                    }
        except Exception as exc:
            checks["backend_ping"] = {
                "status": "fail",
                "backend_url": resolved_backend_base,
                "message": f"{type(exc).__name__}: {exc}",
                "hint": "Verify SYNTH_API_KEY, SYNTH_BACKEND_URL, and backend availability.",
            }
            checks.setdefault(
                "project_access",
                {
                    "status": "fail",
                    "message": "Project access check skipped after backend failure.",
                },
            )
            ok = False
        else:
            ok = ok and True

        return {
            "ok": ok
            and all(
                isinstance(check, dict) and check.get("status") == "pass"
                for check in checks.values()
            ),
            "backend_url": resolved_backend_base,
            "project_id": project_id,
            "checks": checks,
            "tooling": {
                "mcp_server": "synth-ai-managed-research",
                "mcp_server_version": __version__,
            },
            "recommended_next_steps": [
                "Run synth-ai mcp codex install to register this MCP server with Codex.",
                "Run synth-ai setup or export SYNTH_API_KEY before launching Codex.",
            ],
        }

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


def _jsonrpc_error(
    request_id: Any, code: int, message: str, *, data: Any | None = None
) -> JSONDict:
    error: JSONDict = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def _read_message(stdin: Any) -> tuple[JSONDict | None, str | None]:
    headers: dict[str, str] = {}

    while True:
        line = stdin.readline()
        if line == b"":
            return None, None
        stripped = line.strip()
        if not stripped:
            if headers:
                break
            continue
        # Codex CLI sends newline-delimited JSON-RPC requests on stdio during MCP startup.
        # Accept that form in addition to Content-Length framed messages.
        if not headers and stripped[:1] in (b"{", b"["):
            try:
                message = json.loads(stripped.decode("utf-8"))
            except Exception as exc:
                raise ValueError(f"Invalid JSON payload: {exc}") from exc
            if not isinstance(message, dict):
                raise ValueError("JSON-RPC message must be a JSON object")
            return message, "jsonl"
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
    return message, "content-length"


def _write_message(stdout: Any, message: JSONDict, mode: str = "content-length") -> None:
    payload = json.dumps(message, separators=(",", ":"), default=str).encode("utf-8")
    if mode == "jsonl":
        stdout.write(payload + b"\n")
    else:
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
        stdout.write(header)
        stdout.write(payload)
    stdout.flush()


def run_stdio_server() -> None:
    """Run the MCP server using stdio transport."""
    server = ManagedResearchMcpServer()
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    response_mode = "content-length"

    while True:
        try:
            message, message_mode = _read_message(stdin)
            if message_mode is not None:
                response_mode = message_mode
        except Exception as exc:
            _write_message(
                stdout, _jsonrpc_error(None, -32700, f"Parse error: {exc}"), response_mode
            )
            continue

        if message is None:
            break

        method = message.get("method")
        request_id = message.get("id")

        if isinstance(method, str) and request_id is None:
            server.handle_notification(
                method, message.get("params") if isinstance(message.get("params"), dict) else None
            )
            continue

        if isinstance(method, str):
            _write_message(stdout, server.handle_request(message), response_mode)
            continue

        _write_message(
            stdout,
            _jsonrpc_error(message.get("id"), -32600, "Invalid request"),
            response_mode,
        )


def main() -> None:
    """CLI entrypoint for the managed-research MCP server."""
    run_stdio_server()


__all__ = ["ManagedResearchMcpServer", "main", "run_stdio_server"]


if __name__ == "__main__":
    main()
