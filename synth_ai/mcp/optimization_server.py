"""Optimization MCP server (stdio transport).

This server exposes stateful offline optimization controls as MCP tools so
agents can submit candidates and manage trial/rollout queues through synth-ai.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Callable

from synth_ai import __version__
from synth_ai.client import SynthClient

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
    "timeout_seconds": {
        "type": "number",
        "description": "Optional request timeout in seconds (default 30).",
    },
    "api_version": {
        "type": "string",
        "enum": ["v1", "v2"],
        "description": "Optimization API version (default v2).",
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


def _optional_float(payload: JSONDict, key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{key}' must be a number when provided")
    return float(value)


def _optional_object(payload: JSONDict, key: str) -> JSONDict | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be an object when provided")
    return value


def _require_object(payload: JSONDict, key: str) -> JSONDict:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' is required and must be an object")
    return value


def _require_array_of_objects(payload: JSONDict, key: str) -> list[JSONDict]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"'{key}' is required and must be a non-empty array")
    parsed: list[JSONDict] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"'{key}' entries must be objects")
        parsed.append(dict(item))
    return parsed


def _require_array_of_strings(payload: JSONDict, key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"'{key}' is required and must be a non-empty array")
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"'{key}' entries must be non-empty strings")
        parsed.append(item.strip())
    return parsed


def _api_version(args: JSONDict) -> str:
    version = (_optional_string(args, "api_version") or "v2").strip().lower()
    if version not in {"v1", "v2"}:
        raise ValueError("'api_version' must be one of: v1, v2")
    return version


class OptimizationMcpServer:
    """Minimal MCP server for stateful offline optimization control."""

    def __init__(self) -> None:
        self._tools = {tool.name: tool for tool in self._build_tools()}

    def _client_from_args(self, args: JSONDict) -> SynthClient:
        timeout = _optional_float(args, "timeout_seconds") or 30.0
        return SynthClient(
            api_key=_optional_string(args, "api_key"),
            base_url=_optional_string(args, "backend_base"),
            timeout=timeout,
        )

    def _offline_job(self, args: JSONDict):
        client = self._client_from_args(args)
        job_id = _require_string(args, "job_id")
        return client.optimization.offline.get(job_id, api_version=_api_version(args))

    def _build_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="opt_offline_create_job",
                description="Create a new offline optimization job.",
                input_schema=_tool_schema(
                    {
                        "kind": {
                            "type": "string",
                            "enum": ["gepa_offline", "mipro_offline", "eval"],
                            "description": "Offline job kind.",
                        },
                        "system_name": {
                            "type": "string",
                            "description": "Logical system name for the job.",
                        },
                        "config": {
                            "type": "object",
                            "description": "Optimization config payload.",
                        },
                        "system_id": {
                            "type": "string",
                            "description": "Optional existing system id to target.",
                        },
                        "reuse_system": {
                            "type": "boolean",
                            "description": "Whether to reuse an existing system (default true).",
                        },
                        "config_mode": {
                            "type": "string",
                            "enum": ["DEFAULT", "FULL"],
                            "description": "Config mode (default DEFAULT).",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional metadata payload.",
                        },
                        "auto_start": {
                            "type": "boolean",
                            "description": "Whether to auto-start execution (default true).",
                        },
                        "container_worker_token": {
                            "type": "string",
                            "description": "Optional worker token override.",
                        },
                        "prompt_opt_version": {
                            "type": "string",
                            "enum": ["v1", "v2"],
                            "description": "Prompt optimization version contract (default v2).",
                        },
                        "prompt_opt_fallback_policy": {
                            "type": "string",
                            "enum": ["none", "preflight_only", "init_only", "preflight_or_init"],
                            "description": "Optional fallback policy.",
                        },
                    },
                    required=["kind", "system_name", "config"],
                ),
                handler=self._tool_offline_create_job,
            ),
            ToolDefinition(
                name="opt_offline_list_jobs",
                description="List offline optimization jobs.",
                input_schema=_tool_schema(
                    {
                        "state": {"type": "string", "description": "Optional state filter."},
                        "kind": {
                            "type": "string",
                            "enum": ["gepa_offline", "mipro_offline", "eval"],
                        },
                        "system_id": {"type": "string"},
                        "system_name": {"type": "string"},
                        "created_after": {"type": "string"},
                        "created_before": {"type": "string"},
                        "limit": {"type": "integer"},
                        "cursor": {"type": "string"},
                    },
                    required=[],
                ),
                handler=self._tool_offline_list_jobs,
            ),
            ToolDefinition(
                name="opt_offline_get_status",
                description="Fetch current status for an offline optimization job.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                    },
                    required=["job_id"],
                ),
                handler=self._tool_offline_get_status,
            ),
            ToolDefinition(
                name="opt_offline_get_events",
                description="Fetch sequenced events for an offline optimization job.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                        "since_seq": {
                            "type": "integer",
                            "description": "First sequence to return (default 0).",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of events (default 500).",
                        },
                    },
                    required=["job_id"],
                ),
                handler=self._tool_offline_get_events,
            ),
            ToolDefinition(
                name="opt_offline_submit_candidates",
                description="Submit typed candidate envelopes into the stateful offline job.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                        "algorithm_kind": {
                            "type": "string",
                            "enum": ["gepa", "mipro"],
                            "description": "Candidate algorithm discriminator.",
                        },
                        "candidates": {
                            "type": "array",
                            "items": {"type": "object"},
                            "minItems": 1,
                            "description": (
                                "Typed candidate envelopes "
                                "(gepa_prompt_candidate or mipro_transform_candidate)."
                            ),
                        },
                        "proposal_session_id": {
                            "type": "string",
                            "description": "Optional proposer session identifier.",
                        },
                        "proposer_metadata": {
                            "type": "object",
                            "description": "Optional metadata attached to submission.",
                        },
                    },
                    required=["job_id", "algorithm_kind", "candidates"],
                ),
                handler=self._tool_offline_submit_candidates,
            ),
            ToolDefinition(
                name="opt_offline_get_state_baseline_info",
                description="Get compact baseline_info projection from persisted state envelope.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                    },
                    required=["job_id"],
                ),
                handler=self._tool_offline_get_state_baseline_info,
            ),
            ToolDefinition(
                name="opt_offline_get_state_envelope",
                description="Get full persisted state envelope for an offline optimization job.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                    },
                    required=["job_id"],
                ),
                handler=self._tool_offline_get_state_envelope,
            ),
            ToolDefinition(
                name="opt_offline_list_trial_queue",
                description="List trial queue state for an offline optimization job.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                    },
                    required=["job_id"],
                ),
                handler=self._tool_offline_list_trial_queue,
            ),
            ToolDefinition(
                name="opt_offline_enqueue_trial",
                description="Enqueue a TrialSpec into the offline job trial queue.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                        "trial": {
                            "type": "object",
                            "description": "TrialSpec payload.",
                        },
                        "algorithm_kind": {
                            "type": "string",
                            "enum": ["gepa", "mipro"],
                            "description": "Optional algorithm kind override.",
                        },
                    },
                    required=["job_id", "trial"],
                ),
                handler=self._tool_offline_enqueue_trial,
            ),
            ToolDefinition(
                name="opt_offline_update_trial",
                description="Patch an existing TrialSpec entry in the offline job trial queue.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                        "trial_id": {"type": "string", "description": "Trial id to patch."},
                        "patch": {"type": "object", "description": "Trial patch payload."},
                        "algorithm_kind": {
                            "type": "string",
                            "enum": ["gepa", "mipro"],
                            "description": "Optional algorithm kind override.",
                        },
                    },
                    required=["job_id", "trial_id", "patch"],
                ),
                handler=self._tool_offline_update_trial,
            ),
            ToolDefinition(
                name="opt_offline_cancel_trial",
                description="Cancel a trial in the offline job trial queue.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                        "trial_id": {"type": "string", "description": "Trial id to cancel."},
                        "algorithm_kind": {
                            "type": "string",
                            "enum": ["gepa", "mipro"],
                            "description": "Optional algorithm kind override.",
                        },
                    },
                    required=["job_id", "trial_id"],
                ),
                handler=self._tool_offline_cancel_trial,
            ),
            ToolDefinition(
                name="opt_offline_reorder_trials",
                description="Reorder trial queue by trial id list.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                        "trial_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": "Ordered list of trial ids.",
                        },
                        "algorithm_kind": {
                            "type": "string",
                            "enum": ["gepa", "mipro"],
                            "description": "Optional algorithm kind override.",
                        },
                    },
                    required=["job_id", "trial_ids"],
                ),
                handler=self._tool_offline_reorder_trials,
            ),
            ToolDefinition(
                name="opt_offline_apply_default_trial_plan",
                description="Apply default trial planner for the target algorithm.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                        "algorithm_kind": {
                            "type": "string",
                            "enum": ["gepa", "mipro"],
                            "description": "Optional algorithm kind override.",
                        },
                    },
                    required=["job_id"],
                ),
                handler=self._tool_offline_apply_default_trial_plan,
            ),
            ToolDefinition(
                name="opt_offline_get_rollout_queue",
                description="Fetch persisted rollout queue view for an offline optimization job.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                    },
                    required=["job_id"],
                ),
                handler=self._tool_offline_get_rollout_queue,
            ),
            ToolDefinition(
                name="opt_offline_set_rollout_queue_policy",
                description="Patch rollout queue scheduler policy for an offline job.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                        "policy_patch": {
                            "type": "object",
                            "description": (
                                "Rollout queue policy patch "
                                "(semaphore_policy, rate_limit_policy, retry_policy, dispatcher_status)."
                            ),
                        },
                        "algorithm_kind": {
                            "type": "string",
                            "enum": ["gepa", "mipro"],
                            "description": "Optional algorithm kind override.",
                        },
                    },
                    required=["job_id", "policy_patch"],
                ),
                handler=self._tool_offline_set_rollout_queue_policy,
            ),
            ToolDefinition(
                name="opt_offline_get_rollout_dispatch_metrics",
                description="Get dispatch-level rollout queue metrics for an offline job.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                    },
                    required=["job_id"],
                ),
                handler=self._tool_offline_get_rollout_dispatch_metrics,
            ),
            ToolDefinition(
                name="opt_offline_get_rollout_limiter_status",
                description="Get rollout scheduler limiter/runtime status for an offline job.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                    },
                    required=["job_id"],
                ),
                handler=self._tool_offline_get_rollout_limiter_status,
            ),
            ToolDefinition(
                name="opt_offline_retry_rollout_dispatch",
                description="Retry a rollout dispatch item for an offline job.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                        "dispatch_id": {"type": "string", "description": "Dispatch id."},
                        "algorithm_kind": {
                            "type": "string",
                            "enum": ["gepa", "mipro"],
                            "description": "Optional algorithm kind override.",
                        },
                    },
                    required=["job_id", "dispatch_id"],
                ),
                handler=self._tool_offline_retry_rollout_dispatch,
            ),
            ToolDefinition(
                name="opt_offline_drain_rollout_queue",
                description="Set rollout queue draining mode and optionally cancel queued dispatches.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                        "cancel_queued": {
                            "type": "boolean",
                            "description": "Cancel queued dispatches while draining (default false).",
                        },
                        "algorithm_kind": {
                            "type": "string",
                            "enum": ["gepa", "mipro"],
                            "description": "Optional algorithm kind override.",
                        },
                    },
                    required=["job_id"],
                ),
                handler=self._tool_offline_drain_rollout_queue,
            ),
            ToolDefinition(
                name="opt_offline_pause",
                description="Pause an offline optimization job.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                    },
                    required=["job_id"],
                ),
                handler=self._tool_offline_pause,
            ),
            ToolDefinition(
                name="opt_offline_resume",
                description="Resume a paused offline optimization job.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                    },
                    required=["job_id"],
                ),
                handler=self._tool_offline_resume,
            ),
            ToolDefinition(
                name="opt_offline_cancel",
                description="Cancel an offline optimization job.",
                input_schema=_tool_schema(
                    {
                        "job_id": {"type": "string", "description": "Offline job id."},
                    },
                    required=["job_id"],
                ),
                handler=self._tool_offline_cancel,
            ),
        ]

    # Tool handlers -----------------------------------------------------

    def _tool_offline_create_job(self, args: JSONDict) -> Any:
        client = self._client_from_args(args)
        job = client.optimization.offline.create(
            kind=_require_string(args, "kind"),
            system_name=_require_string(args, "system_name"),
            config=_require_object(args, "config"),
            system_id=_optional_string(args, "system_id"),
            reuse_system=_optional_bool(args, "reuse_system", default=True),
            config_mode=_optional_string(args, "config_mode") or "DEFAULT",
            metadata=_optional_object(args, "metadata"),
            auto_start=_optional_bool(args, "auto_start", default=True),
            container_worker_token=_optional_string(args, "container_worker_token"),
            prompt_opt_version=_optional_string(args, "prompt_opt_version") or "v2",
            prompt_opt_fallback_policy=_optional_string(args, "prompt_opt_fallback_policy"),
            api_version=_api_version(args),
        )
        return {
            "job_id": job.job_id,
            "system_id": job.system_id,
            "system_name": job.system_name,
            "status": job.status(),
        }

    def _tool_offline_list_jobs(self, args: JSONDict) -> Any:
        client = self._client_from_args(args)
        return client.optimization.offline.list(
            state=_optional_string(args, "state"),
            kind=_optional_string(args, "kind"),
            system_id=_optional_string(args, "system_id"),
            system_name=_optional_string(args, "system_name"),
            created_after=_optional_string(args, "created_after"),
            created_before=_optional_string(args, "created_before"),
            limit=_optional_int(args, "limit") or 100,
            cursor=_optional_string(args, "cursor"),
            api_version=_api_version(args),
        )

    def _tool_offline_get_status(self, args: JSONDict) -> Any:
        return self._offline_job(args).status()

    def _tool_offline_get_events(self, args: JSONDict) -> Any:
        return self._offline_job(args).events(
            since_seq=_optional_int(args, "since_seq") or 0,
            limit=_optional_int(args, "limit") or 500,
        )

    def _tool_offline_submit_candidates(self, args: JSONDict) -> Any:
        return self._offline_job(args).submit_candidates(
            algorithm_kind=_require_string(args, "algorithm_kind"),
            candidates=_require_array_of_objects(args, "candidates"),
            proposal_session_id=_optional_string(args, "proposal_session_id"),
            proposer_metadata=_optional_object(args, "proposer_metadata"),
        )

    def _tool_offline_get_state_baseline_info(self, args: JSONDict) -> Any:
        return self._offline_job(args).get_state_baseline_info()

    def _tool_offline_get_state_envelope(self, args: JSONDict) -> Any:
        return self._offline_job(args).get_state_envelope()

    def _tool_offline_list_trial_queue(self, args: JSONDict) -> Any:
        return self._offline_job(args).list_trial_queue()

    def _tool_offline_enqueue_trial(self, args: JSONDict) -> Any:
        return self._offline_job(args).enqueue_trial(
            trial=_require_object(args, "trial"),
            algorithm_kind=_optional_string(args, "algorithm_kind"),
        )

    def _tool_offline_update_trial(self, args: JSONDict) -> Any:
        return self._offline_job(args).update_trial(
            _require_string(args, "trial_id"),
            patch=_require_object(args, "patch"),
            algorithm_kind=_optional_string(args, "algorithm_kind"),
        )

    def _tool_offline_cancel_trial(self, args: JSONDict) -> Any:
        return self._offline_job(args).cancel_trial(
            _require_string(args, "trial_id"),
            algorithm_kind=_optional_string(args, "algorithm_kind"),
        )

    def _tool_offline_reorder_trials(self, args: JSONDict) -> Any:
        return self._offline_job(args).reorder_trials(
            trial_ids=_require_array_of_strings(args, "trial_ids"),
            algorithm_kind=_optional_string(args, "algorithm_kind"),
        )

    def _tool_offline_apply_default_trial_plan(self, args: JSONDict) -> Any:
        return self._offline_job(args).apply_default_trial_plan(
            algorithm_kind=_optional_string(args, "algorithm_kind"),
        )

    def _tool_offline_get_rollout_queue(self, args: JSONDict) -> Any:
        return self._offline_job(args).get_rollout_queue()

    def _tool_offline_set_rollout_queue_policy(self, args: JSONDict) -> Any:
        return self._offline_job(args).set_rollout_queue_policy(
            policy_patch=_require_object(args, "policy_patch"),
            algorithm_kind=_optional_string(args, "algorithm_kind"),
        )

    def _tool_offline_get_rollout_dispatch_metrics(self, args: JSONDict) -> Any:
        return self._offline_job(args).get_rollout_dispatch_metrics()

    def _tool_offline_get_rollout_limiter_status(self, args: JSONDict) -> Any:
        return self._offline_job(args).get_rollout_limiter_status()

    def _tool_offline_retry_rollout_dispatch(self, args: JSONDict) -> Any:
        return self._offline_job(args).retry_rollout_dispatch(
            _require_string(args, "dispatch_id"),
            algorithm_kind=_optional_string(args, "algorithm_kind"),
        )

    def _tool_offline_drain_rollout_queue(self, args: JSONDict) -> Any:
        return self._offline_job(args).drain_rollout_queue(
            cancel_queued=_optional_bool(args, "cancel_queued", default=False),
            algorithm_kind=_optional_string(args, "algorithm_kind"),
        )

    def _tool_offline_pause(self, args: JSONDict) -> Any:
        return self._offline_job(args).pause()

    def _tool_offline_resume(self, args: JSONDict) -> Any:
        return self._offline_job(args).resume()

    def _tool_offline_cancel(self, args: JSONDict) -> Any:
        return self._offline_job(args).cancel()

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
                "name": "synth-ai-optimization",
                "version": __version__,
            },
            "instructions": (
                "Use tools to control stateful offline optimization jobs. "
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
    server = OptimizationMcpServer()
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
            server.handle_notification(
                method, message.get("params") if isinstance(message.get("params"), dict) else None
            )
            continue

        if isinstance(method, str):
            _write_message(stdout, server.handle_request(message))
            continue

        _write_message(stdout, _jsonrpc_error(message.get("id"), -32600, "Invalid request"))


def main() -> None:
    """CLI entrypoint for the optimization MCP server."""
    run_stdio_server()


__all__ = ["OptimizationMcpServer", "main", "run_stdio_server"]


if __name__ == "__main__":
    main()
