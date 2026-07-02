"""Managed Research MCP server (stdio transport)."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from typing import Any, cast

from synth_ai.managed_research.auth import get_api_key
from synth_ai.managed_research.errors import SmrApiError
from synth_ai.managed_research.mcp.objective_tools import (
    ObjectiveToolOperation,
    objective_tool_operation_from_wire,
)
from synth_ai.managed_research.mcp.registry import (
    READ_SCOPES,
    WRITE_SCOPES,
    JSONDict,
    ToolDefinition,
    build_tool_registry,
    call_tool,
    list_tool_payload,
    tool_schema,
)
from synth_ai.managed_research.mcp.request_models import (
    OneOffRunLaunchRequest,
    ProjectMutationRequest,
    ProviderKeyRequest,
    RunLaunchRequest,
    RunnableProjectCreateRequest,
    WorkspaceFileUploadRequest,
    optional_bool,
    optional_int,
    optional_string,
    parse_branch_run_request,
    require_string,
)
from synth_ai.managed_research.mcp.tools.approvals import build_approval_tools
from synth_ai.managed_research.mcp.tools.artifacts import build_artifact_tools
from synth_ai.managed_research.mcp.tools.datasets import build_dataset_tools
from synth_ai.managed_research.mcp.tools.dev_environments import build_dev_environment_tools
from synth_ai.managed_research.mcp.tools.exports import build_export_tools
from synth_ai.managed_research.mcp.tools.factories import build_factory_tools
from synth_ai.managed_research.mcp.tools.files import build_file_tools
from synth_ai.managed_research.mcp.tools.integrations import build_integration_tools
from synth_ai.managed_research.mcp.tools.logs import build_log_tools
from synth_ai.managed_research.mcp.tools.models import build_model_tools
from synth_ai.managed_research.mcp.tools.open_research import build_open_research_tools
from synth_ai.managed_research.mcp.tools.outputs import build_output_tools
from synth_ai.managed_research.mcp.tools.progress import build_progress_tools
from synth_ai.managed_research.mcp.tools.projects import build_project_tools
from synth_ai.managed_research.mcp.tools.prs import build_pr_tools
from synth_ai.managed_research.mcp.tools.readiness import build_readiness_tools
from synth_ai.managed_research.mcp.tools.repos import build_repo_tools
from synth_ai.managed_research.mcp.tools.resources import build_resource_tools
from synth_ai.managed_research.mcp.tools.runs import build_run_tools
from synth_ai.managed_research.mcp.tools.tag import build_tag_tools
from synth_ai.managed_research.mcp.tools.trained_models import build_trained_model_tools
from synth_ai.managed_research.mcp.tools.usage import build_usage_tools
from synth_ai.managed_research.mcp.tools.workspace_inputs import build_workspace_input_tools
from synth_ai.managed_research.models.run_control import ManagedResearchActorControlAction
from synth_ai.managed_research.open_research import (
    OpenResearchClient,
    OpenResearchError,
    SubmitQuestionArgs,
    load_or_create_fingerprint,
)
from synth_ai.managed_research.open_research.models import ExperimentStatusFilter
from synth_ai.managed_research.sdk.client import ManagedResearchClient
from synth_ai.managed_research.version import __version__

SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2024-11-05")
DEFAULT_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]
SERVER_NAME = "managed-research"


def _mcp_structured_trigger_error_payload(exc: SmrApiError) -> dict[str, Any]:
    """Shape every ``SmrApiError`` as structured MCP error data."""
    detail = getattr(exc, "detail", None)
    detail_dict: dict[str, Any] = dict(detail) if isinstance(detail, dict) else {}
    code_raw = detail_dict.get("error_code")
    code = code_raw.strip() if isinstance(code_raw, str) and code_raw.strip() else "smr_api_error"
    out: dict[str, Any] = {
        "error": code,
        "detail": detail_dict,
        "message": str(exc),
    }
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        out["http_status"] = status
    return out


def _raise_mcp_tool_denial(exc: SmrApiError) -> None:
    payload = _mcp_structured_trigger_error_payload(exc)
    raise RpcError(
        -32010,
        payload.get("message", str(exc)),
        data=payload,
    ) from exc


def _raise_open_research_error(exc: OpenResearchError) -> None:
    """Surface a typed Open Research envelope as a structured RPC error.

    The contract's typed ``class`` and required ``actionable`` text flow
    through unchanged so MCP callers branch on them — mirrors
    ``feedback_informative_errors.md`` (no collapsed error messages).
    """
    raise RpcError(
        -32011,
        str(exc),
        data=exc.to_mcp_payload(),
    ) from exc


def _tool_body(args: JSONDict, *, exclude: set[str]) -> dict[str, Any]:
    return {
        key: value
        for key, value in args.items()
        if key not in {*exclude, "api_key", "backend_base"}
    }


_DEV_ENVIRONMENT_RUN_REJECTED_ARGS = frozenset(
    {
        "local_execution",
        "execution_profile",
        "sandbox_override",
        "environment",
    }
)


def _reject_dev_environment_run_substrate_args(args: JSONDict) -> None:
    rejected = sorted(
        field
        for field in _DEV_ENVIRONMENT_RUN_REJECTED_ARGS
        if field in args and args[field] is not None
    )
    if rejected:
        raise ValueError(
            "DevEnvironment run tools bind the existing cloud sandbox and reject "
            "local/ad hoc substrate fields: " + ", ".join(rejected)
        )


def _optional_object_arg(args: JSONDict, key: str) -> dict[str, Any] | None:
    value = args.get(key)
    return value if isinstance(value, dict) else None


def _object_arg(args: JSONDict, key: str) -> dict[str, Any] | None:
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be an object when provided")
    return dict(value)


def _object_list_arg(args: JSONDict, key: str) -> list[dict[str, Any]]:
    value = args.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"'{key}' must be an array of objects when provided")
    return [dict(item) for item in value]


def _optional_string_tuple_arg(args: JSONDict, key: str) -> tuple[str, ...]:
    value = args.get(key)
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _mcp_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_mcp_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_mcp_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _mcp_jsonable(item) for key, item in value.items()}
    return value


class RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def _read_message(stream: Any) -> tuple[JSONDict, str] | None:
    """Read either JSONL or content-length framed JSON-RPC from stdin."""

    first = stream.readline()
    if not first:
        return None
    if first.startswith(b"Content-Length:"):
        length = int(first.decode("ascii").split(":", 1)[1].strip())
        while True:
            line = stream.readline()
            if line in {b"\r\n", b"\n", b""}:
                break
        payload = json.loads(stream.read(length).decode("utf-8"))
        return payload, "content-length"
    payload = json.loads(first.decode("utf-8"))
    return payload, "jsonl"


def _write_message(stream: Any, payload: JSONDict, *, framing: str) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    if framing == "content-length":
        stream.write(f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii"))
        stream.write(encoded)
    else:
        stream.write(encoded + b"\n")
    stream.flush()


class ManagedResearchMcpServer:
    """Managed Research MCP server for the authenticated noun-first API surface."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        backend_base: str | None = None,
    ) -> None:
        self._default_api_key = api_key
        self._default_backend_base = backend_base
        self._tools = build_tool_registry(self._build_tools())

    def available_tool_names(self) -> list[str]:
        names = sorted(self._tools.keys())
        research_names = [name for name in names if name.startswith("research_")]
        smr_names = [name for name in names if name.startswith("smr_")]
        other_names = [name for name in names if not name.startswith(("research_", "smr_"))]
        return research_names + other_names + smr_names

    def tool_definitions(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def get_tool_definition(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tool_payload(self) -> list[JSONDict]:
        return list_tool_payload(self._tools)

    def call_tool(self, name: str, arguments: JSONDict | None = None) -> Any:
        return call_tool(self._tools, name, arguments)

    def _client_from_args(self, args: JSONDict) -> ManagedResearchClient:
        resolved_api_key = optional_string(args, "api_key") or self._default_api_key
        resolved_backend_base = optional_string(args, "backend_base") or self._default_backend_base
        return ManagedResearchClient(
            api_key=resolved_api_key,
            backend_base=resolved_backend_base,
        )

    @staticmethod
    def _removed_backend_contract(surface: str) -> None:
        raise SmrApiError(
            f"{surface} is not available in the current Managed Research backend contract.",
            failure_class="unsupported_backend_contract",
            remediation=(
                "Use the runtime-managed run, workspace, objective-event, work-graph, "
                "trace, and work-product surfaces that remain in the public contract."
            ),
        )

    def _build_tools(self) -> list[ToolDefinition]:
        return [
            *build_project_tools(self),
            *build_factory_tools(self),
            *build_dev_environment_tools(self),
            *build_workspace_input_tools(self),
            *build_export_tools(self),
            *build_repo_tools(self),
            *build_dataset_tools(self),
            *build_file_tools(self),
            *build_pr_tools(self),
            *build_model_tools(self),
            *build_open_research_tools(self),
            *build_output_tools(self),
            *build_readiness_tools(self),
            *build_resource_tools(self),
            *build_run_tools(self),
            *build_tag_tools(self),
            *build_progress_tools(self),
            *build_log_tools(self),
            *build_approval_tools(self),
            *build_artifact_tools(self),
            *build_integration_tools(self),
            *build_usage_tools(self),
            *build_trained_model_tools(self),
            *self._build_hosted_compat_tools(),
        ]

    def _build_hosted_compat_tools(self) -> list[ToolDefinition]:
        """Hosted-MCP compatibility aliases kept until clients move to noun-first names."""

        project_id_schema = {
            "project_id": {
                "type": "string",
                "description": "SMR project ID.",
            }
        }
        changeset_payload_schema = {
            "project_id": {"type": "string", "description": "SMR project ID."},
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "run_id": {"type": "string"},
            "source": {"type": "string"},
            "author_ref": {"type": "string"},
            "review_policy": {"type": "string"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target_kind": {"type": "string"},
                        "target_id": {"type": "string"},
                        "operation": {"type": "string"},
                        "proposed_payload": {"type": "object"},
                        "evidence_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["target_kind", "operation"],
                },
            },
            "idempotency_key": {"type": "string"},
            "metadata": {"type": "object"},
        }
        return [
            ToolDefinition(
                name="smr_capabilities_get",
                description="Read SMR capability metadata for the authenticated organization.",
                input_schema=tool_schema({}, required=[]),
                handler=self._tool_get_capabilities,
                required_scopes=READ_SCOPES,
            ),
            ToolDefinition(
                name="smr_projects_list",
                description="List managed research projects for the authenticated organization.",
                input_schema=tool_schema({}, required=[]),
                handler=self._tool_list_projects,
                required_scopes=READ_SCOPES,
            ),
            ToolDefinition(
                name="smr_project_status_get",
                description="Get high-level status and active run summary for a project.",
                input_schema=tool_schema(project_id_schema, required=["project_id"]),
                handler=self._tool_get_project_status,
                required_scopes=READ_SCOPES,
            ),
            ToolDefinition(
                name="smr_project_workspace_get",
                description=(
                    "Read the backend-owned project workspace projection: objectives, runs, "
                    "experiments, knowledge, review queue, reports, and launch risks. Runs "
                    "propose material; review or policy promotion owns durable project truth."
                ),
                input_schema=tool_schema(project_id_schema, required=["project_id"]),
                handler=self._tool_get_project_workspace,
                required_scopes=READ_SCOPES,
            ),
            ToolDefinition(
                name="smr_project_changesets_list",
                description="List review-gated project ChangeSets for a managed research project.",
                input_schema=tool_schema(
                    {
                        **project_id_schema,
                        "status": {
                            "type": "string",
                            "description": "Optional ChangeSet status filter.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum ChangeSets to return.",
                        },
                    },
                    required=["project_id"],
                ),
                handler=self._tool_list_project_changesets,
                required_scopes=READ_SCOPES,
            ),
            ToolDefinition(
                name="smr_project_changeset_create",
                description=(
                    "Create a proposed project ChangeSet. This stages project mutations; "
                    "it does not directly write durable project truth."
                ),
                input_schema=tool_schema(
                    changeset_payload_schema,
                    required=["project_id", "title", "items"],
                ),
                handler=self._tool_create_project_changeset,
                required_scopes=WRITE_SCOPES,
            ),
            ToolDefinition(
                name="smr_project_changeset_get",
                description="Fetch one review-gated project ChangeSet.",
                input_schema=tool_schema(
                    {
                        **project_id_schema,
                        "changeset_id": {"type": "string"},
                    },
                    required=["project_id", "changeset_id"],
                ),
                handler=self._tool_get_project_changeset,
                required_scopes=READ_SCOPES,
            ),
            ToolDefinition(
                name="smr_project_changeset_decide",
                description="Accept, promote, reject, supersede, or invalidate a proposed project ChangeSet.",
                input_schema=tool_schema(
                    {
                        **project_id_schema,
                        "changeset_id": {"type": "string"},
                        "decision": {
                            "type": "string",
                            "enum": [
                                "accepted",
                                "promoted",
                                "rejected",
                                "superseded",
                                "invalidated",
                            ],
                        },
                        "decided_by_ref": {"type": "string"},
                        "decision_reason": {"type": "string"},
                    },
                    required=["project_id", "changeset_id", "decision", "decided_by_ref"],
                ),
                handler=self._tool_decide_project_changeset,
                required_scopes=WRITE_SCOPES,
            ),
            ToolDefinition(
                name="smr_jobs_list",
                description="List SMR runs (jobs feed), optionally filtered by project, state, and active-only mode.",
                input_schema=tool_schema(
                    {
                        "project_id": {"type": "string"},
                        "state": {"type": "string"},
                        "active_only": {"type": "boolean"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                    },
                    required=[],
                ),
                handler=self._tool_jobs_list,
                required_scopes=READ_SCOPES,
            ),
            ToolDefinition(
                name="smr_project_trigger_run",
                description="Trigger a managed research run for a project.",
                input_schema=tool_schema(
                    {
                        "project_id": {"type": "string"},
                        "run_config": {
                            "type": "object",
                            "description": (
                                "Optional trigger body fields such as timebox_seconds, "
                                "agent_model, agent_kind, workflow, and "
                                "idempotency_key_run_create."
                            ),
                            "additionalProperties": True,
                        },
                    },
                    required=["project_id"],
                ),
                handler=self._tool_project_trigger_run,
                required_scopes=WRITE_SCOPES,
            ),
            ToolDefinition(
                name="smr_run_get",
                description="Get details for a specific SMR run by run_id.",
                input_schema=tool_schema(
                    {"run_id": {"type": "string"}},
                    required=["run_id"],
                ),
                handler=self._tool_get_run,
                required_scopes=READ_SCOPES,
            ),
            ToolDefinition(
                name="smr_project_run_actor_control",
                description=(
                    "Pause or resume one actor inside a project-scoped managed research run. "
                    "This is operator control, not project-truth promotion."
                ),
                input_schema=tool_schema(
                    {
                        "project_id": {"type": "string"},
                        "run_id": {"type": "string"},
                        "actor_id": {"type": "string"},
                        "action": {
                            "type": "string",
                            "enum": [item.value for item in ManagedResearchActorControlAction],
                        },
                        "reason": {"type": "string"},
                        "idempotency_key": {"type": "string"},
                    },
                    required=["project_id", "run_id", "actor_id", "action"],
                ),
                handler=self._tool_control_project_run_actor,
                required_scopes=WRITE_SCOPES,
            ),
            ToolDefinition(
                name="smr_run_stop",
                description="Stop a running SMR run.",
                input_schema=tool_schema(
                    {"run_id": {"type": "string"}},
                    required=["run_id"],
                ),
                handler=self._tool_stop_run,
                required_scopes=WRITE_SCOPES,
            ),
        ]

    def _tool_health_check(self, args: JSONDict) -> Any:
        project_id = optional_string(args, "project_id")
        checks: dict[str, Any] = {}
        try:
            api_key = optional_string(args, "api_key") or get_api_key(required=False)
            checks["api_key"] = {
                "status": "pass" if api_key else "warn",
                "configured": bool(api_key),
            }
        except ValueError as exc:
            checks["api_key"] = {
                "status": "fail",
                "configured": False,
                "message": str(exc),
            }
            api_key = None
        with self._client_from_args(args) as client:
            capabilities = client.get_capabilities()
            checks["backend_ping"] = {
                "status": "pass",
                "backend_version": str(capabilities.get("version") or __version__),
            }
            if project_id:
                checks["project_status"] = client.get_project_status(project_id)
        return {"ok": True, "checks": checks}

    def _tool_setup_exports_list_targets(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return client.list_export_targets()

    def _tool_setup_exports_create_target(self, args: JSONDict) -> Any:
        label = require_string(args, "label")
        bucket = require_string(args, "bucket")
        access_key_id = require_string(args, "access_key_id")
        secret_access_key = require_string(args, "secret_access_key")
        prefix = optional_string(args, "prefix")
        region = optional_string(args, "region")
        endpoint_url = optional_string(args, "endpoint_url")
        with self._client_from_args(args) as client:
            return client.create_export_target(
                {
                    "kind": "s3",
                    "label": label,
                    "config": {
                        "bucket": bucket,
                        "prefix": prefix,
                        "region": region,
                        "endpoint_url": endpoint_url,
                    },
                    "access_key_id": access_key_id,
                    "secret_access_key": secret_access_key,
                }
            )

    def _tool_setup_github_status(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return client.get_github_status()

    def _tool_setup_github_start_oauth(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return client.start_github_oauth(
                redirect_uri=optional_string(args, "redirect_uri"),
            )

    def _tool_setup_github_list_repos(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return client.list_github_repos(
                page=optional_int(args, "page"),
                per_page=optional_int(args, "per_page"),
            )

    def _tool_setup_github_disconnect(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return client.disconnect_github()

    def _tool_work_repos_list(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.list_project_repo_bindings(project_id)

    def _tool_work_repos_attach(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        github_repo = require_string(args, "github_repo")
        pr_write_enabled = optional_bool(args, "pr_write_enabled", default=False)
        with self._client_from_args(args) as client:
            return client.attach_project_repo(
                project_id,
                repo=github_repo,
                pr_write_enabled=pr_write_enabled,
            )

    def _tool_work_repos_detach(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        github_repo = require_string(args, "github_repo")
        with self._client_from_args(args) as client:
            return client.detach_project_repo(project_id, repo=github_repo)

    def _tool_work_datasets_list(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.list_project_datasets(project_id)

    def _tool_work_datasets_upload(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        name = require_string(args, "name")
        content = require_string(args, "content")
        payload: dict[str, Any] = {
            "name": name,
            "content": content,
            "encoding": optional_string(args, "encoding") or "text",
        }
        content_type = optional_string(args, "content_type")
        format_value = optional_string(args, "format")
        row_count = optional_int(args, "row_count")
        metadata = args.get("metadata")
        if content_type is not None:
            payload["content_type"] = content_type
        if format_value is not None:
            payload["format"] = format_value
        if row_count is not None:
            payload["row_count"] = row_count
        if isinstance(metadata, dict):
            payload["metadata"] = dict(metadata)
        with self._client_from_args(args) as client:
            return client.upload_project_dataset(project_id, payload)

    def _tool_work_datasets_download(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        dataset_id = require_string(args, "dataset_id")
        with self._client_from_args(args) as client:
            return client.download_project_dataset(project_id, dataset_id)

    def _tool_results_outputs_list(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.list_project_outputs(project_id)

    def _tool_list_run_work_products(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.list_run_work_products(project_id, run_id)

    def _tool_get_run_work_product(self, args: JSONDict) -> Any:
        work_product_id = require_string(args, "work_product_id")
        with self._client_from_args(args) as client:
            return client.get_run_work_product(work_product_id)

    def _tool_get_run_work_product_content(self, args: JSONDict) -> Any:
        work_product_id = require_string(args, "work_product_id")
        with self._client_from_args(args) as client:
            return client.get_run_work_product_content(work_product_id, as_text=True)

    def _tool_export_run_work_product(self, args: JSONDict) -> Any:
        work_product_id = require_string(args, "work_product_id")
        destination = args.get("destination")
        if not isinstance(destination, dict):
            raise ValueError("destination must be an object")
        with self._client_from_args(args) as client:
            return client.export_run_work_product(
                work_product_id,
                destination=destination,
                idempotency_key=optional_string(args, "idempotency_key"),
            )

    def _tool_explain_work_product_blocker(self, args: JSONDict) -> Any:
        work_product_id = require_string(args, "work_product_id")
        with self._client_from_args(args) as client:
            return client.work_products.explain_blocker(work_product_id)

    def _tool_upload_container_eval_package(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        manifest = args.get("manifest")
        metadata = args.get("metadata")
        if manifest is not None and not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        with self._client_from_args(args) as client:
            return client.work_products.upload_container_eval_package(
                project_id,
                run_id,
                kind=require_string(args, "kind"),
                name=require_string(args, "name"),
                version=optional_string(args, "version"),
                artifact_id=optional_string(args, "artifact_id"),
                storage_uri=optional_string(args, "storage_uri"),
                archive_size_bytes=optional_int(args, "archive_size_bytes"),
                manifest=manifest or {},
                metadata=metadata or {},
            )

    def _tool_validate_container_eval_package(self, args: JSONDict) -> Any:
        package_id = require_string(args, "package_id")
        with self._client_from_args(args) as client:
            return client.work_products.validate_container_eval_package(package_id)

    def _tool_results_prs_list(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.list_project_prs(project_id)

    def _tool_results_prs_get(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        pr_id = require_string(args, "pr_id")
        with self._client_from_args(args) as client:
            return client.get_project_pr(project_id, pr_id)

    def _tool_results_models_list(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.list_project_models(project_id)

    def _tool_results_models_get(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        model_id = require_string(args, "model_id")
        with self._client_from_args(args) as client:
            return client.get_project_model(project_id, model_id)

    def _tool_results_models_download(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        model_id = require_string(args, "model_id")
        with self._client_from_args(args) as client:
            return client.download_project_model(project_id, model_id)

    def _tool_results_models_export(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        model_id = require_string(args, "model_id")
        with self._client_from_args(args) as client:
            return client.export_project_model(project_id, model_id)

    def _tool_status_readiness(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_project_readiness(project_id)

    def _tool_create_runnable_project(self, args: JSONDict) -> Any:
        request = RunnableProjectCreateRequest.from_payload(args)
        with self._client_from_args(args) as client:
            return client.create_runnable_project(request.request)

    def _tool_create_project(self, args: JSONDict) -> Any:
        config = args.get("config")
        if config is not None and not isinstance(config, dict):
            raise ValueError("'config' must be an object when provided")
        return self._tool_create_runnable_project(args)

    def _tool_list_projects(self, args: JSONDict) -> Any:
        include_archived = optional_bool(args, "include_archived", default=False)
        limit = optional_int(args, "limit") or 100
        with self._client_from_args(args) as client:
            results = client.projects.list(
                include_archived=include_archived,
                limit=limit,
            )
            return [asdict(item) if is_dataclass(item) else item for item in results]

    def _tool_get_project(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            result = client.projects.get(project_id)
            return asdict(result) if is_dataclass(result) else result

    def _tool_get_default_project(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return client.get_default_project()

    def _tool_create_factory(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return client.factories.create(_tool_body(args, exclude=set())).raw

    def _tool_list_factories(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return [item.raw for item in client.factories.list()]

    def _tool_get_factory(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        with self._client_from_args(args) as client:
            return client.factories.get(factory_id).raw

    def _tool_patch_factory(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        with self._client_from_args(args) as client:
            return client.factories.patch(
                factory_id,
                _tool_body(args, exclude={"factory_id"}),
            ).raw

    def _tool_pause_factory(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        with self._client_from_args(args) as client:
            return client.factories.pause(factory_id).raw

    def _tool_resume_factory(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        with self._client_from_args(args) as client:
            return client.factories.resume(factory_id).raw

    def _tool_archive_factory(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        with self._client_from_args(args) as client:
            return client.factories.archive(factory_id).raw

    def _tool_get_factory_status(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        with self._client_from_args(args) as client:
            return client.factories.status(factory_id).raw

    def _link_factory_project_with_role(
        self,
        args: JSONDict,
        *,
        role: str | None = None,
        status: str | None = None,
    ) -> Any:
        factory_id = require_string(args, "factory_id")
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.factories.link_project(
                factory_id,
                project_id,
                role=role or str(args.get("role") or "canonical"),
                status=status or str(args.get("status") or "active"),
                display_name=optional_string(args, "display_name"),
                description=optional_string(args, "description"),
                workspace_policy=_optional_object_arg(args, "workspace_policy"),
                resource_bindings=_optional_object_arg(args, "resource_bindings"),
                feed_health=_optional_object_arg(args, "feed_health"),
                default_launch_profile=_optional_object_arg(args, "default_launch_profile"),
                metadata=_optional_object_arg(args, "metadata"),
            ).raw

    def _tool_link_factory_project(self, args: JSONDict) -> Any:
        return self._link_factory_project_with_role(args)

    def _tool_link_factory_workspace_project(self, args: JSONDict) -> Any:
        return self._link_factory_project_with_role(args, role="canonical", status="active")

    def _tool_link_factory_auxiliary_project(self, args: JSONDict) -> Any:
        return self._link_factory_project_with_role(args, role="auxiliary", status="active")

    def _tool_list_factory_projects(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        with self._client_from_args(args) as client:
            return [
                item.raw
                for item in client.factories.list_projects(
                    factory_id,
                    include_archived=bool(args.get("include_archived")),
                )
            ]

    def _tool_get_factory_project(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.factories.get_project(factory_id, project_id).raw

    def _tool_patch_factory_project(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.factories.patch_project(
                factory_id,
                project_id,
                _tool_body(args, exclude={"factory_id", "project_id"}),
            ).raw

    def _tool_get_factory_workspace(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        with self._client_from_args(args) as client:
            return client.factories.workspace(
                factory_id,
                include_archived=bool(args.get("include_archived")),
            ).raw

    def _tool_create_factory_idea(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        with self._client_from_args(args) as client:
            return client.factories.create_idea(
                factory_id,
                title=require_string(args, "title"),
                body=optional_string(args, "body"),
                status=str(args.get("status") or "open"),
                source=str(args.get("source") or "human"),
                project_id=optional_string(args, "project_id"),
                effort_id=optional_string(args, "effort_id"),
                run_id=optional_string(args, "run_id"),
                priority=optional_string(args, "priority"),
                tags=_optional_string_tuple_arg(args, "tags"),
                promotion_target=_optional_object_arg(args, "promotion_target"),
                metadata=_optional_object_arg(args, "metadata"),
            ).raw

    def _tool_list_factory_ideas(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        with self._client_from_args(args) as client:
            return [
                item.raw
                for item in client.factories.list_ideas(
                    factory_id,
                    status=optional_string(args, "status"),
                    source=optional_string(args, "source"),
                    include_archived=bool(args.get("include_archived")),
                    limit=optional_int(args, "limit") or 50,
                )
            ]

    def _tool_get_factory_idea(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        idea_id = require_string(args, "idea_id")
        with self._client_from_args(args) as client:
            return client.factories.get_idea(factory_id, idea_id).raw

    def _tool_patch_factory_idea(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        idea_id = require_string(args, "idea_id")
        with self._client_from_args(args) as client:
            return client.factories.patch_idea(
                factory_id,
                idea_id,
                _tool_body(args, exclude={"factory_id", "idea_id"}),
            ).raw

    def _tool_record_factory_actor_output(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        with self._client_from_args(args) as client:
            return client.factories.create_actor_output(
                factory_id,
                actor_role=require_string(args, "actor_role"),
                kind=require_string(args, "kind"),
                title=require_string(args, "title"),
                summary=optional_string(args, "summary"),
                status=str(args.get("status") or "draft"),
                project_id=optional_string(args, "project_id"),
                effort_id=optional_string(args, "effort_id"),
                run_id=optional_string(args, "run_id"),
                report_id=optional_string(args, "report_id"),
                work_product_id=optional_string(args, "work_product_id"),
                payload=_optional_object_arg(args, "payload"),
                metadata=_optional_object_arg(args, "metadata"),
            ).raw

    def _record_named_factory_actor_output(
        self,
        args: JSONDict,
        *,
        actor_role: str,
        kind: str,
    ) -> Any:
        factory_id = require_string(args, "factory_id")
        with self._client_from_args(args) as client:
            return client.factories.create_actor_output(
                factory_id,
                actor_role=actor_role,
                kind=kind,
                title=require_string(args, "title"),
                summary=optional_string(args, "summary"),
                status=str(args.get("status") or "draft"),
                project_id=optional_string(args, "project_id"),
                effort_id=optional_string(args, "effort_id"),
                run_id=optional_string(args, "run_id"),
                report_id=optional_string(args, "report_id"),
                work_product_id=optional_string(args, "work_product_id"),
                payload=_optional_object_arg(args, "payload"),
                metadata=_optional_object_arg(args, "metadata"),
            ).raw

    def _tool_record_seraph_brief(self, args: JSONDict) -> Any:
        return self._record_named_factory_actor_output(
            args,
            actor_role="seraph",
            kind="seraph_brief",
        )

    def _tool_record_gardener_digest(self, args: JSONDict) -> Any:
        return self._record_named_factory_actor_output(
            args,
            actor_role="gardener",
            kind="gardener_digest",
        )

    def _tool_record_architect_feed_health(self, args: JSONDict) -> Any:
        return self._record_named_factory_actor_output(
            args,
            actor_role="architect",
            kind="architect_feed_health",
        )

    def _tool_list_factory_actor_outputs(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        with self._client_from_args(args) as client:
            return [
                item.raw
                for item in client.factories.list_actor_outputs(
                    factory_id,
                    actor_role=optional_string(args, "actor_role"),
                    kind=optional_string(args, "kind"),
                    status=optional_string(args, "status"),
                    include_archived=bool(args.get("include_archived")),
                    limit=optional_int(args, "limit") or 50,
                )
            ]

    def _tool_get_factory_actor_output(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        actor_output_id = require_string(args, "actor_output_id")
        with self._client_from_args(args) as client:
            return client.factories.get_actor_output(factory_id, actor_output_id).raw

    def _tool_patch_factory_actor_output(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        actor_output_id = require_string(args, "actor_output_id")
        with self._client_from_args(args) as client:
            return client.factories.patch_actor_output(
                factory_id,
                actor_output_id,
                _tool_body(args, exclude={"factory_id", "actor_output_id"}),
            ).raw

    def _tool_list_factory_open_decisions(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        with self._client_from_args(args) as client:
            return [item.raw for item in client.factories.list_open_decisions(factory_id)]

    def _tool_wake_due_factory_efforts(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        launch_request = args.get("launch_request")
        with self._client_from_args(args) as client:
            return client.factories.wake_due(
                factory_id,
                launch_request=launch_request if isinstance(launch_request, dict) else None,
                limit=optional_int(args, "limit") or 10,
                allow_overlap=bool(args.get("allow_overlap")),
                dry_run=bool(args.get("dry_run")),
                continue_on_error=bool(args.get("continue_on_error", True)),
            ).raw

    def _tool_list_factory_efforts(self, args: JSONDict) -> Any:
        factory_id = require_string(args, "factory_id")
        with self._client_from_args(args) as client:
            return [item.raw for item in client.factories.list_efforts(factory_id)]

    def _tool_create_effort(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return client.efforts.create(_tool_body(args, exclude=set())).raw

    def _tool_get_effort(self, args: JSONDict) -> Any:
        effort_id = require_string(args, "effort_id")
        with self._client_from_args(args) as client:
            return client.efforts.get(effort_id).raw

    def _tool_patch_effort(self, args: JSONDict) -> Any:
        effort_id = require_string(args, "effort_id")
        with self._client_from_args(args) as client:
            return client.efforts.patch(
                effort_id,
                _tool_body(args, exclude={"effort_id"}),
            ).raw

    def _tool_pause_effort(self, args: JSONDict) -> Any:
        effort_id = require_string(args, "effort_id")
        with self._client_from_args(args) as client:
            return client.efforts.pause(effort_id).raw

    def _tool_resume_effort(self, args: JSONDict) -> Any:
        effort_id = require_string(args, "effort_id")
        with self._client_from_args(args) as client:
            return client.efforts.resume(effort_id).raw

    def _tool_mark_effort_waiting(self, args: JSONDict) -> Any:
        effort_id = require_string(args, "effort_id")
        with self._client_from_args(args) as client:
            return client.efforts.mark_waiting(
                effort_id,
                next_wake_at=args.get("next_wake_at"),
                note=args.get("note"),
            ).raw

    def _tool_schedule_effort(self, args: JSONDict) -> Any:
        effort_id = require_string(args, "effort_id")
        next_wake_at = require_string(args, "next_wake_at")
        recurrence_policy = args.get("recurrence_policy")
        launch_request = args.get("launch_request")
        with self._client_from_args(args) as client:
            return client.efforts.schedule(
                effort_id,
                next_wake_at=next_wake_at,
                recurrence_policy=(
                    recurrence_policy if isinstance(recurrence_policy, dict) else None
                ),
                launch_request=launch_request if isinstance(launch_request, dict) else None,
            ).raw

    def _tool_mark_effort_ready_for_review(self, args: JSONDict) -> Any:
        effort_id = require_string(args, "effort_id")
        with self._client_from_args(args) as client:
            return client.efforts.mark_ready_for_review(
                effort_id,
                note=args.get("note"),
            ).raw

    def _tool_resolve_effort_decision(self, args: JSONDict) -> Any:
        effort_id = require_string(args, "effort_id")
        with self._client_from_args(args) as client:
            return client.efforts.resolve_decision(
                effort_id,
                note=args.get("note"),
            ).raw

    def _tool_launch_effort(self, args: JSONDict) -> Any:
        effort_id = require_string(args, "effort_id")
        objective = args.get("objective")
        launch_args = {
            key: value
            for key, value in args.items()
            if key not in {"effort_id", "objective", "api_key", "backend_base"}
        }
        with self._client_from_args(args) as client:
            handle = client.efforts.launch(
                effort_id,
                objective=str(objective) if objective is not None else None,
                **launch_args,
            )
            return {"project_id": handle.project_id, "run_id": handle.run_id}

    def _tool_rename_project(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        name = require_string(args, "name").strip()
        if not name:
            raise ValueError("'name' must be non-empty")
        with self._client_from_args(args) as client:
            return client.rename_project(project_id, name)

    def _tool_patch_project(self, args: JSONDict) -> Any:
        request = ProjectMutationRequest.for_patch(args)
        with self._client_from_args(args) as client:
            return client.patch_project(
                request.project_id,
                request.config,
                actor_model_assignments=request.actor_model_assignments,
            )

    def _tool_get_project_status(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_project_status(project_id)

    def _tool_get_project_workspace(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_project_workspace(project_id)

    def _tool_list_project_experiments(self, args: JSONDict) -> Any:
        self._removed_backend_contract("Project experiment list")

    def _tool_create_project_experiment(self, args: JSONDict) -> Any:
        self._removed_backend_contract("Project experiment creation")

    def _tool_get_project_experiment(self, args: JSONDict) -> Any:
        self._removed_backend_contract("Project experiment read")

    def _tool_patch_project_experiment(self, args: JSONDict) -> Any:
        self._removed_backend_contract("Project experiment patch")

    def _tool_link_project_experiment_run(self, args: JSONDict) -> Any:
        self._removed_backend_contract("Project experiment run link")

    def _tool_list_project_experiment_runs(self, args: JSONDict) -> Any:
        self._removed_backend_contract("Project experiment run list")

    def _tool_attach_project_experiment_container_run(self, args: JSONDict) -> Any:
        self._removed_backend_contract("Project experiment container-run attachment")

    def _tool_list_project_experiment_container_runs(self, args: JSONDict) -> Any:
        self._removed_backend_contract("Project experiment container-run list")

    def _tool_attach_project_experiment_result(self, args: JSONDict) -> Any:
        self._removed_backend_contract("Project experiment result attachment")

    def _tool_list_project_experiment_results(self, args: JSONDict) -> Any:
        self._removed_backend_contract("Project experiment result list")

    def _tool_rank_project_experiment_results(self, args: JSONDict) -> Any:
        self._removed_backend_contract("Project experiment result ranking")

    def _tool_list_project_changesets(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        status = optional_string(args, "status")
        limit = optional_int(args, "limit")
        with self._client_from_args(args) as client:
            return client.list_project_changesets(
                project_id,
                status=status,
                limit=limit,
            )

    def _tool_create_project_changeset(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        payload = {key: value for key, value in args.items() if key != "project_id"}
        with self._client_from_args(args) as client:
            return client.create_project_changeset(project_id, payload)

    def _tool_get_project_changeset(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        changeset_id = require_string(args, "changeset_id")
        with self._client_from_args(args) as client:
            return client.get_project_changeset(project_id, changeset_id)

    def _tool_decide_project_changeset(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        changeset_id = require_string(args, "changeset_id")
        payload = {
            key: value for key, value in args.items() if key not in {"project_id", "changeset_id"}
        }
        with self._client_from_args(args) as client:
            return client.decide_project_changeset(project_id, changeset_id, payload)

    def _tool_get_project_entitlement(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_project_entitlement(project_id)

    def _tool_get_project_setup(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_project_setup(project_id)

    def _tool_prepare_project_setup(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.prepare_project_setup(project_id)

    def _tool_list_dev_environment_topologies(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return _mcp_jsonable(client.dev_environments.topologies())

    def _tool_get_dev_environment_topology(self, args: JSONDict) -> Any:
        topology_id = require_string(args, "topology_id")
        version = optional_string(args, "version")
        with self._client_from_args(args) as client:
            return _mcp_jsonable(
                client.dev_environments.topology(topology_id, version=version)
            )

    def _tool_seed_dev_environment_topology_manifest(self, args: JSONDict) -> Any:
        topology_id = optional_string(args, "topology_id") or "synth-dev"
        version = optional_string(args, "version")
        with self._client_from_args(args) as client:
            return _mcp_jsonable(
                client.dev_environments.seed_topology_environment(
                    topology_id=topology_id,
                    version=version,
                )
            )

    def _tool_list_dev_environments(self, args: JSONDict) -> Any:
        project_id = optional_string(args, "project_id")
        limit = optional_int(args, "limit")
        with self._client_from_args(args) as client:
            return _mcp_jsonable(
                client.dev_environments.list(project_id=project_id, limit=limit)
            )

    def _tool_list_dev_environment_materialization_queue(
        self,
        args: JSONDict,
    ) -> Any:
        project_id = optional_string(args, "project_id")
        host_kind = optional_string(args, "host_kind")
        backend_target = optional_string(args, "backend_target")
        worker_id = optional_string(args, "worker_id")
        include_leased = optional_bool(args, "include_leased")
        limit = optional_int(args, "limit")
        with self._client_from_args(args) as client:
            return _mcp_jsonable(
                client.dev_environments.materialization_queue(
                    project_id=project_id,
                    host_kind=host_kind,
                    backend_target=backend_target,
                    worker_id=worker_id,
                    include_leased=include_leased,
                    limit=limit,
                )
            )

    def _tool_create_dev_environment(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return _mcp_jsonable(
                client.dev_environments.create(
                    project_id=require_string(args, "project_id"),
                    name=require_string(args, "name"),
                    environment_name=require_string(args, "environment_name"),
                    backend_target=optional_string(args, "backend_target") or "dev",
                    topology_id=optional_string(args, "topology_id") or "synth-dev",
                    topology_version=optional_string(args, "topology_version"),
                    environment_digest=optional_string(args, "environment_digest"),
                    host_kind=optional_string(args, "host_kind") or "daytona",
                    quota_class=optional_string(args, "quota_class"),
                    metadata=_object_arg(args, "metadata"),
                    uptime_rate_microcents_per_hour=optional_int(
                        args,
                        "uptime_rate_microcents_per_hour",
                    ),
                    billing_model_class=optional_string(args, "billing_model_class"),
                )
            )

    def _tool_create_dev_environment_from_topology(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return _mcp_jsonable(
                client.dev_environments.create_from_topology(
                    project_id=require_string(args, "project_id"),
                    name=require_string(args, "name"),
                    backend_target=optional_string(args, "backend_target") or "dev",
                    topology_id=optional_string(args, "topology_id") or "synth-dev",
                    topology_version=optional_string(args, "topology_version"),
                    host_kind=optional_string(args, "host_kind") or "daytona",
                    quota_class=optional_string(args, "quota_class"),
                    metadata=_object_arg(args, "metadata"),
                    uptime_rate_microcents_per_hour=optional_int(
                        args,
                        "uptime_rate_microcents_per_hour",
                    ),
                    billing_model_class=optional_string(args, "billing_model_class"),
                )
            )

    def _tool_get_dev_environment(self, args: JSONDict) -> Any:
        dev_environment_id = require_string(args, "dev_environment_id")
        with self._client_from_args(args) as client:
            return _mcp_jsonable(client.dev_environments.get(dev_environment_id))

    def _tool_claim_dev_environment_materialization(self, args: JSONDict) -> Any:
        dev_environment_id = require_string(args, "dev_environment_id")
        worker_id = require_string(args, "worker_id")
        lease_seconds = optional_int(args, "lease_seconds")
        metadata = _object_arg(args, "metadata")
        with self._client_from_args(args) as client:
            return _mcp_jsonable(
                client.dev_environments.claim_materialization(
                    dev_environment_id,
                    worker_id=worker_id,
                    lease_seconds=lease_seconds,
                    metadata=metadata,
                )
            )

    def _tool_preflight_dev_environment(self, args: JSONDict) -> Any:
        dev_environment_id = require_string(args, "dev_environment_id")
        with self._client_from_args(args) as client:
            return _mcp_jsonable(client.dev_environments.preflight(dev_environment_id))

    def _tool_dev_environment_action(
        self,
        args: JSONDict,
        *,
        action: str,
    ) -> Any:
        dev_environment_id = require_string(args, "dev_environment_id")
        metadata = _object_arg(args, "metadata")
        with self._client_from_args(args) as client:
            namespace = client.dev_environments
            if action == "deploy":
                return _mcp_jsonable(namespace.deploy(dev_environment_id, metadata=metadata))
            if action == "start":
                return _mcp_jsonable(namespace.start(dev_environment_id, metadata=metadata))
            if action == "snapshot":
                return _mcp_jsonable(namespace.snapshot(dev_environment_id, metadata=metadata))
        raise ValueError(f"unsupported DevEnvironment action: {action}")

    def _tool_deploy_dev_environment(self, args: JSONDict) -> Any:
        return self._tool_dev_environment_action(args, action="deploy")

    def _tool_start_dev_environment(self, args: JSONDict) -> Any:
        return self._tool_dev_environment_action(args, action="start")

    def _tool_stop_dev_environment(self, args: JSONDict) -> Any:
        dev_environment_id = require_string(args, "dev_environment_id")
        decision = optional_string(args, "decision") or "retain"
        metadata = _object_arg(args, "metadata")
        with self._client_from_args(args) as client:
            return _mcp_jsonable(
                client.dev_environments.stop(
                    dev_environment_id,
                    decision=decision,
                    metadata=metadata,
                )
            )

    def _tool_snapshot_dev_environment(self, args: JSONDict) -> Any:
        return self._tool_dev_environment_action(args, action="snapshot")

    def _tool_report_dev_environment_materialization(self, args: JSONDict) -> Any:
        dev_environment_id = require_string(args, "dev_environment_id")
        with self._client_from_args(args) as client:
            return _mcp_jsonable(
                client.dev_environments.materialize(
                    dev_environment_id,
                    result=optional_string(args, "result") or "succeeded",
                    lifecycle_state=optional_string(args, "lifecycle_state"),
                    service_summary=_object_arg(args, "service_summary"),
                    log_entries=_object_list_arg(args, "log_entries"),
                    receipt_refs=_object_list_arg(args, "receipt_refs"),
                    metadata=_object_arg(args, "metadata"),
                    error=_object_arg(args, "error"),
                )
            )

    def _tool_destroy_dev_environment(self, args: JSONDict) -> Any:
        dev_environment_id = require_string(args, "dev_environment_id")
        with self._client_from_args(args) as client:
            return _mcp_jsonable(client.dev_environments.destroy(dev_environment_id))

    def _tool_get_dev_environment_services(self, args: JSONDict) -> Any:
        dev_environment_id = require_string(args, "dev_environment_id")
        with self._client_from_args(args) as client:
            return _mcp_jsonable(client.dev_environments.services(dev_environment_id))

    def _tool_get_dev_environment_attach(self, args: JSONDict) -> Any:
        dev_environment_id = require_string(args, "dev_environment_id")
        with self._client_from_args(args) as client:
            return _mcp_jsonable(client.dev_environments.attach(dev_environment_id))

    def _tool_get_dev_environment_logs(self, args: JSONDict) -> Any:
        dev_environment_id = require_string(args, "dev_environment_id")
        with self._client_from_args(args) as client:
            return _mcp_jsonable(client.dev_environments.logs(dev_environment_id))

    def _tool_get_dev_environment_runs(self, args: JSONDict) -> Any:
        dev_environment_id = require_string(args, "dev_environment_id")
        with self._client_from_args(args) as client:
            return _mcp_jsonable(client.dev_environments.runs(dev_environment_id))

    def _tool_get_dev_environment_usage(self, args: JSONDict) -> Any:
        dev_environment_id = require_string(args, "dev_environment_id")
        limit = optional_int(args, "limit")
        with self._client_from_args(args) as client:
            return _mcp_jsonable(
                client.dev_environments.usage(dev_environment_id, limit=limit)
            )

    def _tool_preflight_dev_environment_billing(self, args: JSONDict) -> Any:
        dev_environment_id = require_string(args, "dev_environment_id")
        model_class = optional_string(args, "model_class") or "value"
        estimated = optional_int(args, "estimated_customer_debit_microcents") or 0
        with self._client_from_args(args) as client:
            return _mcp_jsonable(
                client.dev_environments.billing_preflight(
                    dev_environment_id,
                    model_class=model_class,
                    estimated_customer_debit_microcents=estimated,
                )
            )

    def _tool_get_dev_environment_billing_drawdown(self, args: JSONDict) -> Any:
        dev_environment_id = require_string(args, "dev_environment_id")
        with self._client_from_args(args) as client:
            return _mcp_jsonable(
                client.dev_environments.billing_drawdown(dev_environment_id)
            )

    def _tool_get_dev_environment_receipts(self, args: JSONDict) -> Any:
        dev_environment_id = require_string(args, "dev_environment_id")
        with self._client_from_args(args) as client:
            return _mcp_jsonable(client.dev_environments.receipts(dev_environment_id))

    def _tool_get_dev_environment_evidence(self, args: JSONDict) -> Any:
        dev_environment_id = require_string(args, "dev_environment_id")
        usage_limit = optional_int(args, "usage_limit")
        include_preflight = optional_bool(args, "include_preflight", default=True)
        include_logs = optional_bool(args, "include_logs")
        include_billing = optional_bool(args, "include_billing", default=True)
        with self._client_from_args(args) as client:
            return _mcp_jsonable(
                client.dev_environments.evidence(
                    dev_environment_id,
                    usage_limit=usage_limit,
                    include_preflight=include_preflight,
                    include_logs=include_logs,
                    include_billing=include_billing,
                )
            )

    def _tool_get_project_notes(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_project_notes(project_id)

    def _tool_set_project_notes(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        notes = require_string(args, "notes")
        with self._client_from_args(args) as client:
            return client.set_project_notes(project_id, notes)

    def _tool_append_project_notes(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        notes = require_string(args, "notes")
        with self._client_from_args(args) as client:
            return client.append_project_notes(project_id, notes)

    def _tool_get_org_knowledge(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return client.get_org_knowledge()

    def _tool_set_org_knowledge(self, args: JSONDict) -> Any:
        content = require_string(args, "content")
        with self._client_from_args(args) as client:
            return client.set_org_knowledge(content)

    def _tool_get_project_knowledge(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_project_knowledge(project_id)

    def _tool_set_project_knowledge(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        content = require_string(args, "content")
        with self._client_from_args(args) as client:
            return client.set_project_knowledge(project_id, content)

    def _tool_curated_knowledge(self, args: JSONDict) -> Any:
        operation = require_string(args, "operation").strip().lower()
        if operation not in {"get", "set"}:
            raise ValueError("'operation' must be either 'get' or 'set'")
        scope = require_string(args, "scope").strip().lower()
        if scope not in {"org", "project"}:
            raise ValueError("'scope' must be either 'org' or 'project'")
        project_id = require_string(args, "project_id") if scope == "project" else None
        if scope == "org" and optional_string(args, "project_id") is not None:
            raise ValueError("'project_id' must not be set when scope is 'org'")

        with self._client_from_args(args) as client:
            if operation == "get":
                if scope == "org":
                    return client.get_org_knowledge()
                return client.get_project_knowledge(project_id)

            content = require_string(args, "content")
            if scope == "org":
                return client.set_org_knowledge(content)
            return client.set_project_knowledge(project_id, content)

    def _tool_pause_project(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.pause_project(project_id)

    def _tool_resume_project(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.resume_project(project_id)

    def _tool_archive_project(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.archive_project(project_id)

    def _tool_unarchive_project(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.unarchive_project(project_id)

    def _tool_get_capabilities(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return client.get_capabilities()

    def _tool_get_limits(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return client.get_limits()

    def _tool_get_capacity_lane_preview(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_capacity_lane_preview(project_id)

    def _tool_set_provider_key(self, args: JSONDict) -> Any:
        request = ProviderKeyRequest.from_payload(args)
        with self._client_from_args(args) as client:
            return client.set_provider_key(
                request.project_id,
                provider=request.provider,
                funding_source=request.funding_source,
                api_key=request.api_key,
                encrypted_key_b64=request.encrypted_key_b64,
            )

    def _tool_get_provider_key_status(self, args: JSONDict) -> Any:
        request = ProviderKeyRequest.from_payload(args)
        with self._client_from_args(args) as client:
            return client.get_provider_key_status(
                request.project_id,
                provider=request.provider,
                funding_source=request.funding_source,
            )

    def _tool_get_workspace_download_url(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_workspace_download_url(project_id)

    def _tool_get_billing_entitlements(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            result = client.get_billing_entitlements()
            return asdict(result) if is_dataclass(result) else result

    def _tool_get_run_usage(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        with self._client_from_args(args) as client:
            result = client.get_run_usage(run_id)
            return asdict(result) if is_dataclass(result) else result

    def _tool_get_run_resource_limits(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            result = (
                client.get_project_run_resource_limits(project_id, run_id)
                if project_id
                else client.get_run_resource_limits(run_id)
            )
            return asdict(result) if is_dataclass(result) else result

    def _tool_get_run_progress_toward_resource_limits(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            result = (
                client.get_project_run_progress_toward_resource_limits(project_id, run_id)
                if project_id
                else client.get_run_progress_toward_resource_limits(run_id)
            )
            return asdict(result) if is_dataclass(result) else result

    def _tool_request_resource_limit_extension(self, args: JSONDict) -> Any:
        scope = require_string(args, "scope").strip().lower()
        if scope != "run":
            raise ValueError(
                "scope must be 'run' (project-scope resource-limit extension is no longer supported)"
            )
        limit_value = self._optional_float_arg(args, "limit_value")
        additional_value = self._optional_float_arg(args, "additional_value")
        resolve_blockers = optional_bool(args, "resolve_blockers", default=True)
        resume = optional_bool(args, "resume", default=True)
        kwargs = {
            "limit_value": limit_value,
            "additional_value": additional_value,
            "reason": optional_string(args, "reason"),
            "selector": _optional_object_arg(args, "selector"),
            "resource_limit_id": optional_string(args, "resource_limit_id"),
            "metric": optional_string(args, "metric"),
            "unit": optional_string(args, "unit"),
            "resolve_blockers": resolve_blockers,
            "resume": resume,
            "idempotency_key": optional_string(args, "idempotency_key"),
        }
        with self._client_from_args(args) as client:
            run_id = require_string(args, "run_id")
            project_id = optional_string(args, "project_id")
            result = (
                client.extend_project_run_resource_limit(
                    project_id,
                    run_id,
                    **kwargs,
                )
                if project_id
                else client.extend_run_resource_limit(run_id, **kwargs)
            )
            return asdict(result) if is_dataclass(result) else result

    def _tool_get_project_usage(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            result = client.get_project_usage(project_id)
            return asdict(result) if is_dataclass(result) else result

    def _tool_get_project_economics(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            result = client.get_project_economics(project_id)
            return asdict(result) if is_dataclass(result) else result

    def _tool_get_project_git(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_project_git(project_id)

    def _tool_download_workspace_archive(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = optional_string(args, "run_id")
        output_path = require_string(args, "output_path")
        timeout_raw = optional_int(args, "timeout_seconds")
        timeout_seconds = float(timeout_raw) if timeout_raw is not None else None
        with self._client_from_args(args) as client:
            if run_id:
                if timeout_seconds is not None:
                    return client.download_run_workspace_archive(
                        project_id,
                        run_id,
                        output_path,
                        timeout_seconds=timeout_seconds,
                    )
                return client.download_run_workspace_archive(
                    project_id,
                    run_id,
                    output_path,
                )
            if timeout_seconds is not None:
                return client.download_workspace_archive(
                    project_id,
                    output_path,
                    timeout_seconds=timeout_seconds,
                )
            return client.download_workspace_archive(project_id, output_path)

    def _tool_download_code(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = optional_string(args, "run_id")
        output_path = require_string(args, "output_path")
        timeout_raw = optional_int(args, "timeout_seconds")
        timeout_seconds = float(timeout_raw) if timeout_raw is not None else None
        with self._client_from_args(args) as client:
            if timeout_seconds is not None:
                return client.download_code_archive(
                    project_id,
                    output_path,
                    run_id=run_id,
                    timeout_seconds=timeout_seconds,
                )
            return client.download_code_archive(
                project_id,
                output_path,
                run_id=run_id,
            )

    def _tool_attach_source_repo(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        url = require_string(args, "url")
        default_branch = optional_string(args, "default_branch")
        with self._client_from_args(args) as client:
            return client.attach_source_repo(project_id, url, default_branch=default_branch)

    def _tool_get_workspace_inputs(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.get_workspace_inputs(project_id)

    def _tool_upload_workspace_files(self, args: JSONDict) -> Any:
        request = WorkspaceFileUploadRequest.from_payload(args)
        with self._client_from_args(args) as client:
            return client.upload_workspace_files(request.project_id, request.files)

    # --- trained model registry --------------------------------------------

    @staticmethod
    def _optional_float_arg(payload: JSONDict, key: str) -> float | None:
        value = payload.get(key)
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a number, not bool")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number") from exc

    def _tool_register_trained_model(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        base_model = require_string(args, "base_model")
        method = require_string(args, "method")
        tinker_path = require_string(args, "tinker_path")
        metadata = args.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        with self._client_from_args(args) as client:
            result = client.trained_models.register(
                run_id=run_id,
                base_model=base_model,
                method=method,
                tinker_path=tinker_path,
                task_id=optional_string(args, "task_id"),
                episode_id=optional_string(args, "episode_id"),
                lora_rank=optional_int(args, "lora_rank"),
                base_metric=self._optional_float_arg(args, "base_metric"),
                tuned_metric=self._optional_float_arg(args, "tuned_metric"),
                uplift_abs=self._optional_float_arg(args, "uplift_abs"),
                train_cost_usd=self._optional_float_arg(args, "train_cost_usd"),
                metadata=metadata or {},
            )
            return asdict(result) if is_dataclass(result) else result

    def _tool_get_trained_model(self, args: JSONDict) -> Any:
        model_id = require_string(args, "model_id")
        with self._client_from_args(args) as client:
            result = client.trained_models.get(model_id)
            return asdict(result) if is_dataclass(result) else result

    def _tool_list_trained_models_for_run(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return [
                asdict(item) if is_dataclass(item) else item
                for item in client.trained_models.list_for_run(run_id)
            ]

    def _tool_export_trained_model(self, args: JSONDict) -> Any:
        model_id = require_string(args, "model_id")
        destination = args.get("destination")
        if not isinstance(destination, dict):
            raise ValueError("destination must be an object")
        with self._client_from_args(args) as client:
            result = client.trained_models.export(
                model_id,
                destination=destination,
                idempotency_key=optional_string(args, "idempotency_key"),
            )
            return asdict(result) if is_dataclass(result) else result

    def _tool_create_trained_model_adapter_upload_url(self, args: JSONDict) -> Any:
        model_id = require_string(args, "model_id")
        with self._client_from_args(args) as client:
            result = client.trained_models.create_adapter_upload_url(
                model_id,
                expires_in=optional_int(args, "expires_in") or 3600,
                content_type=optional_string(args, "content_type") or "application/gzip",
            )
            return asdict(result) if is_dataclass(result) else result

    def _tool_complete_trained_model_adapter_upload(self, args: JSONDict) -> Any:
        model_id = require_string(args, "model_id")
        metadata_patch = args.get("metadata_patch")
        if metadata_patch is not None and not isinstance(metadata_patch, dict):
            raise ValueError("metadata_patch must be an object")
        adapter_size_bytes = optional_int(args, "adapter_size_bytes")
        if adapter_size_bytes is None:
            raise ValueError("adapter_size_bytes is required")
        with self._client_from_args(args) as client:
            result = client.trained_models.complete_adapter_upload(
                model_id,
                bucket=require_string(args, "bucket"),
                key=require_string(args, "key"),
                adapter_size_bytes=adapter_size_bytes,
                metadata_patch=metadata_patch,
            )
            return asdict(result) if is_dataclass(result) else result

    def _tool_update_trained_model(self, args: JSONDict) -> Any:
        model_id = require_string(args, "model_id")
        metadata_patch = args.get("metadata_patch")
        if metadata_patch is not None and not isinstance(metadata_patch, dict):
            raise ValueError("metadata_patch must be an object")
        with self._client_from_args(args) as client:
            result = client.trained_models.update(
                model_id,
                tuned_metric=self._optional_float_arg(args, "tuned_metric"),
                uplift_abs=self._optional_float_arg(args, "uplift_abs"),
                train_cost_usd=self._optional_float_arg(args, "train_cost_usd"),
                status=optional_string(args, "status"),
                metadata_patch=metadata_patch,
            )
            return asdict(result) if is_dataclass(result) else result

    def _tool_delete_trained_model(self, args: JSONDict) -> Any:
        model_id = require_string(args, "model_id")
        with self._client_from_args(args) as client:
            result = client.trained_models.delete(model_id)
            return asdict(result) if is_dataclass(result) else result

    def _tool_get_run_cost_summary(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.run_cost.summary(run_id)

    def _tool_list_project_files(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        visibility = optional_string(args, "visibility")
        limit = optional_int(args, "limit")
        with self._client_from_args(args) as client:
            return client.list_project_files(
                project_id,
                visibility=visibility,
                limit=limit,
            )

    def _tool_create_project_files(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        files = args.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("'files' must be a non-empty array")
        with self._client_from_args(args) as client:
            return client.create_project_files(project_id, files)

    def _tool_get_project_file(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        file_id = require_string(args, "file_id")
        with self._client_from_args(args) as client:
            return client.get_project_file(project_id, file_id)

    def _tool_get_file_content(self, args: JSONDict) -> Any:
        file_id = require_string(args, "file_id")
        with self._client_from_args(args) as client:
            return client.get_file_content(file_id)

    def _tool_list_run_file_mounts(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.list_run_file_mounts(run_id)

    def _tool_upload_run_files(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        files = args.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("'files' must be a non-empty array")
        with self._client_from_args(args) as client:
            return client.upload_run_files(run_id, files)

    def _tool_list_run_output_files(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        artifact_type = optional_string(args, "artifact_type")
        limit = optional_int(args, "limit")
        with self._client_from_args(args) as client:
            return client.files.list_outputs(
                run_id,
                artifact_type=artifact_type,
                limit=limit,
            )

    def _tool_get_run_output_file_content(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        output_file_id = require_string(args, "output_file_id")
        disposition = optional_string(args, "disposition") or "inline"
        with self._client_from_args(args) as client:
            return client.files.get_output_content(
                run_id,
                output_file_id,
                disposition=disposition,
            )

    def _tool_list_run_artifacts(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        artifact_type = optional_string(args, "artifact_type")
        limit = optional_int(args, "limit")
        cursor = optional_string(args, "cursor")
        with self._client_from_args(args) as client:
            return [
                asdict(item)
                for item in client.runs.artifacts(
                    run_id,
                    project_id=project_id,
                    artifact_type=artifact_type,
                    limit=limit,
                    cursor=cursor,
                )
            ]

    def _tool_get_run_artifact_manifest(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            return asdict(client.runs.artifact_manifest(run_id, project_id=project_id))

    def _tool_get_artifact(self, args: JSONDict) -> Any:
        artifact_id = require_string(args, "artifact_id")
        with self._client_from_args(args) as client:
            return asdict(client.get_artifact(artifact_id))

    def _tool_get_artifact_content(self, args: JSONDict) -> Any:
        artifact_id = require_string(args, "artifact_id")
        disposition = optional_string(args, "disposition") or "inline"
        with self._client_from_args(args) as client:
            return client.get_artifact_content(artifact_id, disposition=disposition)

    def _tool_download_artifact(self, args: JSONDict) -> Any:
        artifact_id = require_string(args, "artifact_id")
        output_path = require_string(args, "output_path")
        disposition = optional_string(args, "disposition") or "attachment"
        with self._client_from_args(args) as client:
            return client.download_artifact(
                artifact_id,
                output_path,
                disposition=disposition,
            )

    def _tool_list_run_models(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.runs.models(run_id, project_id=project_id)

    def _tool_list_run_datasets(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.runs.datasets(run_id, project_id=project_id)

    def _tool_list_project_external_repositories(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.list_project_external_repositories(project_id)

    def _tool_create_project_external_repository(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        name = require_string(args, "name")
        url = require_string(args, "url")
        default_branch = optional_string(args, "default_branch")
        role = optional_string(args, "role")
        metadata = args.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("'metadata' must be an object when provided")
        with self._client_from_args(args) as client:
            return client.create_project_external_repository(
                project_id,
                name=name,
                url=url,
                default_branch=default_branch,
                role=role,
                metadata=metadata,
            )

    def _tool_patch_project_external_repository(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        repository_id = require_string(args, "repository_id")
        metadata = args.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("'metadata' must be an object when provided")
        with self._client_from_args(args) as client:
            return client.patch_project_external_repository(
                project_id,
                repository_id,
                url=optional_string(args, "url"),
                default_branch=optional_string(args, "default_branch"),
                role=optional_string(args, "role"),
                metadata=metadata,
            )

    def _tool_list_run_repository_mounts(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.list_run_repository_mounts(run_id)

    def _tool_create_run_repository_mount(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        repository_id = require_string(args, "repository_id")
        mount_name = optional_string(args, "mount_name")
        with self._client_from_args(args) as client:
            return client.create_run_repository_mount(
                run_id,
                repository_id=repository_id,
                mount_name=mount_name,
            )

    def _tool_list_project_credential_refs(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        kind = optional_string(args, "kind")
        with self._client_from_args(args) as client:
            return client.list_project_credential_refs(project_id, kind=kind)

    def _tool_create_project_credential_ref(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        kind = require_string(args, "kind")
        label = require_string(args, "label")
        metadata = args.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("'metadata' must be an object when provided")
        with self._client_from_args(args) as client:
            return client.create_project_credential_ref(
                project_id,
                kind=kind,
                label=label,
                provider=optional_string(args, "provider"),
                funding_source=optional_string(args, "funding_source"),
                credential_name=optional_string(args, "credential_name"),
                metadata=metadata,
            )

    def _tool_patch_project_credential_ref(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        credential_ref_id = require_string(args, "credential_ref_id")
        metadata = args.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("'metadata' must be an object when provided")
        with self._client_from_args(args) as client:
            return client.patch_project_credential_ref(
                project_id,
                credential_ref_id,
                provider=optional_string(args, "provider"),
                funding_source=optional_string(args, "funding_source"),
                credential_name=optional_string(args, "credential_name"),
                metadata=metadata,
            )

    def _tool_list_run_credential_bindings(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.list_run_credential_bindings(run_id)

    def _tool_create_run_credential_binding(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        credential_ref_id = require_string(args, "credential_ref_id")
        with self._client_from_args(args) as client:
            return client.create_run_credential_binding(
                run_id,
                credential_ref_id=credential_ref_id,
            )

    def _tool_trigger_run(self, args: JSONDict) -> Any:
        request = RunLaunchRequest.from_payload(args)
        try:
            with self._client_from_args(args) as client:
                if request.project_id is None:
                    return client.trigger_one_off_run(**request.client_kwargs())
                return client.trigger_run(request.project_id, **request.client_kwargs())
        except SmrApiError as exc:
            _raise_mcp_tool_denial(exc)

    def _tool_start_run(self, args: JSONDict) -> Any:
        request = RunLaunchRequest.from_payload(args)
        try:
            with self._client_from_args(args) as client:
                if request.project_id is None:
                    return client.trigger_one_off_run(**request.client_kwargs())
                return client.start_run(request.project_id, **request.client_kwargs())
        except SmrApiError as exc:
            _raise_mcp_tool_denial(exc)

    def _tool_get_launch_preflight_in_dev_environment(self, args: JSONDict) -> Any:
        _reject_dev_environment_run_substrate_args(args)
        payload = dict(args)
        payload.setdefault("host_kind", "daytona")
        request = RunLaunchRequest.from_payload(payload)
        if request.project_id is None:
            raise ValueError("project_id is required")
        if request.dev_environment_id is None:
            raise ValueError("dev_environment_id is required")
        client_kwargs = request.client_kwargs()
        dev_environment_id = str(client_kwargs.pop("dev_environment_id") or "").strip()
        with self._client_from_args(args) as client:
            return client.runs.launch_preflight_in_dev_environment(
                request.project_id,
                dev_environment_id=dev_environment_id,
                **client_kwargs,
            )

    def _tool_start_run_in_dev_environment(self, args: JSONDict) -> Any:
        _reject_dev_environment_run_substrate_args(args)
        payload = dict(args)
        payload.setdefault("host_kind", "daytona")
        request = RunLaunchRequest.from_payload(payload)
        if request.project_id is None:
            raise ValueError("project_id is required")
        if request.dev_environment_id is None:
            raise ValueError("dev_environment_id is required")
        if not str(request.objective or "").strip():
            raise ValueError("objective is required")
        client_kwargs = request.client_kwargs()
        dev_environment_id = str(client_kwargs.pop("dev_environment_id") or "").strip()
        try:
            with self._client_from_args(args) as client:
                return client.runs.start_run_in_dev_environment(
                    request.project_id,
                    dev_environment_id=dev_environment_id,
                    **client_kwargs,
                )
        except SmrApiError as exc:
            _raise_mcp_tool_denial(exc)

    def _tool_start_one_off_run(self, args: JSONDict) -> Any:
        request = OneOffRunLaunchRequest.from_payload(args)
        try:
            with self._client_from_args(args) as client:
                return client.trigger_one_off_run(**request.client_kwargs())
        except SmrApiError as exc:
            _raise_mcp_tool_denial(exc)

    def _tool_get_run_start_blockers(self, args: JSONDict) -> Any:
        request = RunLaunchRequest.from_payload(args)
        with self._client_from_args(args) as client:
            if request.project_id is None:
                return client.get_one_off_launch_preflight(**request.client_kwargs())
            if hasattr(client, "get_run_start_blockers"):
                return client.get_run_start_blockers(
                    request.project_id,
                    **request.client_kwargs(),
                )
            return client.get_launch_preflight(
                request.project_id,
                **request.client_kwargs(),
            )

    def _tool_list_runbook_presets(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return [preset.to_wire() for preset in client.list_runbook_presets()]

    def _tool_list_runs(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        active_only = optional_bool(args, "active_only", default=False)
        public_state = optional_string(args, "public_state")
        limit = optional_int(args, "limit") or 50
        with self._client_from_args(args) as client:
            return client.list_runs(
                project_id,
                active_only=active_only,
                public_state=public_state,
                limit=limit,
            )

    def _tool_jobs_list(self, args: JSONDict) -> Any:
        active_only = optional_bool(args, "active_only") if "active_only" in args else None
        with self._client_from_args(args) as client:
            return client.list_jobs(
                project_id=optional_string(args, "project_id"),
                state=optional_string(args, "state"),
                active_only=active_only,
                limit=optional_int(args, "limit"),
            )

    def _tool_project_trigger_run(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_config = args.get("run_config")
        if run_config is None:
            run_config = {}
        if not isinstance(run_config, dict):
            raise ValueError("'run_config' must be an object when provided")
        return self._tool_trigger_run({"project_id": project_id, **run_config})

    def _tool_get_run(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            result = client.runs.get(run_id, project_id=project_id)
            return asdict(result) if is_dataclass(result) else result

    def _tool_get_run_contract(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        with self._client_from_args(args) as client:
            result = client.runs.get_run_contract(project_id, run_id)
            return asdict(result) if is_dataclass(result) else result

    def _tool_get_run_primary_parent(self, args: JSONDict) -> Any:
        self._removed_backend_contract("Run primary-parent read")

    def _tool_run_objective_scopes(self, args: JSONDict) -> Any:
        self._removed_backend_contract("Run objective-scope management")

    def _tool_stop_run(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            result = client.runs.stop(run_id, project_id=project_id)
            return asdict(result) if is_dataclass(result) else result

    def _tool_pause_run(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            result = client.runs.pause(run_id, project_id=project_id)
            return asdict(result) if is_dataclass(result) else result

    def _tool_resume_run(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            result = client.runs.resume(run_id, project_id=project_id)
            return asdict(result) if is_dataclass(result) else result

    def _tool_get_run_logical_timeline(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.get_run_logical_timeline(project_id, run_id)

    def _tool_get_run_execution(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.get_run_execution(
                project_id,
                run_id,
                view=optional_string(args, "view") or "summary",
                event_limit=optional_int(args, "event_limit") or 100,
                actor_limit=optional_int(args, "actor_limit") or 50,
                task_limit=optional_int(args, "task_limit") or 100,
                message_limit=optional_int(args, "message_limit") or 50,
                work_product_limit=optional_int(args, "work_product_limit") or 50,
            )

    def _tool_list_run_task_events(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.list_run_task_events(
                project_id,
                run_id,
                limit=optional_int(args, "limit"),
                cursor=optional_string(args, "cursor"),
            )

    def _tool_list_run_objective_events(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.list_run_objective_events(
                project_id,
                run_id,
                limit=optional_int(args, "limit"),
                cursor=optional_string(args, "cursor"),
            )

    def _tool_get_run_work_graph(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.get_run_work_graph(
                project_id,
                run_id,
                limit=optional_int(args, "limit"),
            )

    def _tool_list_tasks(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.runs.list_tasks(
                project_id,
                run_id=optional_string(args, "run_id"),
                objective_id=optional_string(args, "objective_id"),
                kind=optional_string(args, "kind"),
                limit=optional_int(args, "limit"),
            )

    def _tool_create_task(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        payload = args.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("'payload' must be an object")
        with self._client_from_args(args) as client:
            return asdict(
                client.runs.create_task(
                    run_id,
                    payload,
                    project_id=project_id,
                    mode=optional_string(args, "mode") or "queue",
                    body=optional_string(args, "body"),
                )
            )

    def _tool_update_task(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        task_id = require_string(args, "task_id")
        payload = args.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("'payload' must be an object")
        with self._client_from_args(args) as client:
            return asdict(
                client.runs.update_task(
                    run_id,
                    task_id,
                    payload,
                    project_id=project_id,
                    mode=optional_string(args, "mode") or "queue",
                    body=optional_string(args, "body"),
                )
            )

    def _tool_cancel_task(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        task_id = require_string(args, "task_id")
        with self._client_from_args(args) as client:
            return asdict(
                client.runs.cancel_task(
                    run_id,
                    task_id,
                    project_id=project_id,
                    reason=optional_string(args, "reason"),
                    mode=optional_string(args, "mode") or "queue",
                    body=optional_string(args, "body"),
                )
            )

    def _tool_reassign_task(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        task_id = require_string(args, "task_id")
        assignee = require_string(args, "assignee")
        with self._client_from_args(args) as client:
            return asdict(
                client.runs.reassign_task(
                    run_id,
                    task_id,
                    project_id=project_id,
                    assignee=assignee,
                    mode=optional_string(args, "mode") or "queue",
                    body=optional_string(args, "body"),
                )
            )

    def _tool_get_run_event_log(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.get_project_run_event_log(
                project_id,
                run_id,
                sources=args.get("sources"),
                event_kinds=args.get("event_kinds"),
                statuses=args.get("statuses"),
                limit=optional_int(args, "limit"),
            )

    def _tool_get_run_authority_readouts(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            if project_id:
                return client.get_project_run_authority_readouts(
                    project_id,
                    run_id,
                    include_runtime_authority=optional_bool(
                        args,
                        "include_runtime_authority",
                        default=False,
                    ),
                )
            return client.get_run_authority_readouts(
                run_id,
                include_runtime_authority=optional_bool(
                    args,
                    "include_runtime_authority",
                    default=False,
                ),
            )

    def _tool_get_run_operator_evidence(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.get_project_run_operator_evidence(
                project_id,
                run_id,
                runtime_timeline_limit=optional_int(args, "runtime_timeline_limit"),
                logical_timeline_limit=optional_int(args, "logical_timeline_limit"),
                transcript_limit=optional_int(args, "transcript_limit"),
                reconciliation_limit=optional_int(args, "reconciliation_limit"),
            )

    def _tool_get_run_traces(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            if project_id:
                return client.get_project_run_traces(project_id, run_id)
            return client.get_run_traces(run_id)

    def _tool_get_run_actor_trace(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        actor_key = require_string(args, "actor_key")
        with self._client_from_args(args) as client:
            return client.get_project_run_actor_trace(
                project_id,
                run_id,
                actor_key,
                cursor=optional_string(args, "cursor"),
                live_cursor=optional_string(args, "live_cursor"),
                limit=optional_int(args, "limit"),
                include_live=optional_bool(args, "include_live"),
                include_traces=optional_bool(args, "include_traces"),
            )

    def _tool_list_run_actor_traces(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        actor_key = optional_string(args, "actor_key")
        with self._client_from_args(args) as client:
            if actor_key:
                return client.get_project_run_actor_raw_traces(project_id, run_id, actor_key)
            return client.get_project_run_actor_trace_index(project_id, run_id)

    def _tool_get_raw_trace_events(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        artifact_id = require_string(args, "artifact_id")
        with self._client_from_args(args) as client:
            return client.get_project_run_raw_trace_events(
                project_id,
                run_id,
                artifact_id,
                cursor=optional_string(args, "cursor"),
                limit=optional_int(args, "limit"),
                redaction_mode=optional_string(args, "redaction_mode"),
                reconstruct=optional_bool(args, "reconstruct"),
                category=args.get("category"),
                method=args.get("method"),
            )

    def _tool_download_raw_trace(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        artifact_id = require_string(args, "artifact_id")
        if not optional_bool(args, "confirm_raw_download"):
            raise ValueError("confirm_raw_download=true is required for raw trace downloads")
        destination = optional_string(args, "destination")
        with self._client_from_args(args) as client:
            if destination:
                return client.download_project_run_raw_trace(
                    project_id,
                    run_id,
                    artifact_id,
                    destination,
                    expires_in=optional_int(args, "expires_in"),
                )
            return client.create_project_run_raw_trace_download_url(
                project_id,
                run_id,
                artifact_id,
                expires_in=optional_int(args, "expires_in"),
            )

    def _tool_get_run_actor_usage(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            if project_id:
                return client.get_project_run_actor_usage(project_id, run_id)
            return client.get_run_actor_usage(run_id)

    def _tool_control_project_run_actor(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        actor_id = require_string(args, "actor_id")
        action = require_string(args, "action")
        with self._client_from_args(args) as client:
            result = client.runs.control_actor(
                project_id,
                run_id,
                actor_id,
                action=action,
                reason=optional_string(args, "reason"),
                idempotency_key=optional_string(args, "idempotency_key"),
            )
            return asdict(result) if is_dataclass(result) else result

    def _tool_list_run_participants(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            result = client.list_run_participants(run_id, project_id=project_id)
            return asdict(result) if is_dataclass(result) else result

    def _tool_get_run_artifact_progress(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            result = client.get_run_artifact_progress(run_id, project_id=project_id)
            return asdict(result) if is_dataclass(result) else result

    def _tool_list_run_actor_logs(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            result = client.list_run_actor_logs(
                run_id,
                project_id=project_id,
                actor_id=optional_string(args, "actor_id"),
                turn_id=optional_string(args, "turn_id"),
                kind=optional_string(args, "kind"),
                since=optional_string(args, "since"),
                cursor=optional_string(args, "cursor"),
                limit=optional_int(args, "limit"),
            )
            return asdict(result) if is_dataclass(result) else result

    def _tool_branch_run_from_checkpoint(self, args: JSONDict) -> Any:
        run_id = optional_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        request = parse_branch_run_request(args)
        with self._client_from_args(args) as client:
            return client.branch_run_from_checkpoint(
                run_id,
                project_id=project_id,
                checkpoint_id=request.checkpoint_id,
                checkpoint_record_id=request.checkpoint_record_id,
                checkpoint_uri=request.checkpoint_uri,
                mode=request.mode,
                message=request.message,
                reason=request.reason,
                title=request.title,
                source_node_id=request.source_node_id,
            )

    def _tool_runtime_message_queue(self, args: JSONDict) -> Any:
        operation = require_string(args, "operation").strip().lower()
        if operation not in {"list", "enqueue"}:
            raise ValueError("'operation' must be either 'list' or 'enqueue'")
        run_id = require_string(args, "run_id")
        with self._client_from_args(args) as client:
            if operation == "list":
                status = optional_string(args, "status")
                viewer_role = optional_string(args, "viewer_role")
                limit = optional_int(args, "limit")
                raw_viewer_target = args.get("viewer_target")
                viewer_target: str | list[str] | None = None
                if isinstance(raw_viewer_target, str) and raw_viewer_target.strip():
                    viewer_target = raw_viewer_target.strip()
                elif isinstance(raw_viewer_target, list):
                    cleaned_targets = [
                        str(item).strip() for item in raw_viewer_target if str(item).strip()
                    ]
                    viewer_target = cleaned_targets or None
                return client.runs.list_runtime_messages(
                    run_id,
                    status=status,
                    viewer_role=viewer_role,
                    viewer_target=viewer_target,
                    limit=limit,
                )

            payload = args.get("payload")
            if payload is not None and not isinstance(payload, dict):
                raise ValueError("'payload' must be an object when provided")
            return client.enqueue_runtime_message(
                run_id,
                topic=optional_string(args, "topic"),
                causation_id=optional_string(args, "causation_id"),
                mode=optional_string(args, "mode"),
                spawn_policy=optional_string(args, "spawn_policy"),
                sender=optional_string(args, "sender"),
                target=optional_string(args, "target"),
                participant_session_id=optional_string(args, "participant_session_id"),
                action=optional_string(args, "action"),
                body=optional_string(args, "body"),
                payload=payload,
            )

    def _tool_list_messages(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.runs.list_messages(
                run_id,
                project_id=project_id,
                thread_id=optional_string(args, "thread_id"),
                limit=optional_int(args, "limit"),
            )

    def _tool_send_message(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        payload = args.get("payload")
        if payload is not None and not isinstance(payload, dict):
            raise ValueError("'payload' must be an object when provided")
        audience = args.get("audience")
        if audience is not None and not isinstance(audience, dict):
            raise ValueError("'audience' must be an object when provided")
        with self._client_from_args(args) as client:
            return client.runs.send_message(
                run_id,
                project_id=project_id,
                intent=optional_string(args, "intent") or "queue",
                audience=audience,
                body=optional_string(args, "body"),
                payload=payload,
                message_kind=optional_string(args, "message_kind") or "runtime_message",
                thread_id=optional_string(args, "thread_id"),
                parent_message_id=optional_string(args, "parent_message_id"),
                fallback_policy=optional_string(args, "fallback_policy") or "block",
                idempotency_key=optional_string(args, "idempotency_key"),
                correlation_id=optional_string(args, "correlation_id"),
                causation_id=optional_string(args, "causation_id"),
            )

    def _tool_edit_message(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        message_id = require_string(args, "message_id")
        payload = args.get("payload")
        if payload is not None and not isinstance(payload, dict):
            raise ValueError("'payload' must be an object when provided")
        with self._client_from_args(args) as client:
            return client.runs.edit_message(
                run_id,
                message_id,
                project_id=project_id,
                body=optional_string(args, "body"),
                payload=payload,
            )

    def _tool_retract_message(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        message_id = require_string(args, "message_id")
        with self._client_from_args(args) as client:
            return client.runs.retract_message(
                run_id,
                message_id,
                project_id=project_id,
            )

    def _tool_runtime_intents(self, args: JSONDict) -> Any:
        operation = require_string(args, "operation").strip().lower()
        if operation not in {"submit", "list", "get"}:
            raise ValueError("'operation' must be one of: submit, list, get")
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            if operation == "list":
                rows = client.runs.intents(
                    run_id,
                    project_id=project_id,
                    status=optional_string(args, "status"),
                    limit=optional_int(args, "limit"),
                )
                return [asdict(row) for row in rows]
            if operation == "get":
                runtime_intent_id = require_string(args, "runtime_intent_id")
                return asdict(
                    client.runs.intent(
                        run_id,
                        runtime_intent_id,
                        project_id=project_id,
                    )
                )

            intent = args.get("intent")
            if not isinstance(intent, dict):
                raise ValueError("'intent' must be an object for operation=submit")
            return asdict(
                client.runs.submit_intent(
                    run_id,
                    intent,
                    project_id=project_id,
                    mode=optional_string(args, "mode") or "queue",
                    body=optional_string(args, "body"),
                    causation_id=optional_string(args, "causation_id"),
                )
            )

    def _tool_list_active_runs(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        with self._client_from_args(args) as client:
            return client.list_active_runs(project_id)

    def _tool_list_run_questions(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        status_filter = optional_string(args, "status_filter")
        limit = optional_int(args, "limit") or 100
        with self._client_from_args(args) as client:
            return client.list_run_questions(
                run_id,
                project_id=project_id,
                status_filter=status_filter,
                limit=limit,
            )

    def _tool_respond_to_run_question(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        question_id = require_string(args, "question_id")
        project_id = optional_string(args, "project_id")
        response_text = require_string(args, "response_text")
        with self._client_from_args(args) as client:
            return client.respond_to_run_question(
                run_id,
                question_id,
                project_id=project_id,
                response_text=response_text,
            )

    def _tool_list_run_approvals(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        status_filter = optional_string(args, "status_filter")
        limit = optional_int(args, "limit") or 100
        with self._client_from_args(args) as client:
            return client.list_run_approvals(
                run_id,
                project_id=project_id,
                status_filter=status_filter,
                limit=limit,
            )

    def _tool_approve_run_approval(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        approval_id = require_string(args, "approval_id")
        project_id = optional_string(args, "project_id")
        comment = optional_string(args, "comment")
        with self._client_from_args(args) as client:
            return client.approve_run_approval(
                run_id,
                approval_id,
                project_id=project_id,
                comment=comment,
            )

    def _tool_deny_run_approval(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        approval_id = require_string(args, "approval_id")
        project_id = optional_string(args, "project_id")
        comment = optional_string(args, "comment")
        with self._client_from_args(args) as client:
            return client.deny_run_approval(
                run_id,
                approval_id,
                project_id=project_id,
                comment=comment,
            )

    def _tool_create_run_checkpoint(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        checkpoint_id = optional_string(args, "checkpoint_id")
        reason = optional_string(args, "reason")
        with self._client_from_args(args) as client:
            return client.request_run_checkpoint(
                run_id,
                project_id=project_id,
                checkpoint_id=checkpoint_id,
                reason=reason,
            )

    def _tool_list_run_checkpoints(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        with self._client_from_args(args) as client:
            return [
                checkpoint.to_wire()
                for checkpoint in client.list_run_checkpoints(
                    run_id,
                    project_id=project_id,
                )
            ]

    def _tool_restore_run_checkpoint(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        project_id = optional_string(args, "project_id")
        checkpoint_id = optional_string(args, "checkpoint_id")
        checkpoint_record_id = optional_string(args, "checkpoint_record_id")
        checkpoint_uri = optional_string(args, "checkpoint_uri")
        reason = optional_string(args, "reason")
        mode = optional_string(args, "mode") or "in_place"
        with self._client_from_args(args) as client:
            return client.restore_run_checkpoint(
                run_id,
                project_id=project_id,
                checkpoint_id=checkpoint_id,
                checkpoint_record_id=checkpoint_record_id,
                checkpoint_uri=checkpoint_uri,
                reason=reason,
                mode=mode,
            )

    def _tool_list_run_log_archives(self, args: JSONDict) -> Any:
        project_id = require_string(args, "project_id")
        run_id = require_string(args, "run_id")
        with self._client_from_args(args) as client:
            return client.logs.list_run_archives(project_id, run_id)

    def _tool_get_run_transcript(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        cursor = optional_string(args, "cursor")
        limit = optional_int(args, "limit") or 100
        participant_session_id = optional_string(args, "participant_session_id")
        view = optional_string(args, "view")
        with self._client_from_args(args) as client:
            return client.runs.transcript(
                run_id,
                cursor=cursor,
                limit=min(limit, 200),
                participant_session_id=participant_session_id,
                view=view,
            )

    def _tool_watch_run_events(self, args: JSONDict) -> Any:
        run_id = require_string(args, "run_id")
        transcript_cursor = optional_string(args, "transcript_cursor")
        last_event_id = optional_string(args, "last_event_id")
        view = optional_string(args, "view") or "operator"
        max_events = min(optional_int(args, "max_events") or 20, 50)
        timeout_seconds = optional_int(args, "timeout_seconds") or 30
        events: list[dict[str, Any]] = []
        with self._client_from_args(args) as client:
            for event in client.runs.stream_events(
                run_id,
                transcript_cursor=transcript_cursor,
                view=view,
                last_event_id=last_event_id,
                timeout=float(timeout_seconds),
            ):
                row = asdict(event)
                occurred_at = row.get("occurred_at")
                if hasattr(occurred_at, "isoformat"):
                    row["occurred_at"] = occurred_at.isoformat()
                events.append(row)
                if len(events) >= max_events:
                    break
        return {
            "run_id": run_id,
            "view": view,
            "event_count": len(events),
            "events": events,
            "next_last_event_id": events[-1].get("event_id") if events else last_event_id,
            "next_transcript_cursor": (
                events[-1].get("transcript_cursor") if events else transcript_cursor
            ),
        }

    def _tool_get_launch_preflight(self, args: JSONDict) -> Any:
        request = RunLaunchRequest.from_payload(args)
        with self._client_from_args(args) as client:
            if request.project_id is None:
                return client.get_one_off_launch_preflight(**request.client_kwargs())
            return client.get_launch_preflight(
                request.project_id,
                **request.client_kwargs(),
            )

    def _tool_open_ended_questions(self, args: JSONDict) -> Any:
        self._removed_backend_contract("Project open-ended-question management")

    def _tool_objectives(self, args: JSONDict) -> Any:
        operation = objective_tool_operation_from_wire(require_string(args, "operation"))
        project_id = require_string(args, "project_id")
        kind = optional_string(args, "kind")
        with self._client_from_args(args) as client:
            if operation is ObjectiveToolOperation.LIST:
                return client.list_objectives(
                    project_id,
                    kind=kind,
                    run_id=optional_string(args, "run_id"),
                    limit=optional_int(args, "limit"),
                )
            if operation is ObjectiveToolOperation.PROGRESS:
                return client.get_objective_progress(
                    project_id,
                    require_string(args, "objective_id"),
                    kind=kind,
                )
            if operation is ObjectiveToolOperation.CLAIMS:
                return client.list_objective_progress_claims(
                    project_id,
                    require_string(args, "objective_id"),
                    kind=kind,
                    limit=optional_int(args, "limit"),
                )
            if operation is ObjectiveToolOperation.CLAIM:
                payload = _optional_object_arg(args, "payload") or {}
                return client.create_objective_progress_claim(
                    project_id,
                    require_string(args, "objective_id"),
                    payload=payload,
                    kind=kind,
                )
        self._removed_backend_contract(f"Project objective operation '{operation.value}'")

    def _tool_get_objective_status(self, args: JSONDict) -> Any:
        with self._client_from_args(args) as client:
            return client.get_objective_status(
                require_string(args, "project_id"),
                require_string(args, "objective_id"),
                kind=optional_string(args, "kind"),
                task_limit=optional_int(args, "task_limit"),
                claim_limit=optional_int(args, "claim_limit"),
                event_limit=optional_int(args, "event_limit"),
                milestone_limit=optional_int(args, "milestone_limit"),
            )

    def _tool_milestones(self, args: JSONDict) -> Any:
        operation = require_string(args, "operation").strip().lower()
        project_id = require_string(args, "project_id")
        payload = _optional_object_arg(args, "payload") or {}
        with self._client_from_args(args) as client:
            if operation == "list":
                return client.list_milestones(
                    project_id,
                    run_id=optional_string(args, "run_id"),
                    parent_kind=optional_string(args, "parent_kind"),
                    parent_id=optional_string(args, "parent_id"),
                    limit=optional_int(args, "limit"),
                )
            if operation == "create":
                return client.create_milestone(project_id, payload=payload)
            if operation == "get":
                return client.get_milestone(
                    project_id,
                    require_string(args, "milestone_id"),
                )
            if operation == "patch":
                return client.patch_milestone(
                    project_id,
                    require_string(args, "milestone_id"),
                    payload=payload,
                )
            if operation == "transition":
                return client.transition_milestone(
                    project_id,
                    require_string(args, "milestone_id"),
                    payload=payload,
                )
        raise ValueError(
            "smr_milestones.operation must be one of: list, create, get, patch, transition"
        )

    def _tool_directed_effort_outcomes(self, args: JSONDict) -> Any:
        self._removed_backend_contract("Project directed-effort-outcome management")

    # ---- Open Research tool helpers ----------------------------------

    def _open_research_client_from_args(
        self,
        args: JSONDict,
        *,
        ensure_fingerprint: bool = False,
    ) -> OpenResearchClient:
        api_key = optional_string(args, "api_key") or self._default_api_key
        backend_base = optional_string(args, "backend_base") or self._default_backend_base
        fingerprint = optional_string(args, "submitter_fingerprint")
        if ensure_fingerprint and not api_key and not fingerprint:
            fingerprint = load_or_create_fingerprint()
        return OpenResearchClient(
            api_key=api_key,
            fingerprint=fingerprint,
            backend_base=backend_base,
        )

    def _tool_open_research_list_projects(self, args: JSONDict) -> Any:
        with self._open_research_client_from_args(args) as client:
            return client.list_projects().model_dump(mode="json")

    def _tool_open_research_get_project(self, args: JSONDict) -> Any:
        slug = require_string(args, "slug")
        with self._open_research_client_from_args(args) as client:
            try:
                return client.get_project(slug).model_dump(mode="json")
            except OpenResearchError as exc:
                _raise_open_research_error(exc)

    def _tool_open_research_list_queues(self, args: JSONDict) -> Any:
        project_slug = optional_string(args, "project_slug")
        with self._open_research_client_from_args(args) as client:
            return client.list_queues(project_slug=project_slug).model_dump(mode="json")

    def _tool_open_research_submit_question(self, args: JSONDict) -> Any:
        # Build the typed args once. Validation errors surface as plain
        # ValueError (which the JSON-RPC wrapper maps to -32000).
        metric_target = args.get("metric_target")
        if not isinstance(metric_target, dict):
            raise ValueError("'metric_target' is required and must be an object")
        submit_args = SubmitQuestionArgs.model_validate(
            {
                "project_slug": require_string(args, "project_slug"),
                "queue_id": require_string(args, "queue_id"),
                "prompt": require_string(args, "prompt"),
                "hypothesis": optional_string(args, "hypothesis") or "",
                "metric_target": metric_target,
                "deo_kind": require_string(args, "deo_kind"),
                "rubric_acknowledged": optional_bool(args, "rubric_acknowledged"),
                "submitter_handle": require_string(args, "submitter_handle"),
                "submitter_fingerprint": optional_string(args, "submitter_fingerprint"),
            }
        )
        with self._open_research_client_from_args(args, ensure_fingerprint=True) as client:
            # If the client minted a fingerprint and the caller did not
            # supply one, populate the submitter so the request body and
            # the X-OR-Fingerprint header stay consistent.
            if submit_args.submitter_fingerprint is None and client.fingerprint:
                submit_args = submit_args.model_copy(
                    update={"submitter_fingerprint": client.fingerprint}
                )
            try:
                response = client.submit_question(submit_args)
            except OpenResearchError as exc:
                _raise_open_research_error(exc)
        return response.model_dump(mode="json")

    def _tool_open_research_get_submission(self, args: JSONDict) -> Any:
        submission_id = require_string(args, "submission_id")
        with self._open_research_client_from_args(args) as client:
            try:
                return client.get_submission(submission_id).model_dump(mode="json")
            except OpenResearchError as exc:
                _raise_open_research_error(exc)

    def _tool_open_research_list_experiments(self, args: JSONDict) -> Any:
        project_slug = optional_string(args, "project_slug")
        raw_status = optional_string(args, "status")
        status: ExperimentStatusFilter | None = None
        if raw_status is not None:
            allowed = {"running", "done", "failed", "all"}
            if raw_status not in allowed:
                raise ValueError(f"'status' must be one of {sorted(allowed)} when provided")
            status = cast(ExperimentStatusFilter, raw_status)
        limit = optional_int(args, "limit")
        cursor = optional_string(args, "cursor")
        with self._open_research_client_from_args(args) as client:
            return client.list_experiments(
                project_slug=project_slug,
                status=status,
                limit=limit,
                cursor=cursor,
            ).model_dump(mode="json")

    def _tool_open_research_get_experiment(self, args: JSONDict) -> Any:
        experiment_id = require_string(args, "experiment_id")
        with self._open_research_client_from_args(args) as client:
            try:
                return client.get_experiment(experiment_id).model_dump(mode="json")
            except OpenResearchError as exc:
                _raise_open_research_error(exc)

    def _tool_open_research_get_receipt(self, args: JSONDict) -> Any:
        experiment_id = require_string(args, "experiment_id")
        with self._open_research_client_from_args(args) as client:
            try:
                return client.get_receipt(experiment_id).model_dump(mode="json")
            except OpenResearchError as exc:
                _raise_open_research_error(exc)

    def _tool_open_research_download_bundle(self, args: JSONDict) -> Any:
        experiment_id = require_string(args, "experiment_id")
        dest_path = require_string(args, "dest_path")
        raw_timeout = args.get("timeout_seconds")
        timeout_seconds: float | None
        if raw_timeout is None:
            timeout_seconds = None
        elif isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)):
            raise ValueError("'timeout_seconds' must be a number when provided")
        else:
            timeout_seconds = float(raw_timeout)
        with self._open_research_client_from_args(args) as client:
            try:
                result = client.download_bundle(
                    experiment_id,
                    dest_path,
                    timeout_seconds=timeout_seconds,
                )
            except OpenResearchError as exc:
                _raise_open_research_error(exc)
        return result.model_dump(mode="json")

    def serve_stdio(self) -> None:
        framing = "jsonl"
        while True:
            message = _read_message(sys.stdin.buffer)
            if message is None:
                return
            request, framing = message
            response = self._handle_message(request)
            if response is None:
                continue
            _write_message(sys.stdout.buffer, response, framing=framing)

    def _handle_message(self, request: JSONDict) -> JSONDict | None:
        method = request.get("method")
        request_id = request.get("id")
        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": DEFAULT_PROTOCOL_VERSION,
                        "serverInfo": {"name": SERVER_NAME, "version": __version__},
                        "capabilities": {"tools": {}},
                    },
                }
            if method == "ping":
                return {"jsonrpc": "2.0", "id": request_id, "result": {}}
            if method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": self.list_tool_payload()},
                }
            if method == "tools/call":
                params = request.get("params")
                if not isinstance(params, dict):
                    raise RpcError(-32602, "tools/call requires object params")
                tool_name = params.get("name")
                if not isinstance(tool_name, str) or tool_name not in self._tools:
                    raise RpcError(-32601, f"Unknown tool: {tool_name!r}")
                arguments = params.get("arguments")
                if arguments is not None and not isinstance(arguments, dict):
                    raise RpcError(-32602, "tools/call arguments must be an object")
                result = self.call_tool(tool_name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, sort_keys=True)}],
                        "structuredContent": result,
                    },
                }
            if method in {"shutdown", "exit"}:
                return {"jsonrpc": "2.0", "id": request_id, "result": {}}
            if method in {"initialized", "notifications/initialized"}:
                return None
            raise RpcError(-32601, f"Unsupported method: {method!r}")
        except RpcError as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": exc.code, "message": exc.message, "data": exc.data},
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": str(exc)},
            }


def main() -> None:
    """CLI entrypoint for the stdio MCP server."""
    ManagedResearchMcpServer().serve_stdio()


__all__ = [
    "DEFAULT_PROTOCOL_VERSION",
    "ManagedResearchMcpServer",
    "SERVER_NAME",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "_read_message",
    "main",
]
