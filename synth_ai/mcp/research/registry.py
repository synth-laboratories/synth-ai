"""Small MCP registry helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

JSONDict = dict[str, Any]
ToolHandler = Callable[[JSONDict], Any]

READ_SCOPE = "smr:read"
WRITE_SCOPE = "smr:write"
READ_SCOPES: tuple[str, ...] = (READ_SCOPE,)
WRITE_SCOPES: tuple[str, ...] = (WRITE_SCOPE,)

_DEFAULT_REQUIRED_SCOPES_BY_TOOL_NAME: dict[str, tuple[str, ...]] = {
    "smr_health_check": READ_SCOPES,
    "smr_create_runnable_project": WRITE_SCOPES,
    "smr_list_projects": READ_SCOPES,
    "smr_get_project": READ_SCOPES,
    "smr_rename_project": WRITE_SCOPES,
    "smr_patch_project": WRITE_SCOPES,
    "smr_get_project_status": READ_SCOPES,
    "smr_get_project_workspace": READ_SCOPES,
    "smr_list_project_changesets": READ_SCOPES,
    "smr_get_project_changeset": READ_SCOPES,
    "smr_create_project_changeset": WRITE_SCOPES,
    "smr_decide_project_changeset": WRITE_SCOPES,
    "smr_get_project_entitlement": READ_SCOPES,
    "smr_get_project_notes": READ_SCOPES,
    "smr_set_project_notes": WRITE_SCOPES,
    "smr_append_project_notes": WRITE_SCOPES,
    "smr_get_org_knowledge": READ_SCOPES,
    "smr_set_org_knowledge": WRITE_SCOPES,
    "smr_get_project_knowledge": READ_SCOPES,
    "smr_set_project_knowledge": WRITE_SCOPES,
    "smr_curated_knowledge": READ_SCOPES,
    "smr_pause_project": WRITE_SCOPES,
    "smr_resume_project": WRITE_SCOPES,
    "smr_archive_project": WRITE_SCOPES,
    "smr_unarchive_project": WRITE_SCOPES,
    "smr_get_capabilities": READ_SCOPES,
    "smr_get_limits": READ_SCOPES,
    "smr_get_capacity_lane_preview": READ_SCOPES,
    "smr_set_provider_key": WRITE_SCOPES,
    "smr_get_provider_key_status": READ_SCOPES,
    "smr_get_workspace_download_url": READ_SCOPES,
    "smr_get_project_git": READ_SCOPES,
    "smr_download_workspace_archive": READ_SCOPES,
    "smr_download_code": READ_SCOPES,
    "smr_attach_source_repo": WRITE_SCOPES,
    "smr_get_workspace_inputs": READ_SCOPES,
    "smr_upload_workspace_files": WRITE_SCOPES,
    "smr_list_project_files": READ_SCOPES,
    "smr_create_project_files": WRITE_SCOPES,
    "smr_get_project_file": READ_SCOPES,
    "smr_get_file_content": READ_SCOPES,
    "smr_list_run_file_mounts": READ_SCOPES,
    "smr_upload_run_files": WRITE_SCOPES,
    "smr_list_run_output_files": READ_SCOPES,
    "smr_get_run_output_file_content": READ_SCOPES,
    "smr_list_project_external_repositories": READ_SCOPES,
    "smr_create_project_external_repository": WRITE_SCOPES,
    "smr_patch_project_external_repository": WRITE_SCOPES,
    "smr_list_run_repository_mounts": READ_SCOPES,
    "smr_create_run_repository_mount": WRITE_SCOPES,
    "smr_list_project_credential_refs": READ_SCOPES,
    "smr_create_project_credential_ref": WRITE_SCOPES,
    "smr_patch_project_credential_ref": WRITE_SCOPES,
    "smr_list_run_credential_bindings": READ_SCOPES,
    "smr_create_run_credential_binding": WRITE_SCOPES,
    "smr_get_project_setup": READ_SCOPES,
    "smr_prepare_project_setup": WRITE_SCOPES,
    "smr_list_dev_environment_topologies": READ_SCOPES,
    "smr_get_dev_environment_topology": READ_SCOPES,
    "smr_seed_dev_environment_topology_manifest": WRITE_SCOPES,
    "smr_list_dev_environment_materialization_queue": READ_SCOPES,
    "smr_list_dev_environments": READ_SCOPES,
    "smr_create_dev_environment": WRITE_SCOPES,
    "smr_create_dev_environment_from_topology": WRITE_SCOPES,
    "smr_get_dev_environment": READ_SCOPES,
    "smr_claim_dev_environment_materialization": WRITE_SCOPES,
    "smr_preflight_dev_environment": READ_SCOPES,
    "smr_deploy_dev_environment": WRITE_SCOPES,
    "smr_start_dev_environment": WRITE_SCOPES,
    "smr_stop_dev_environment": WRITE_SCOPES,
    "smr_snapshot_dev_environment": WRITE_SCOPES,
    "smr_report_dev_environment_materialization": WRITE_SCOPES,
    "smr_destroy_dev_environment": WRITE_SCOPES,
    "smr_get_dev_environment_services": READ_SCOPES,
    "smr_get_dev_environment_attach": READ_SCOPES,
    "smr_get_dev_environment_logs": READ_SCOPES,
    "smr_get_dev_environment_runs": READ_SCOPES,
    "smr_get_dev_environment_usage": READ_SCOPES,
    "smr_preflight_dev_environment_billing": READ_SCOPES,
    "smr_get_dev_environment_billing_drawdown": READ_SCOPES,
    "smr_get_dev_environment_receipts": READ_SCOPES,
    "smr_get_dev_environment_evidence": READ_SCOPES,
    "smr_list_cloud_deployments": READ_SCOPES,
    "smr_create_cloud_deployment": WRITE_SCOPES,
    "smr_get_cloud_deployment": READ_SCOPES,
    "smr_get_cloud_deployment_services": READ_SCOPES,
    "smr_get_cloud_deployment_workspace": READ_SCOPES,
    "smr_materialize_cloud_deployment_workspace": WRITE_SCOPES,
    "smr_exec_cloud_deployment": WRITE_SCOPES,
    "smr_get_cloud_deployment_logs": READ_SCOPES,
    "smr_get_cloud_deployment_artifacts": READ_SCOPES,
    "smr_get_cloud_deployment_artifact_content": READ_SCOPES,
    "smr_observe_cloud_deployment": WRITE_SCOPES,
    "smr_deploy_cloud_deployment": WRITE_SCOPES,
    "smr_retire_cloud_deployment": WRITE_SCOPES,
    "smr_acquire_cloud_deployment_claim": WRITE_SCOPES,
    "smr_heartbeat_cloud_deployment_claim": WRITE_SCOPES,
    "smr_release_cloud_deployment_claim": WRITE_SCOPES,
    "smr_get_cloud_deployment_claims": READ_SCOPES,
    "smr_get_launch_preflight": READ_SCOPES,
    "smr_get_launch_preflight_in_dev_environment": READ_SCOPES,
    "smr_start_run": WRITE_SCOPES,
    "smr_start_run_in_dev_environment": WRITE_SCOPES,
    "smr_trigger_run": WRITE_SCOPES,
    "smr_list_runs": READ_SCOPES,
    "smr_get_run": READ_SCOPES,
    "smr_get_run_execution": READ_SCOPES,
    "smr_get_run_logical_timeline": READ_SCOPES,
    "smr_get_run_event_log": READ_SCOPES,
    "smr_get_run_authority_readouts": READ_SCOPES,
    "smr_get_run_operator_evidence": READ_SCOPES,
    "smr_get_run_traces": READ_SCOPES,
    "smr_list_run_actor_traces": READ_SCOPES,
    "smr_get_run_actor_trace": READ_SCOPES,
    "smr_get_raw_trace_events": READ_SCOPES,
    "smr_download_raw_trace": READ_SCOPES,
    "smr_get_run_actor_usage": READ_SCOPES,
    "smr_control_project_run_actor": WRITE_SCOPES,
    "smr_list_run_participants": READ_SCOPES,
    "smr_get_run_artifact_progress": READ_SCOPES,
    "smr_list_run_actor_logs": READ_SCOPES,
    "smr_list_tasks": READ_SCOPES,
    "smr_create_task": WRITE_SCOPES,
    "smr_update_task": WRITE_SCOPES,
    "smr_cancel_task": WRITE_SCOPES,
    "smr_reassign_task": WRITE_SCOPES,
    "smr_stop_run": WRITE_SCOPES,
    "smr_branch_run_from_checkpoint": WRITE_SCOPES,
    "smr_runtime_message_queue": READ_SCOPES,
    "smr_list_messages": READ_SCOPES,
    "smr_send_message": WRITE_SCOPES,
    "smr_edit_message": WRITE_SCOPES,
    "smr_retract_message": WRITE_SCOPES,
    "smr_runtime_intents": WRITE_SCOPES,
    "smr_list_active_runs": READ_SCOPES,
    "smr_list_run_questions": READ_SCOPES,
    "smr_respond_to_run_question": WRITE_SCOPES,
    "smr_list_run_approvals": READ_SCOPES,
    "smr_approve_run_approval": WRITE_SCOPES,
    "smr_deny_run_approval": WRITE_SCOPES,
    "smr_create_run_checkpoint": WRITE_SCOPES,
    "smr_list_run_checkpoints": READ_SCOPES,
    "smr_restore_run_checkpoint": WRITE_SCOPES,
    "smr_objectives": READ_SCOPES,
    "smr_list_run_log_archives": READ_SCOPES,
    "smr_get_billing_entitlements": READ_SCOPES,
    "smr_preview_admin_promotion_discount": READ_SCOPES,
    "smr_get_run_usage": READ_SCOPES,
    "smr_get_run_resource_limits": READ_SCOPES,
    "smr_get_run_progress_toward_resource_limits": READ_SCOPES,
    "smr_request_resource_limit_extension": WRITE_SCOPES,
    "smr_get_project_usage": READ_SCOPES,
    "smr_get_project_economics": READ_SCOPES,
    "smr_setup_github_status": READ_SCOPES,
    "smr_setup_github_start_oauth": WRITE_SCOPES,
    "smr_setup_github_list_repos": READ_SCOPES,
    "smr_setup_github_disconnect": WRITE_SCOPES,
    "smr_list_run_artifacts": READ_SCOPES,
    "smr_get_run_artifact_manifest": READ_SCOPES,
    "smr_get_artifact": READ_SCOPES,
    "smr_get_artifact_content": READ_SCOPES,
    "smr_download_artifact": READ_SCOPES,
    "smr_list_run_models": READ_SCOPES,
    "smr_list_run_datasets": READ_SCOPES,
}


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: JSONDict
    handler: ToolHandler
    required_scopes: tuple[str, ...] = ()


def _normalized_tool_definition(tool: ToolDefinition) -> ToolDefinition:
    default_scopes = _DEFAULT_REQUIRED_SCOPES_BY_TOOL_NAME.get(tool.name, ())
    if not default_scopes and tool.name.startswith("research_"):
        default_scopes = _DEFAULT_REQUIRED_SCOPES_BY_TOOL_NAME.get(
            f"smr_{tool.name[9:]}",
            (),
        )
    if tool.required_scopes:
        return tool
    if not default_scopes:
        return tool
    return replace(tool, required_scopes=default_scopes)


def tool_schema(properties: JSONDict, *, required: list[str]) -> JSONDict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_tool_registry(tools: list[ToolDefinition]) -> dict[str, ToolDefinition]:
    """Build the advertised noun-first registry without duplicate legacy names.

    Legacy ``smr_*`` names remain callable through :func:`resolve_tool` but are
    intentionally absent from discovery. This preserves compatibility without
    doubling the public tool surface.
    """
    registry: dict[str, ToolDefinition] = {}
    for raw_tool in tools:
        tool = _normalized_tool_definition(raw_tool)
        if tool.name.startswith("smr_"):
            tool = _normalized_tool_definition(
                replace(tool, name=f"research_{tool.name[4:]}")
            )
        if tool.name in registry:
            raise ValueError(f"duplicate MCP tool definition: {tool.name}")
        registry[tool.name] = tool
    return registry


def resolve_tool(
    tools: dict[str, ToolDefinition],
    name: str,
) -> ToolDefinition | None:
    tool = tools.get(name)
    if tool is not None:
        return tool
    if name.startswith("smr_"):
        return tools.get(f"research_{name[4:]}")
    return None


def list_tool_payload(
    tools: dict[str, ToolDefinition] | list[ToolDefinition],
) -> list[JSONDict]:
    if isinstance(tools, dict):
        tool_values = tools.values()
    else:
        tool_values = [_normalized_tool_definition(tool) for tool in tools]
    payload: list[JSONDict] = []
    for tool in tool_values:
        payload.append(
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
                "requiredScopes": list(tool.required_scopes),
            }
        )
    return payload


def call_tool(
    tools: dict[str, ToolDefinition],
    name: str,
    arguments: JSONDict | None = None,
) -> Any:
    tool = resolve_tool(tools, name)
    if tool is None:
        raise KeyError(name)
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise TypeError("tool arguments must be an object")
    return tool.handler(arguments)


__all__ = [
    "JSONDict",
    "READ_SCOPE",
    "READ_SCOPES",
    "ToolDefinition",
    "WRITE_SCOPE",
    "WRITE_SCOPES",
    "build_tool_registry",
    "call_tool",
    "list_tool_payload",
    "resolve_tool",
    "tool_schema",
]
