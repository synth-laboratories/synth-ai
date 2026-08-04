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

# Keyed on the advertised `research_*` tool name as declared in `tools/`. A
# tool absent from this table is a build-time error, not an unauthenticated
# tool -- see `_scoped_tool_definition`.
_DEFAULT_REQUIRED_SCOPES_BY_TOOL_NAME: dict[str, tuple[str, ...]] = {
    "research_health_check": READ_SCOPES,
    "research_create_runnable_project": WRITE_SCOPES,
    "research_list_projects": READ_SCOPES,
    "research_get_project": READ_SCOPES,
    "research_rename_project": WRITE_SCOPES,
    "research_patch_project": WRITE_SCOPES,
    "research_get_project_status": READ_SCOPES,
    "research_get_project_workspace": READ_SCOPES,
    "research_list_project_changesets": READ_SCOPES,
    "research_get_project_changeset": READ_SCOPES,
    "research_create_project_changeset": WRITE_SCOPES,
    "research_decide_project_changeset": WRITE_SCOPES,
    "research_get_project_entitlement": READ_SCOPES,
    "research_get_project_notes": READ_SCOPES,
    "research_set_project_notes": WRITE_SCOPES,
    "research_append_project_notes": WRITE_SCOPES,
    "research_get_org_knowledge": READ_SCOPES,
    "research_set_org_knowledge": WRITE_SCOPES,
    "research_get_project_knowledge": READ_SCOPES,
    "research_set_project_knowledge": WRITE_SCOPES,
    "research_curated_knowledge": READ_SCOPES,
    "research_pause_project": WRITE_SCOPES,
    "research_resume_project": WRITE_SCOPES,
    "research_archive_project": WRITE_SCOPES,
    "research_unarchive_project": WRITE_SCOPES,
    "research_get_capabilities": READ_SCOPES,
    "research_get_limits": READ_SCOPES,
    "research_get_capacity_lane_preview": READ_SCOPES,
    "research_set_provider_key": WRITE_SCOPES,
    "research_get_provider_key_status": READ_SCOPES,
    "research_get_workspace_download_url": READ_SCOPES,
    "research_get_project_git": READ_SCOPES,
    "research_download_workspace_archive": READ_SCOPES,
    "research_download_code": READ_SCOPES,
    "research_attach_source_repo": WRITE_SCOPES,
    "research_get_workspace_inputs": READ_SCOPES,
    "research_upload_workspace_files": WRITE_SCOPES,
    "research_confirm_workspace_push": WRITE_SCOPES,
    "research_list_project_files": READ_SCOPES,
    "research_create_project_files": WRITE_SCOPES,
    "research_get_project_file": READ_SCOPES,
    "research_get_file_content": READ_SCOPES,
    "research_list_run_file_mounts": READ_SCOPES,
    "research_upload_run_files": WRITE_SCOPES,
    "research_list_run_output_files": READ_SCOPES,
    "research_get_run_output_file_content": READ_SCOPES,
    "research_list_project_external_repositories": READ_SCOPES,
    "research_create_project_external_repository": WRITE_SCOPES,
    "research_patch_project_external_repository": WRITE_SCOPES,
    "research_list_run_repository_mounts": READ_SCOPES,
    "research_create_run_repository_mount": WRITE_SCOPES,
    "research_list_project_credential_refs": READ_SCOPES,
    "research_create_project_credential_ref": WRITE_SCOPES,
    "research_patch_project_credential_ref": WRITE_SCOPES,
    "research_list_run_credential_bindings": READ_SCOPES,
    "research_create_run_credential_binding": WRITE_SCOPES,
    "research_get_project_setup": READ_SCOPES,
    "research_prepare_project_setup": WRITE_SCOPES,
    "research_get_launch_preflight": READ_SCOPES,
    "research_get_launch_preflight_in_dev_environment": READ_SCOPES,
    "research_start_run": WRITE_SCOPES,
    "research_start_run_in_dev_environment": WRITE_SCOPES,
    "research_trigger_run": WRITE_SCOPES,
    "research_list_runs": READ_SCOPES,
    "research_get_run": READ_SCOPES,
    "research_get_swarm_activity": READ_SCOPES,
    "research_get_swarm_configuration": READ_SCOPES,
    "research_get_swarm_evidence": READ_SCOPES,
    "research_get_swarm_usage": READ_SCOPES,
    "research_get_run_execution": READ_SCOPES,
    "research_get_run_logical_timeline": READ_SCOPES,
    "research_get_run_event_log": READ_SCOPES,
    "research_get_run_authority_readouts": READ_SCOPES,
    "research_get_run_operator_evidence": READ_SCOPES,
    "research_get_run_traces": READ_SCOPES,
    "research_list_run_actor_traces": READ_SCOPES,
    "research_get_run_actor_trace": READ_SCOPES,
    "research_get_raw_trace_events": READ_SCOPES,
    "research_download_raw_trace": READ_SCOPES,
    "research_get_run_actor_usage": READ_SCOPES,
    "research_control_project_run_actor": WRITE_SCOPES,
    "research_list_run_participants": READ_SCOPES,
    "research_get_run_artifact_progress": READ_SCOPES,
    "research_list_run_actor_logs": READ_SCOPES,
    "research_list_tasks": READ_SCOPES,
    "research_create_task": WRITE_SCOPES,
    "research_update_task": WRITE_SCOPES,
    "research_cancel_task": WRITE_SCOPES,
    "research_reassign_task": WRITE_SCOPES,
    "research_stop_run": WRITE_SCOPES,
    "research_branch_run_from_checkpoint": WRITE_SCOPES,
    "research_runtime_message_queue": READ_SCOPES,
    "research_list_messages": READ_SCOPES,
    "research_send_message": WRITE_SCOPES,
    "research_edit_message": WRITE_SCOPES,
    "research_retract_message": WRITE_SCOPES,
    "research_runtime_intents": WRITE_SCOPES,
    "research_list_active_runs": READ_SCOPES,
    "research_list_run_questions": READ_SCOPES,
    "research_respond_to_run_question": WRITE_SCOPES,
    "research_list_run_approvals": READ_SCOPES,
    "research_approve_run_approval": WRITE_SCOPES,
    "research_deny_run_approval": WRITE_SCOPES,
    "research_create_run_checkpoint": WRITE_SCOPES,
    "research_list_run_checkpoints": READ_SCOPES,
    "research_restore_run_checkpoint": WRITE_SCOPES,
    "research_objectives": READ_SCOPES,
    "research_list_run_log_archives": READ_SCOPES,
    "research_get_billing_entitlements": READ_SCOPES,
    "research_preview_admin_promotion_discount": READ_SCOPES,
    "research_get_run_usage": READ_SCOPES,
    "research_get_run_resource_limits": READ_SCOPES,
    "research_get_run_progress_toward_resource_limits": READ_SCOPES,
    "research_request_resource_limit_extension": WRITE_SCOPES,
    "research_get_project_usage": READ_SCOPES,
    "research_get_project_economics": READ_SCOPES,
    "research_setup_github_status": READ_SCOPES,
    "research_setup_github_start_oauth": WRITE_SCOPES,
    "research_setup_github_list_repos": READ_SCOPES,
    "research_setup_github_disconnect": WRITE_SCOPES,
    "research_list_run_artifacts": READ_SCOPES,
    "research_get_run_artifact_manifest": READ_SCOPES,
    "research_get_artifact": READ_SCOPES,
    "research_get_artifact_content": READ_SCOPES,
    "research_download_artifact": READ_SCOPES,
    "research_list_run_models": READ_SCOPES,
    "research_list_run_datasets": READ_SCOPES,
    "research_get_default_project": READ_SCOPES,
    "research_get_objective_status": READ_SCOPES,
    "research_milestones": WRITE_SCOPES,
    "research_status_readiness": READ_SCOPES,
    "research_explain_work_product_blocker": READ_SCOPES,
    "research_export_run_work_product": WRITE_SCOPES,
    "research_get_run_work_product": READ_SCOPES,
    "research_get_run_work_product_content": READ_SCOPES,
    "research_list_run_work_products": READ_SCOPES,
    "research_upload_container_eval_package": WRITE_SCOPES,
    "research_validate_container_eval_package": WRITE_SCOPES,
    "research_get_run_contract": READ_SCOPES,
    "research_get_run_cost_summary": READ_SCOPES,
    "research_get_run_transcript": READ_SCOPES,
    "research_get_run_work_graph": READ_SCOPES,
    "research_get_swarm_status": READ_SCOPES,
    "research_get_swarm_workspace_archive": READ_SCOPES,
    "research_list_run_objective_events": READ_SCOPES,
    "research_list_run_task_events": READ_SCOPES,
    "research_list_runs_by_effort": READ_SCOPES,
    "research_watch_run_events": READ_SCOPES,
    "research_pause_run": WRITE_SCOPES,
    "research_resume_run": WRITE_SCOPES,
    "research_start_one_off_run": WRITE_SCOPES,
    "research_get_trained_model": READ_SCOPES,
    "research_list_trained_models_for_run": READ_SCOPES,
    "research_register_trained_model": WRITE_SCOPES,
    "research_update_trained_model": WRITE_SCOPES,
    "research_delete_trained_model": WRITE_SCOPES,
    "research_export_trained_model": WRITE_SCOPES,
    "research_create_trained_model_adapter_upload_url": WRITE_SCOPES,
    "research_complete_trained_model_adapter_upload": WRITE_SCOPES,
    "research_results_models_list": READ_SCOPES,
    "research_results_models_get": READ_SCOPES,
    "research_results_models_download": READ_SCOPES,
    "research_results_models_export": WRITE_SCOPES,
    "research_results_prs_list": READ_SCOPES,
    "research_results_prs_get": READ_SCOPES,
    "research_setup_exports_list_targets": READ_SCOPES,
    "research_setup_exports_create_target": WRITE_SCOPES,
    "research_work_datasets_list": READ_SCOPES,
    "research_work_datasets_download": READ_SCOPES,
    "research_work_datasets_upload": WRITE_SCOPES,
    "research_work_repos_list": READ_SCOPES,
    "research_work_repos_attach": WRITE_SCOPES,
    "research_work_repos_detach": WRITE_SCOPES,
    "research_create_visual": WRITE_SCOPES,
    "research_update_visual": WRITE_SCOPES,
    "research_promote_visual": WRITE_SCOPES,
    "research_unpublish_visual": WRITE_SCOPES,
}


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: JSONDict
    handler: ToolHandler
    required_scopes: tuple[str, ...] = ()


def _scoped_tool_definition(tool: ToolDefinition) -> ToolDefinition:
    """Give a tool its required scopes, or refuse it.

    An unscoped tool is a tool anyone can call, so a missing table entry is a
    build-time failure rather than a silent grant. Adding a tool means deciding
    whether it reads or writes.
    """
    if tool.required_scopes:
        return tool
    scopes = _DEFAULT_REQUIRED_SCOPES_BY_TOOL_NAME.get(tool.name)
    if not scopes:
        raise ValueError(
            f"MCP tool {tool.name!r} declares no required scopes and is missing from "
            "_DEFAULT_REQUIRED_SCOPES_BY_TOOL_NAME; add it as READ_SCOPES or WRITE_SCOPES."
        )
    return replace(tool, required_scopes=scopes)


def tool_schema(properties: JSONDict, *, required: list[str]) -> JSONDict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_tool_registry(tools: list[ToolDefinition]) -> dict[str, ToolDefinition]:
    """Build the advertised noun-first registry, keyed on declared names.

    Tool definitions declare their advertised ``research_*`` names directly.
    For every ``research_*`` tool the legacy ``smr_*`` spelling is registered as
    a generated wire alias in :func:`resolve_tool`, so it stays callable while
    remaining intentionally absent from discovery. This preserves compatibility
    without doubling the public tool surface.
    """
    registry: dict[str, ToolDefinition] = {}
    for raw_tool in tools:
        tool = _scoped_tool_definition(raw_tool)
        if tool.name.startswith("smr_"):
            raise ValueError(
                f"MCP tool {tool.name!r} uses the legacy smr_ prefix; declare it as "
                f"'research_{tool.name[4:]}' -- the smr_ spelling is a generated alias."
            )
        if tool.name in registry:
            raise ValueError(f"duplicate MCP tool definition: {tool.name}")
        registry[tool.name] = tool
    return registry


def resolve_tool(
    tools: dict[str, ToolDefinition],
    name: str,
) -> ToolDefinition | None:
    """Resolve an advertised name or its legacy ``smr_*`` wire alias.

    Every ``research_*`` tool answers to its ``smr_*`` spelling for backward
    compatibility; the alias is generated from the advertised name rather than
    stored, so it never appears in discovery.
    """
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
        tool_values = [_scoped_tool_definition(tool) for tool in tools]
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
