"""Backend-authored Research operation registry.

# See: openapi/research-v1.json
"""

from __future__ import annotations

from synth_ai.core.http.request import HttpMethod, OperationId, OperationMetadata


def _operation(
    operation_id: str,
    method: HttpMethod,
    path: str,
    *,
    mutation: bool = False,
    idempotent: bool = False,
) -> OperationMetadata:
    return OperationMetadata(OperationId(operation_id), method, path, mutation, idempotent)


RESEARCH_OPERATIONS = {
    operation.operation_id: operation
    for operation in (
        _operation(
            "archive_factory", HttpMethod.POST, "/smr/factories/{factory_id}/archive", mutation=True
        ),
        _operation(
            "archive_project",
            HttpMethod.POST,
            "/smr/projects/{project_id}/archive",
            mutation=True,
            idempotent=True,
        ),
        _operation(
            "confirm_project_workspace_push",
            HttpMethod.POST,
            "/smr/projects/{project_id}/workspace/confirm-push",
            mutation=True,
            idempotent=True,
        ),
        _operation("branch_run", HttpMethod.POST, "/smr/runs/{run_id}/branches", mutation=True),
        _operation("create_effort", HttpMethod.POST, "/smr/efforts", mutation=True),
        _operation(
            "create_research_environment", HttpMethod.POST, "/smr/environments", mutation=True
        ),
        _operation(
            "create_image_release_upload",
            HttpMethod.POST,
            "/smr/v1/image-releases/upload-url",
            mutation=True,
        ),
        _operation("create_factory", HttpMethod.POST, "/smr/factories", mutation=True),
        _operation("create_project", HttpMethod.POST, "/smr/projects:runnable", mutation=True),
        _operation(
            "create_project_dataset",
            HttpMethod.POST,
            "/smr/projects/{project_id}/datasets",
            mutation=True,
        ),
        _operation(
            "create_project_external_repository",
            HttpMethod.POST,
            "/smr/projects/{project_id}/external-repositories",
            mutation=True,
        ),
        _operation(
            "delete_project_external_repository",
            HttpMethod.DELETE,
            "/smr/projects/{project_id}/external-repositories/{repository_id}",
            mutation=True,
            idempotent=True,
        ),
        _operation("list_factories", HttpMethod.GET, "/smr/factories", idempotent=True),
        _operation(
            "list_customer_actor_images", HttpMethod.GET, "/smr/v1/image-releases", idempotent=True
        ),
        _operation(
            "list_research_environments", HttpMethod.GET, "/smr/environments", idempotent=True
        ),
        _operation(
            "list_factory_efforts",
            HttpMethod.GET,
            "/smr/factories/{factory_id}/efforts",
            idempotent=True,
        ),
        _operation(
            "list_factory_candidates",
            HttpMethod.GET,
            "/smr/factories/{factory_id}/candidates",
            idempotent=True,
        ),
        _operation(
            "list_factory_champion_events",
            HttpMethod.GET,
            "/smr/factories/{factory_id}/champion/events",
            idempotent=True,
        ),
        _operation("list_jobs", HttpMethod.GET, "/smr/jobs", idempotent=True),
        _operation(
            "list_project_active_runs",
            HttpMethod.GET,
            "/smr/projects/{project_id}/runs/active",
            idempotent=True,
        ),
        _operation(
            "list_project_datasets",
            HttpMethod.GET,
            "/smr/projects/{project_id}/datasets",
            idempotent=True,
        ),
        _operation(
            "list_project_external_repositories",
            HttpMethod.GET,
            "/smr/projects/{project_id}/external-repositories",
            idempotent=True,
        ),
        _operation(
            "list_project_runs", HttpMethod.GET, "/smr/projects/{project_id}/runs", idempotent=True
        ),
        _operation("list_projects", HttpMethod.GET, "/smr/projects", idempotent=True),
        _operation(
            "list_run_transcript",
            HttpMethod.GET,
            "/smr/runs/{run_id}/runtime/transcript",
            idempotent=True,
        ),
        _operation("list_runbook_presets", HttpMethod.GET, "/smr/runbook-presets", idempotent=True),
        _operation("list_runs", HttpMethod.GET, "/smr/runs", idempotent=True),
        _operation(
            "pause_factory", HttpMethod.POST, "/smr/factories/{factory_id}/pause", mutation=True
        ),
        _operation("pause_run", HttpMethod.POST, "/smr/runs/{run_id}/pause", mutation=True),
        _operation("preflight_one_off_run", HttpMethod.POST, "/smr/runs:one-off/launch-preflight"),
        _operation(
            "preflight_project_run", HttpMethod.POST, "/smr/projects/{project_id}/launch-preflight"
        ),
        _operation(
            "preflight_research_environment", HttpMethod.POST, "/smr/environments/{name}/preflight"
        ),
        _operation(
            "prepare_project_setup",
            HttpMethod.POST,
            "/smr/projects/{project_id}/setup/prepare",
            mutation=True,
        ),
        _operation(
            "resume_factory", HttpMethod.POST, "/smr/factories/{factory_id}/resume", mutation=True
        ),
        _operation(
            "record_factory_candidate_grading",
            HttpMethod.POST,
            "/smr/factories/{factory_id}/candidates/{candidate_id}/grading",
            mutation=True,
        ),
        _operation(
            "rollback_factory_champion",
            HttpMethod.POST,
            "/smr/factories/{factory_id}/champion/rollback",
            mutation=True,
        ),
        _operation("resume_run", HttpMethod.POST, "/smr/runs/{run_id}/resume", mutation=True),
        _operation(
            "finalize_image_release",
            HttpMethod.POST,
            "/smr/v1/image-releases/finalize",
            mutation=True,
        ),
        _operation(
            "archive_customer_actor_image",
            HttpMethod.POST,
            "/smr/v1/image-releases/{runtime_image_release_id}/archive",
            mutation=True,
        ),
        _operation("retrieve_effort", HttpMethod.GET, "/smr/efforts/{effort_id}", idempotent=True),
        _operation(
            "retrieve_factory", HttpMethod.GET, "/smr/factories/{factory_id}", idempotent=True
        ),
        _operation(
            "retrieve_project", HttpMethod.GET, "/smr/projects/{project_id}", idempotent=True
        ),
        _operation(
            "retrieve_image_release",
            HttpMethod.GET,
            "/smr/v1/image-releases/{release_id}",
            idempotent=True,
        ),
        _operation(
            "retrieve_research_environment",
            HttpMethod.GET,
            "/smr/environments/{name}",
            idempotent=True,
        ),
        _operation(
            "retrieve_project_dataset_content",
            HttpMethod.GET,
            "/smr/projects/{project_id}/datasets/{dataset_id}/download",
            idempotent=True,
        ),
        _operation(
            "retrieve_project_workspace_inputs",
            HttpMethod.GET,
            "/smr/projects/{project_id}/workspace-inputs",
            idempotent=True,
        ),
        _operation(
            "retrieve_billing_catalog", HttpMethod.GET, "/smr/billing/catalog", idempotent=True
        ),
        _operation(
            "retrieve_billing_entitlements",
            HttpMethod.GET,
            "/api/v1/billing/entitlements",
            idempotent=True,
        ),
        _operation("retrieve_billing_plan", HttpMethod.GET, "/smr/billing/plan", idempotent=True),
        _operation(
            "retrieve_effort_billing_drawdown",
            HttpMethod.GET,
            "/smr/billing/factory-efforts/{factory_effort_id}/drawdown",
            idempotent=True,
        ),
        _operation(
            "retrieve_project_run",
            HttpMethod.GET,
            "/smr/projects/{project_id}/runs/{run_id}",
            idempotent=True,
        ),
        _operation(
            "retrieve_project_economics",
            HttpMethod.GET,
            "/smr/projects/{project_id}/economics",
            idempotent=True,
        ),
        _operation(
            "retrieve_project_setup",
            HttpMethod.GET,
            "/smr/projects/{project_id}/setup",
            idempotent=True,
        ),
        _operation("retrieve_research_limits", HttpMethod.GET, "/smr/limits", idempotent=True),
        _operation("retrieve_run", HttpMethod.GET, "/smr/runs/{run_id}", idempotent=True),
        _operation(
            "retrieve_swarm_artifact_content",
            HttpMethod.GET,
            "/smr/artifacts/{artifact_id}/content",
            idempotent=True,
        ),
        _operation(
            "retrieve_swarm_configuration",
            HttpMethod.GET,
            "/smr/runs/{run_id}/configuration",
            idempotent=True,
        ),
        _operation(
            "retrieve_swarm_activity",
            HttpMethod.GET,
            "/smr/runs/{run_id}/activity",
            idempotent=True,
        ),
        _operation(
            "retrieve_swarm_status",
            HttpMethod.GET,
            "/smr/runs/{run_id}/status",
            idempotent=True,
        ),
        _operation(
            "retrieve_swarm_evidence",
            HttpMethod.GET,
            "/smr/runs/{run_id}/evidence",
            idempotent=True,
        ),
        _operation(
            "retrieve_swarm_usage",
            HttpMethod.GET,
            "/smr/runs/{run_id}/usage-summary",
            idempotent=True,
        ),
        _operation(
            "retrieve_swarm_workspace_archive",
            HttpMethod.GET,
            "/smr/runs/{run_id}/workspace/archive",
            idempotent=True,
        ),
        _operation(
            "retrieve_swarm_billing_drawdown",
            HttpMethod.GET,
            "/smr/billing/runs/{run_id}/drawdown",
            idempotent=True,
        ),
        _operation(
            "retrieve_swarm_work_product_content",
            HttpMethod.GET,
            "/smr/work-products/{work_product_id}/content",
            idempotent=True,
        ),
        _operation(
            "start_factory", HttpMethod.POST, "/smr/factories/{factory_id}/start", mutation=True
        ),
        _operation(
            "select_factory_champion",
            HttpMethod.POST,
            "/smr/factories/{factory_id}/champion/select",
            mutation=True,
        ),
        _operation(
            "set_project_workspace_source_repository",
            HttpMethod.PUT,
            "/smr/projects/{project_id}/workspace-inputs/source-repo",
            mutation=True,
        ),
        _operation("stop_run", HttpMethod.POST, "/smr/runs/{run_id}/stop", mutation=True),
        _operation(
            "stream_run_events",
            HttpMethod.GET,
            "/smr/runs/{run_id}/runtime/stream",
            idempotent=True,
        ),
        _operation("trigger_one_off_run", HttpMethod.POST, "/smr/runs:one-off", mutation=True),
        _operation(
            "trigger_project_run",
            HttpMethod.POST,
            "/smr/projects/{project_id}/trigger",
            mutation=True,
        ),
        _operation(
            "unarchive_project",
            HttpMethod.POST,
            "/smr/projects/{project_id}/unarchive",
            mutation=True,
        ),
        _operation(
            "upload_project_workspace_files",
            HttpMethod.POST,
            "/smr/projects/{project_id}/workspace-inputs/files:upload",
            mutation=True,
            idempotent=True,
        ),
        _operation("update_effort", HttpMethod.PATCH, "/smr/efforts/{effort_id}", mutation=True),
        _operation(
            "update_factory", HttpMethod.PATCH, "/smr/factories/{factory_id}", mutation=True
        ),
        _operation("update_project", HttpMethod.PATCH, "/smr/projects/{project_id}", mutation=True),
        _operation(
            "update_project_external_repository",
            HttpMethod.PATCH,
            "/smr/projects/{project_id}/external-repositories/{repository_id}",
            mutation=True,
        ),
    )
}


def research_operation(operation_id: str) -> OperationMetadata:
    try:
        return RESEARCH_OPERATIONS[OperationId(operation_id)]
    except KeyError as error:
        raise ValueError(f"unknown Research operation_id {operation_id!r}") from error


__all__ = ["RESEARCH_OPERATIONS", "research_operation"]
