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
        _operation("archive_factory", HttpMethod.POST, "/smr/factories/{factory_id}/archive", mutation=True),
        _operation("archive_project", HttpMethod.POST, "/smr/projects/{project_id}/archive", mutation=True),
        _operation("branch_run", HttpMethod.POST, "/smr/runs/{run_id}/branches", mutation=True),
        _operation("create_effort", HttpMethod.POST, "/smr/efforts", mutation=True),
        _operation("create_factory", HttpMethod.POST, "/smr/factories", mutation=True),
        _operation("create_project", HttpMethod.POST, "/smr/projects:runnable", mutation=True),
        _operation("list_factories", HttpMethod.GET, "/smr/factories", idempotent=True),
        _operation("list_factory_efforts", HttpMethod.GET, "/smr/factories/{factory_id}/efforts", idempotent=True),
        _operation("list_jobs", HttpMethod.GET, "/smr/jobs", idempotent=True),
        _operation("list_project_active_runs", HttpMethod.GET, "/smr/projects/{project_id}/runs/active", idempotent=True),
        _operation("list_project_runs", HttpMethod.GET, "/smr/projects/{project_id}/runs", idempotent=True),
        _operation("list_projects", HttpMethod.GET, "/smr/projects", idempotent=True),
        _operation("list_run_transcript", HttpMethod.GET, "/smr/runs/{run_id}/runtime/transcript", idempotent=True),
        _operation("list_runbook_presets", HttpMethod.GET, "/smr/runbook-presets", idempotent=True),
        _operation("list_runs", HttpMethod.GET, "/smr/runs", idempotent=True),
        _operation("pause_factory", HttpMethod.POST, "/smr/factories/{factory_id}/pause", mutation=True),
        _operation("pause_run", HttpMethod.POST, "/smr/runs/{run_id}/pause", mutation=True),
        _operation("preflight_one_off_run", HttpMethod.POST, "/smr/runs:one-off/launch-preflight"),
        _operation("preflight_project_run", HttpMethod.POST, "/smr/projects/{project_id}/launch-preflight"),
        _operation("prepare_project_setup", HttpMethod.POST, "/smr/projects/{project_id}/setup/prepare", mutation=True),
        _operation("resume_factory", HttpMethod.POST, "/smr/factories/{factory_id}/resume", mutation=True),
        _operation("resume_run", HttpMethod.POST, "/smr/runs/{run_id}/resume", mutation=True),
        _operation("retrieve_effort", HttpMethod.GET, "/smr/efforts/{effort_id}", idempotent=True),
        _operation("retrieve_factory", HttpMethod.GET, "/smr/factories/{factory_id}", idempotent=True),
        _operation("retrieve_project", HttpMethod.GET, "/smr/projects/{project_id}", idempotent=True),
        _operation("retrieve_billing_catalog", HttpMethod.GET, "/smr/billing/catalog", idempotent=True),
        _operation("retrieve_billing_entitlements", HttpMethod.GET, "/api/v1/billing/entitlements", idempotent=True),
        _operation("retrieve_billing_plan", HttpMethod.GET, "/smr/billing/plan", idempotent=True),
        _operation("retrieve_effort_billing_drawdown", HttpMethod.GET, "/smr/billing/factory-efforts/{factory_effort_id}/drawdown", idempotent=True),
        _operation("retrieve_project_run", HttpMethod.GET, "/smr/projects/{project_id}/runs/{run_id}", idempotent=True),
        _operation("retrieve_project_economics", HttpMethod.GET, "/smr/projects/{project_id}/economics", idempotent=True),
        _operation("retrieve_project_setup", HttpMethod.GET, "/smr/projects/{project_id}/setup", idempotent=True),
        _operation("retrieve_research_limits", HttpMethod.GET, "/smr/limits", idempotent=True),
        _operation("retrieve_run", HttpMethod.GET, "/smr/runs/{run_id}", idempotent=True),
        _operation("retrieve_swarm_configuration", HttpMethod.GET, "/smr/runs/{run_id}/configuration", idempotent=True),
        _operation(
            "retrieve_swarm_usage",
            HttpMethod.GET,
            "/smr/runs/{run_id}/usage-summary",
            idempotent=True,
        ),
        _operation("retrieve_swarm_billing_drawdown", HttpMethod.GET, "/smr/billing/runs/{run_id}/drawdown", idempotent=True),
        _operation("start_factory", HttpMethod.POST, "/smr/factories/{factory_id}/start", mutation=True),
        _operation("stop_run", HttpMethod.POST, "/smr/runs/{run_id}/stop", mutation=True),
        _operation("stream_run_events", HttpMethod.GET, "/smr/runs/{run_id}/runtime/stream", idempotent=True),
        _operation("trigger_one_off_run", HttpMethod.POST, "/smr/runs:one-off", mutation=True),
        _operation("trigger_project_run", HttpMethod.POST, "/smr/projects/{project_id}/trigger", mutation=True),
        _operation("unarchive_project", HttpMethod.POST, "/smr/projects/{project_id}/unarchive", mutation=True),
        _operation("update_effort", HttpMethod.PATCH, "/smr/efforts/{effort_id}", mutation=True),
        _operation("update_factory", HttpMethod.PATCH, "/smr/factories/{factory_id}", mutation=True),
        _operation("update_project", HttpMethod.PATCH, "/smr/projects/{project_id}", mutation=True),
    )
}


def research_operation(operation_id: str) -> OperationMetadata:
    try:
        return RESEARCH_OPERATIONS[OperationId(operation_id)]
    except KeyError as error:
        raise ValueError(f"unknown Research operation_id {operation_id!r}") from error


__all__ = ["RESEARCH_OPERATIONS", "research_operation"]
