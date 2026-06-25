"""Nested run readout namespaces on ``ResearchRunHandle``."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any, List

from synth_ai.managed_research.models.canonical_usage import (
    SmrResourceLimitProgress,
    SmrResourceLimits,
    SmrRunUsage,
)
from synth_ai.managed_research.models.run_diagnostics import (
    SmrRunActorUsage,
    SmrRunCostSummary,
)
from synth_ai.managed_research.models.run_events import RunRuntimeStreamEvent
from synth_ai.managed_research.models.run_observability import RunObservabilitySnapshot
from synth_ai.managed_research.sdk.runs import RunHandle


class _RunReadoutBound:
    def __init__(self, handle: RunHandle) -> None:
        self._handle = handle


class ResearchRunUsageActorsAPI(_RunReadoutBound):
    def get(self) -> SmrRunActorUsage:
        return self._handle.actor_usage()


class ResearchRunUsageCostAPI(_RunReadoutBound):
    def get(self) -> SmrRunCostSummary:
        return self._handle.cost_summary()


class ResearchRunUsageLimitsAPI(_RunReadoutBound):
    def get(self) -> SmrResourceLimits:
        return self._handle.resource_limits()

    def progress(self) -> SmrResourceLimitProgress:
        return self._handle.progress_toward_resource_limits()


class ResearchRunUsageAPI(_RunReadoutBound):
    def __init__(self, handle: RunHandle) -> None:
        super().__init__(handle)
        self._actors: ResearchRunUsageActorsAPI | None = None
        self._cost: ResearchRunUsageCostAPI | None = None
        self._limits: ResearchRunUsageLimitsAPI | None = None

    @property
    def actors(self) -> ResearchRunUsageActorsAPI:
        if self._actors is None:
            self._actors = ResearchRunUsageActorsAPI(self._handle)
        return self._actors

    @property
    def cost(self) -> ResearchRunUsageCostAPI:
        if self._cost is None:
            self._cost = ResearchRunUsageCostAPI(self._handle)
        return self._cost

    @property
    def limits(self) -> ResearchRunUsageLimitsAPI:
        if self._limits is None:
            self._limits = ResearchRunUsageLimitsAPI(self._handle)
        return self._limits

    def get(self) -> SmrRunUsage:
        return self._handle._client.get_run_usage(self._handle.run_id)


class ResearchRunProgressAPI(_RunReadoutBound):
    def get(self) -> dict[str, Any]:
        return self._handle._client.get_run_progress(
            self._handle.project_id,
            self._handle.run_id,
        )


class ResearchRunSnapshotsAPI(_RunReadoutBound):
    def get(
        self,
        *,
        detail: str = "control",
        event_limit: int = 40,
        actor_limit: int = 25,
        task_limit: int = 40,
        question_limit: int = 10,
        timeline_limit: int = 10,
        message_limit: int = 8,
    ) -> Any:
        if detail == "full":
            return self._handle._client.get_run_observability_snapshot_full(
                self._handle.project_id,
                self._handle.run_id,
            )
        return self._handle._client.get_run_observability_snapshot(
            self._handle.project_id,
            self._handle.run_id,
            detail_level=detail,
            event_limit=event_limit,
            actor_limit=actor_limit,
            task_limit=task_limit,
            question_limit=question_limit,
            timeline_limit=timeline_limit,
            message_limit=message_limit,
        )


class ResearchRunEventsObjectivesAPI(_RunReadoutBound):
    def list(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._handle.objective_events(limit=limit, cursor=cursor)


class ResearchRunEventsTasksAPI(_RunReadoutBound):
    def list(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._handle.task_events(limit=limit, cursor=cursor)


class ResearchRunEventsAPI(_RunReadoutBound):
    def __init__(self, handle: RunHandle) -> None:
        super().__init__(handle)
        self._objectives: ResearchRunEventsObjectivesAPI | None = None
        self._tasks: ResearchRunEventsTasksAPI | None = None

    @property
    def objectives(self) -> ResearchRunEventsObjectivesAPI:
        if self._objectives is None:
            self._objectives = ResearchRunEventsObjectivesAPI(self._handle)
        return self._objectives

    @property
    def tasks(self) -> ResearchRunEventsTasksAPI:
        if self._tasks is None:
            self._tasks = ResearchRunEventsTasksAPI(self._handle)
        return self._tasks

    def stream(
        self,
        *,
        transcript_cursor: str | None = None,
        view: str = "operator",
        last_event_id: str | None = None,
        timeout: float | None = None,
    ) -> Iterator[RunRuntimeStreamEvent]:
        return self._handle.stream_events(
            transcript_cursor=transcript_cursor,
            view=view,
            last_event_id=last_event_id,
            timeout=timeout,
        )


class ResearchRunTasksAPI(_RunReadoutBound):
    def list(
        self,
        *,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[Any]:
        return self._handle._client.list_task_summaries(
            self._handle.project_id,
            run_id=self._handle.run_id,
            kind=kind,
            limit=limit,
        )


class ResearchRunMessageQueueMessagesAPI(_RunReadoutBound):
    def list(
        self,
        *,
        thread_id: str | None = None,
        limit: int | None = None,
    ) -> list[Any]:
        return self._handle.message_queue_messages(thread_id=thread_id, limit=limit)

    def send(self, *, body: str, **kwargs: Any) -> dict[str, Any]:
        return self._handle.send_message(body=body, **kwargs)


class ResearchRunMessageQueueThreadsAPI(_RunReadoutBound):
    def list(self, *, limit: int | None = None) -> list[Any]:
        return self._handle.message_queue_threads(limit=limit)


class ResearchRunMessageQueueInteractionsAPI(_RunReadoutBound):
    def list(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[Any]:
        return self._handle.message_queue_interactions(status=status, limit=limit)


class ResearchRunMessageQueueAPI(_RunReadoutBound):
    def __init__(self, handle: RunHandle) -> None:
        super().__init__(handle)
        self._messages: ResearchRunMessageQueueMessagesAPI | None = None
        self._threads: ResearchRunMessageQueueThreadsAPI | None = None
        self._interactions: ResearchRunMessageQueueInteractionsAPI | None = None

    @property
    def messages(self) -> ResearchRunMessageQueueMessagesAPI:
        if self._messages is None:
            self._messages = ResearchRunMessageQueueMessagesAPI(self._handle)
        return self._messages

    @property
    def threads(self) -> ResearchRunMessageQueueThreadsAPI:
        if self._threads is None:
            self._threads = ResearchRunMessageQueueThreadsAPI(self._handle)
        return self._threads

    @property
    def interactions(self) -> ResearchRunMessageQueueInteractionsAPI:
        if self._interactions is None:
            self._interactions = ResearchRunMessageQueueInteractionsAPI(self._handle)
        return self._interactions


class ResearchRunRuntimeMessagesAPI(_RunReadoutBound):
    def list(
        self,
        *,
        status: str | None = None,
        viewer_role: str | None = None,
        viewer_target: str | List[str] | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        return self._handle._client.list_project_run_runtime_messages(
            self._handle.project_id,
            self._handle.run_id,
            status=status,
            viewer_role=viewer_role,
            viewer_target=viewer_target,
            limit=limit,
        )


class ResearchRunTranscriptAPI(_RunReadoutBound):
    def get(
        self,
        *,
        cursor: str | None = None,
        limit: int = 200,
        participant_session_id: str | None = None,
        view: str | None = None,
    ) -> dict[str, Any]:
        return self._handle._client.runs.transcript(
            self._handle.run_id,
            cursor=cursor,
            limit=limit,
            participant_session_id=participant_session_id,
            view=view,
        )

    def get_page(
        self,
        *,
        cursor: str | None = None,
        limit: int = 200,
        participant_session_id: str | None = None,
        view: str | None = None,
    ) -> "SyncPage[dict[str, Any]]":
        from synth_ai.sdk.pagination import SyncPage

        payload = self.get(
            cursor=cursor,
            limit=limit,
            participant_session_id=participant_session_id,
            view=view,
        )
        events = payload.get("events") if isinstance(payload, dict) else None
        normalized = [item for item in events if isinstance(item, dict)] if isinstance(events, list) else []
        next_cursor = (
            str(payload.get("next_cursor") or "").strip() or None
            if isinstance(payload, dict)
            else None
        )
        return SyncPage(
            items=normalized,
            next_cursor=next_cursor,
            has_more=bool(next_cursor),
        )


class ResearchRunMilestonesAPI(_RunReadoutBound):
    def list_primary_parent(self) -> list[dict[str, Any]]:
        return self._handle._client.list_run_primary_parent_milestones(
            self._handle.run_id,
            project_id=self._handle.project_id,
        )


class ResearchRunWorkProductsContentAPI(_RunReadoutBound):
    def get(
        self,
        work_product_id: str,
        *,
        as_text: bool = True,
    ) -> str | bytes:
        return self._handle._client.work_products.content(
            work_product_id,
            as_text=as_text,
        )


class ResearchRunTrainedModelsAPI(_RunReadoutBound):
    def list(self) -> list[Any]:
        return self._handle._client.trained_models.list_for_run(self._handle.run_id)


class ResearchRunWorkProductsEvalPackagesAPI(_RunReadoutBound):
    def list(self) -> list[Any]:
        return self._handle._client.work_products.list_container_eval_packages(
            self._handle.project_id,
            self._handle.run_id,
        )


class ResearchRunWorkProductsAPI(_RunReadoutBound):
    def __init__(self, handle: RunHandle) -> None:
        super().__init__(handle)
        self._content: ResearchRunWorkProductsContentAPI | None = None
        self._eval_packages: ResearchRunWorkProductsEvalPackagesAPI | None = None

    @property
    def content(self) -> ResearchRunWorkProductsContentAPI:
        if self._content is None:
            self._content = ResearchRunWorkProductsContentAPI(self._handle)
        return self._content

    @property
    def eval_packages(self) -> ResearchRunWorkProductsEvalPackagesAPI:
        if self._eval_packages is None:
            self._eval_packages = ResearchRunWorkProductsEvalPackagesAPI(self._handle)
        return self._eval_packages

    def list(self) -> list[Any]:
        return self._handle._client.work_products.list_for_run(
            self._handle.project_id,
            self._handle.run_id,
        )


class ResearchRunArtifactsManifestAPI(_RunReadoutBound):
    def get(self) -> Any:
        return self._handle._client.get_run_artifact_manifest(
            self._handle.run_id,
            project_id=self._handle.project_id,
        )


class ResearchRunArtifactsContentAPI(_RunReadoutBound):
    def get(
        self,
        artifact_id: str,
        *,
        as_text: bool = True,
    ) -> str | bytes:
        return self._handle._client.get_artifact_content(
            artifact_id,
            as_text=as_text,
        )


class ResearchRunArtifactsAPI(_RunReadoutBound):
    def __init__(self, handle: RunHandle) -> None:
        super().__init__(handle)
        self._manifest: ResearchRunArtifactsManifestAPI | None = None
        self._content: ResearchRunArtifactsContentAPI | None = None

    @property
    def manifest(self) -> ResearchRunArtifactsManifestAPI:
        if self._manifest is None:
            self._manifest = ResearchRunArtifactsManifestAPI(self._handle)
        return self._manifest

    @property
    def content(self) -> ResearchRunArtifactsContentAPI:
        if self._content is None:
            self._content = ResearchRunArtifactsContentAPI(self._handle)
        return self._content

    def list(
        self,
        *,
        artifact_type: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[Any]:
        return self._handle._client.list_run_artifacts(
            self._handle.run_id,
            project_id=self._handle.project_id,
            artifact_type=artifact_type,
            limit=limit,
            cursor=cursor,
        )


class ResearchRunResultsAPI(_RunReadoutBound):
    def get(self) -> dict[str, Any]:
        return self._handle._client.get_run_results(
            self._handle.project_id,
            self._handle.run_id,
        )


class ResearchRunLogsAPI(_RunReadoutBound):
    def list(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._handle._client.get_run_logs(
            self._handle.project_id,
            self._handle.run_id,
            limit=limit,
            cursor=cursor,
        )


class ResearchRunOrchestratorAPI(_RunReadoutBound):
    def get(self) -> dict[str, Any]:
        return self._handle._client.get_run_orchestrator(
            self._handle.project_id,
            self._handle.run_id,
        )


class ResearchRunWorkspaceAPI(_RunReadoutBound):
    def download(
        self,
        destination: str,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return self._handle._client.download_run_workspace_archive(
            self._handle.project_id,
            self._handle.run_id,
            destination,
            timeout_seconds=timeout_seconds,
        )


class ResearchRunCodeAPI(_RunReadoutBound):
    def download(self, path: str) -> dict[str, Any]:
        return self._handle.download_code(path)


class ResearchRunActorsAPI(_RunReadoutBound):
    def list(self) -> dict[str, Any]:
        return self._handle.actor_inventory()


class ResearchRunEvidenceAPI(_RunReadoutBound):
    def get(
        self,
        *,
        runtime_timeline_limit: int | None = None,
        logical_timeline_limit: int | None = None,
        transcript_limit: int | None = None,
        reconciliation_limit: int | None = None,
    ) -> dict[str, Any]:
        return self._handle.operator_evidence(
            runtime_timeline_limit=runtime_timeline_limit,
            logical_timeline_limit=logical_timeline_limit,
            transcript_limit=transcript_limit,
            reconciliation_limit=reconciliation_limit,
        )


class ResearchRunAuthorityAPI(_RunReadoutBound):
    def get(self, *, include_runtime_authority: bool = False) -> Any:
        return self._handle.authority_readouts(
            include_runtime_authority=include_runtime_authority,
        )


class ResearchRunExecutionAPI(_RunReadoutBound):
    def get(self, **kwargs: Any) -> Any:
        return self._handle._client.get_run_execution(
            self._handle.project_id,
            self._handle.run_id,
            **kwargs,
        )


class ResearchRunTickingAPI(_RunReadoutBound):
    def get(self) -> Any:
        return self._handle.ticking()

    def set(self, update: Any = None, **kwargs: Any) -> Any:
        return self._handle.set_ticking(update, **kwargs)


class ResearchRunReadoutsMixin:
    """Lazy nested readout namespaces for a run handle."""

    _usage_api: ResearchRunUsageAPI | None = None
    _progress_api: ResearchRunProgressAPI | None = None
    _snapshots_api: ResearchRunSnapshotsAPI | None = None
    _events_api: ResearchRunEventsAPI | None = None
    _tasks_api: ResearchRunTasksAPI | None = None
    _message_queue_api: ResearchRunMessageQueueAPI | None = None
    _messages_api: ResearchRunRuntimeMessagesAPI | None = None
    _transcript_api: ResearchRunTranscriptAPI | None = None
    _work_products_api: ResearchRunWorkProductsAPI | None = None
    _trained_models_api: ResearchRunTrainedModelsAPI | None = None
    _artifacts_api: ResearchRunArtifactsAPI | None = None
    _results_api: ResearchRunResultsAPI | None = None
    _logs_api: ResearchRunLogsAPI | None = None
    _orchestrator_api: ResearchRunOrchestratorAPI | None = None
    _workspace_api: ResearchRunWorkspaceAPI | None = None
    _code_api: ResearchRunCodeAPI | None = None
    _actors_api: ResearchRunActorsAPI | None = None
    _evidence_api: ResearchRunEvidenceAPI | None = None
    _authority_api: ResearchRunAuthorityAPI | None = None
    _execution_api: ResearchRunExecutionAPI | None = None
    _milestones_api: ResearchRunMilestonesAPI | None = None
    _ticking_api: ResearchRunTickingAPI | None = None

    @property
    def usage(self) -> ResearchRunUsageAPI:
        if self._usage_api is None:
            self._usage_api = ResearchRunUsageAPI(self)  # type: ignore[arg-type]
        return self._usage_api

    @property
    def progress(self) -> ResearchRunProgressAPI:
        if self._progress_api is None:
            self._progress_api = ResearchRunProgressAPI(self)  # type: ignore[arg-type]
        return self._progress_api

    @property
    def snapshots(self) -> ResearchRunSnapshotsAPI:
        if self._snapshots_api is None:
            self._snapshots_api = ResearchRunSnapshotsAPI(self)  # type: ignore[arg-type]
        return self._snapshots_api

    @property
    def events(self) -> ResearchRunEventsAPI:
        if self._events_api is None:
            self._events_api = ResearchRunEventsAPI(self)  # type: ignore[arg-type]
        return self._events_api

    @property
    def tasks(self) -> ResearchRunTasksAPI:
        if self._tasks_api is None:
            self._tasks_api = ResearchRunTasksAPI(self)  # type: ignore[arg-type]
        return self._tasks_api

    @property
    def message_queue(self) -> ResearchRunMessageQueueAPI:
        if self._message_queue_api is None:
            self._message_queue_api = ResearchRunMessageQueueAPI(self)  # type: ignore[arg-type]
        return self._message_queue_api

    @property
    def messages(self) -> ResearchRunRuntimeMessagesAPI:
        if self._messages_api is None:
            self._messages_api = ResearchRunRuntimeMessagesAPI(self)  # type: ignore[arg-type]
        return self._messages_api

    @property
    def transcript(self) -> ResearchRunTranscriptAPI:
        if self._transcript_api is None:
            self._transcript_api = ResearchRunTranscriptAPI(self)  # type: ignore[arg-type]
        return self._transcript_api

    @property
    def work_products(self) -> ResearchRunWorkProductsAPI:
        if self._work_products_api is None:
            self._work_products_api = ResearchRunWorkProductsAPI(self)  # type: ignore[arg-type]
        return self._work_products_api

    @property
    def trained_models(self) -> ResearchRunTrainedModelsAPI:
        if self._trained_models_api is None:
            self._trained_models_api = ResearchRunTrainedModelsAPI(self)  # type: ignore[arg-type]
        return self._trained_models_api

    @property
    def artifacts(self) -> ResearchRunArtifactsAPI:
        if self._artifacts_api is None:
            self._artifacts_api = ResearchRunArtifactsAPI(self)  # type: ignore[arg-type]
        return self._artifacts_api

    @property
    def results(self) -> ResearchRunResultsAPI:
        if self._results_api is None:
            self._results_api = ResearchRunResultsAPI(self)  # type: ignore[arg-type]
        return self._results_api

    @property
    def logs(self) -> ResearchRunLogsAPI:
        if self._logs_api is None:
            self._logs_api = ResearchRunLogsAPI(self)  # type: ignore[arg-type]
        return self._logs_api

    @property
    def orchestrator(self) -> ResearchRunOrchestratorAPI:
        if self._orchestrator_api is None:
            self._orchestrator_api = ResearchRunOrchestratorAPI(self)  # type: ignore[arg-type]
        return self._orchestrator_api

    @property
    def workspace(self) -> ResearchRunWorkspaceAPI:
        if self._workspace_api is None:
            self._workspace_api = ResearchRunWorkspaceAPI(self)  # type: ignore[arg-type]
        return self._workspace_api

    @property
    def code(self) -> ResearchRunCodeAPI:
        if self._code_api is None:
            self._code_api = ResearchRunCodeAPI(self)  # type: ignore[arg-type]
        return self._code_api

    @property
    def actors(self) -> ResearchRunActorsAPI:
        if self._actors_api is None:
            self._actors_api = ResearchRunActorsAPI(self)  # type: ignore[arg-type]
        return self._actors_api

    @property
    def evidence(self) -> ResearchRunEvidenceAPI:
        if self._evidence_api is None:
            self._evidence_api = ResearchRunEvidenceAPI(self)  # type: ignore[arg-type]
        return self._evidence_api

    @property
    def authority(self) -> ResearchRunAuthorityAPI:
        if self._authority_api is None:
            self._authority_api = ResearchRunAuthorityAPI(self)  # type: ignore[arg-type]
        return self._authority_api

    @property
    def execution(self) -> ResearchRunExecutionAPI:
        if self._execution_api is None:
            self._execution_api = ResearchRunExecutionAPI(self)  # type: ignore[arg-type]
        return self._execution_api

    @property
    def milestones(self) -> ResearchRunMilestonesAPI:
        if self._milestones_api is None:
            self._milestones_api = ResearchRunMilestonesAPI(self)  # type: ignore[arg-type]
        return self._milestones_api

    @property
    def ticking(self) -> ResearchRunTickingAPI:
        if self._ticking_api is None:
            self._ticking_api = ResearchRunTickingAPI(self)  # type: ignore[arg-type]
        return self._ticking_api


def _deprecated_method(name: str, replacement: str) -> None:
    warnings.warn(
        f"{name} is deprecated; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


__all__ = [
    "ResearchRunReadoutsMixin",
    "ResearchRunUsageAPI",
    "ResearchRunProgressAPI",
    "ResearchRunSnapshotsAPI",
    "ResearchRunEventsAPI",
    "ResearchRunTasksAPI",
    "ResearchRunMessageQueueAPI",
    "ResearchRunTranscriptAPI",
    "ResearchRunWorkProductsAPI",
    "ResearchRunArtifactsAPI",
    "ResearchRunResultsAPI",
    "ResearchRunLogsAPI",
    "ResearchRunMilestonesAPI",
    "ResearchRunOrchestratorAPI",
]
