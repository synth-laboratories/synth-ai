"""Nested run readout namespaces on ``ResearchRunHandle``."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any, List

from synth_ai.core.research._legacy.models.canonical_usage import (
    SmrResourceLimitProgress,
    SmrResourceLimits,
    SmrRunUsage,
)
from synth_ai.core.research._legacy.models.operator_evidence import SmrRunOperatorEvidence
from synth_ai.core.research._legacy.models.run_diagnostics import (
    SmrRunActorUsage,
    SmrRunCostSummary,
)
from synth_ai.core.research._legacy.models.run_events import RunRuntimeStreamEvent
from synth_ai.core.research.models import (
    ResearchArtifact,
    ResearchArtifactManifest,
    ResearchRunProgress,
    ResearchWorkProduct,
)
from synth_ai.sdk.pagination import SyncPage


class _RunReadoutBound:
    def __init__(self, handle: Any) -> None:
        self._handle = handle


class ResearchRunUsageActorsAPI(_RunReadoutBound):
    """Per-actor usage breakdown for a run."""

    def get(self) -> SmrRunActorUsage:
        """Return token and cost usage grouped by runtime actor."""
        return self._handle.actor_usage()


class ResearchRunUsageCostAPI(_RunReadoutBound):
    """Run-level cost summary readouts."""

    def get(self) -> SmrRunCostSummary:
        """Return aggregated cost fields for the run."""
        return self._handle.cost_summary()


class ResearchRunUsageLimitsAPI(_RunReadoutBound):
    """Org and run resource limit readouts."""

    def get(self) -> SmrResourceLimits:
        """Return configured resource limits applicable to the run."""
        return self._handle.resource_limits()

    def progress(self) -> SmrResourceLimitProgress:
        """Return progress toward resource limits (tokens, spend, concurrency)."""
        return self._handle.progress_toward_resource_limits()


class ResearchRunUsageAPI(_RunReadoutBound):
    """Token, cost, and limit readouts for a run."""

    def __init__(self, handle: Any) -> None:
        super().__init__(handle)
        self._actors: ResearchRunUsageActorsAPI | None = None
        self._cost: ResearchRunUsageCostAPI | None = None
        self._limits: ResearchRunUsageLimitsAPI | None = None

    @property
    def actors(self) -> ResearchRunUsageActorsAPI:
        """Per-actor usage slice."""
        if self._actors is None:
            self._actors = ResearchRunUsageActorsAPI(self._handle)
        return self._actors

    @property
    def cost(self) -> ResearchRunUsageCostAPI:
        """Run cost summary slice."""
        if self._cost is None:
            self._cost = ResearchRunUsageCostAPI(self._handle)
        return self._cost

    @property
    def limits(self) -> ResearchRunUsageLimitsAPI:
        """Resource limit and progress slice."""
        if self._limits is None:
            self._limits = ResearchRunUsageLimitsAPI(self._handle)
        return self._limits

    def get(self) -> SmrRunUsage:
        """Return canonical usage totals for the run."""
        return self._handle._client.get_run_usage(self._handle.run_id)


class ResearchRunProgressAPI(_RunReadoutBound):
    """High-level progress summary for dashboards and polling loops."""

    def get(self) -> dict[str, Any]:
        """Return the backward-compatible raw progress payload."""
        return self._handle._client.get_run_progress(
            self._handle.project_id,
            self._handle.run_id,
        )

    def get_typed(self) -> ResearchRunProgress:
        """Return typed state, phase, stalls, tasks, and recommended actions."""
        return ResearchRunProgress.from_wire(self.get())


class ResearchRunSnapshotsAPI(_RunReadoutBound):
    """Observability snapshots (control vs full detail)."""

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
        """Return an observability snapshot for the run.

        Args:
            detail: ``"control"`` for operator dashboard fields or ``"full"`` for
                the expanded observability projection.
        """
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
    """Objective lifecycle events emitted during a run."""

    def list(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List objective events with optional cursor pagination.

        Args:
            limit: Maximum events to return on this page.
            cursor: Opaque cursor from a prior response.

        Returns:
            Event page payload including items and an optional next cursor.
        """
        return self._handle.objective_events(limit=limit, cursor=cursor)


class ResearchRunEventsTasksAPI(_RunReadoutBound):
    """Task lifecycle events emitted during a run."""

    def list(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List task events with optional cursor pagination.

        Args:
            limit: Maximum events to return on this page.
            cursor: Opaque cursor from a prior response.

        Returns:
            Event page payload including items and an optional next cursor.
        """
        return self._handle.task_events(limit=limit, cursor=cursor)


class ResearchRunEventsAPI(_RunReadoutBound):
    """Structured and streaming runtime events for a run."""

    def __init__(self, handle: Any) -> None:
        super().__init__(handle)
        self._objectives: ResearchRunEventsObjectivesAPI | None = None
        self._tasks: ResearchRunEventsTasksAPI | None = None

    @property
    def objectives(self) -> ResearchRunEventsObjectivesAPI:
        """Objective-scoped event list."""
        if self._objectives is None:
            self._objectives = ResearchRunEventsObjectivesAPI(self._handle)
        return self._objectives

    @property
    def tasks(self) -> ResearchRunEventsTasksAPI:
        """Task-scoped event list."""
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
        """Stream runtime events over SSE.

        Args:
            transcript_cursor: Resume streaming after this transcript cursor.
            view: Projection name (for example ``"operator"``).
            last_event_id: Resume after this event id when reconnecting.
            timeout: Optional read timeout in seconds.

        Yields:
            Parsed runtime stream events.
        """
        return self._handle.stream_events(
            transcript_cursor=transcript_cursor,
            view=view,
            last_event_id=last_event_id,
            timeout=timeout,
        )


class ResearchRunTasksAPI(_RunReadoutBound):
    """Task summaries attached to a run."""

    def list(
        self,
        *,
        kind: str | None = None,
        limit: int | None = None,
    ) -> List[Any]:
        """List task summaries for the run.

        Args:
            kind: Optional task kind filter.
            limit: Maximum summaries to return.
        """
        return self._handle._client.list_task_summaries(
            self._handle.project_id,
            run_id=self._handle.run_id,
            kind=kind,
            limit=limit,
        )


class ResearchRunMessageQueueMessagesAPI(_RunReadoutBound):
    """Outbound operator messages on the run message queue."""

    def list(
        self,
        *,
        thread_id: str | None = None,
        limit: int | None = None,
    ) -> List[Any]:
        """List queued messages, optionally scoped to a thread."""
        return self._handle.message_queue_messages(thread_id=thread_id, limit=limit)

    def send(self, *, body: str, **kwargs: Any) -> dict[str, Any]:
        """Publish a message to the run message queue.

        Args:
            body: Message body text.
            **kwargs: Additional wire fields forwarded to the backend.
        """
        return self._handle.send_message(body=body, **kwargs)


class ResearchRunMessageQueueThreadsAPI(_RunReadoutBound):
    """Message queue threads for operator steering."""

    def list(self, *, limit: int | None = None) -> List[Any]:
        """List message queue threads for the run."""
        return self._handle.message_queue_threads(limit=limit)


class ResearchRunMessageQueueInteractionsAPI(_RunReadoutBound):
    """Pending and completed message queue interactions."""

    def list(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> List[Any]:
        """List interactions, optionally filtered by status."""
        return self._handle.message_queue_interactions(status=status, limit=limit)


class ResearchRunMessageQueueAPI(_RunReadoutBound):
    """Operator steering via threads, messages, and interactions."""

    def __init__(self, handle: Any) -> None:
        super().__init__(handle)
        self._messages: ResearchRunMessageQueueMessagesAPI | None = None
        self._threads: ResearchRunMessageQueueThreadsAPI | None = None
        self._interactions: ResearchRunMessageQueueInteractionsAPI | None = None

    @property
    def messages(self) -> ResearchRunMessageQueueMessagesAPI:
        """Outbound messages API."""
        if self._messages is None:
            self._messages = ResearchRunMessageQueueMessagesAPI(self._handle)
        return self._messages

    @property
    def threads(self) -> ResearchRunMessageQueueThreadsAPI:
        """Thread listing API."""
        if self._threads is None:
            self._threads = ResearchRunMessageQueueThreadsAPI(self._handle)
        return self._threads

    @property
    def interactions(self) -> ResearchRunMessageQueueInteractionsAPI:
        """Interaction listing API."""
        if self._interactions is None:
            self._interactions = ResearchRunMessageQueueInteractionsAPI(self._handle)
        return self._interactions


class ResearchRunRuntimeMessagesAPI(_RunReadoutBound):
    """Runtime messages visible to operators and viewers."""

    def list(
        self,
        *,
        status: str | None = None,
        viewer_role: str | None = None,
        viewer_target: str | List[str] | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        """List runtime messages for the run.

        Args:
            status: Optional delivery or read status filter.
            viewer_role: Role used to project the message list.
            viewer_target: Optional target id or list of targets.
            limit: Maximum messages to return.
        """
        return self._handle._client.list_project_run_runtime_messages(
            self._handle.project_id,
            self._handle.run_id,
            status=status,
            viewer_role=viewer_role,
            viewer_target=viewer_target,
            limit=limit,
        )


class ResearchRunTranscriptAPI(_RunReadoutBound):
    """Transcript pages and cursor-based pagination for run events."""

    def get(
        self,
        *,
        cursor: str | None = None,
        limit: int = 200,
        participant_session_id: str | None = None,
        view: str | None = None,
    ) -> dict[str, Any]:
        """Fetch a transcript page (use ``get_page`` for ``SyncPage`` iteration)."""
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
    ) -> SyncPage[dict[str, Any]]:
        """Fetch a transcript page wrapped as ``SyncPage`` for iteration."""
        payload = self.get(
            cursor=cursor,
            limit=limit,
            participant_session_id=participant_session_id,
            view=view,
        )
        events = payload.get("events") if isinstance(payload, dict) else None
        normalized = (
            [item for item in events if isinstance(item, dict)] if isinstance(events, list) else []
        )
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
    """Run-scoped milestone readouts."""

    def list_primary_parent(self) -> List[dict[str, Any]]:
        """List primary parent milestones linked to the run."""
        return self._handle._client.list_run_primary_parent_milestones(
            self._handle.run_id,
            project_id=self._handle.project_id,
        )


class ResearchRunWorkProductsContentAPI(_RunReadoutBound):
    """Fetch work product payload bytes or text."""

    def get(
        self,
        work_product_id: str,
        *,
        as_text: bool = True,
    ) -> str | bytes:
        """Return work product content.

        Args:
            work_product_id: Work product identifier from ``list``.
            as_text: When ``True``, decode as UTF-8 text; otherwise return bytes.
        """
        return self._handle._client.work_products.content(
            work_product_id,
            as_text=as_text,
        )


class ResearchRunTrainedModelsAPI(_RunReadoutBound):
    """Trained model artifacts produced by a run."""

    def list(self) -> List[Any]:
        """List trained models registered for the run."""
        return self._handle._client.trained_models.list_for_run(self._handle.run_id)


class ResearchRunWorkProductsEvalPackagesAPI(_RunReadoutBound):
    """Container eval packages attached to run work products."""

    def list(self) -> List[Any]:
        """List eval packages exported from the run workspace."""
        return self._handle._client.work_products.list_container_eval_packages(
            self._handle.project_id,
            self._handle.run_id,
        )


class ResearchRunWorkProductsAPI(_RunReadoutBound):
    """Work products and derived outputs from a run."""

    def __init__(self, handle: Any) -> None:
        super().__init__(handle)
        self._content: ResearchRunWorkProductsContentAPI | None = None
        self._eval_packages: ResearchRunWorkProductsEvalPackagesAPI | None = None

    @property
    def content(self) -> ResearchRunWorkProductsContentAPI:
        """Download work product bodies."""
        if self._content is None:
            self._content = ResearchRunWorkProductsContentAPI(self._handle)
        return self._content

    @property
    def eval_packages(self) -> ResearchRunWorkProductsEvalPackagesAPI:
        """List container eval packages."""
        if self._eval_packages is None:
            self._eval_packages = ResearchRunWorkProductsEvalPackagesAPI(self._handle)
        return self._eval_packages

    def list(self) -> List[ResearchWorkProduct]:
        """List work product metadata for the run."""
        return self._handle._client.work_products.list_for_run(
            self._handle.project_id,
            self._handle.run_id,
        )


class ResearchRunHostedArtifactsAPI(_RunReadoutBound):
    """Hosted artifact receipt and operator actions for a run."""

    def get(self) -> dict[str, Any]:
        """Return hosted artifact status for this run."""
        return self._handle._client.get_run_hosted_artifact(self._handle.run_id)

    def content(
        self,
        hosted_artifact_id: str | None = None,
        *,
        as_text: bool = True,
    ) -> str | bytes:
        """Fetch hosted HTML for this run's artifact."""
        artifact_id = hosted_artifact_id
        if artifact_id is None:
            status = self.get()
            artifact_id = str(status.get("hosted_artifact_id") or "").strip()
            if not artifact_id:
                raise ValueError("run has no hosted_artifact_id yet")
        payload = self._handle._client.get_hosted_artifact_content(artifact_id)
        encoding = str(payload.get("encoding") or "utf-8")
        content = payload["content"]
        if encoding == "base64":
            import base64

            raw = base64.b64decode(str(content))
            return raw.decode("utf-8") if as_text else raw
        text = str(content)
        return text if as_text else text.encode("utf-8")

    def publish_public(
        self,
        slug: str,
        *,
        hosted_artifact_id: str | None = None,
        kind: str = "result",
        theme: str | None = None,
        summary: str | None = None,
        factory_id: str | None = None,
        effort_id: str | None = None,
    ) -> dict[str, Any]:
        """Promote this run's hosted artifact to the public index."""
        artifact_id = hosted_artifact_id
        if artifact_id is None:
            status = self.get()
            artifact_id = str(status.get("hosted_artifact_id") or "").strip()
            if not artifact_id:
                raise ValueError("run has no hosted_artifact_id yet")
        return self._handle._client.publish_hosted_artifact_public(
            artifact_id,
            slug=slug,
            kind=kind,
            theme=theme,
            summary=summary,
            factory_id=factory_id,
            effort_id=effort_id,
        )

    def assign_reviewer(
        self,
        reason: str,
        *,
        hosted_artifact_id: str | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Dispatch an artifact_reviewer for this run's hosted artifact."""
        artifact_id = hosted_artifact_id
        if artifact_id is None:
            status = self.get()
            artifact_id = str(status.get("hosted_artifact_id") or "").strip()
            if not artifact_id:
                raise ValueError("run has no hosted_artifact_id yet")
        return self._handle._client.assign_hosted_artifact_reviewer(
            artifact_id,
            reason=reason,
            summary=summary,
        )


class ResearchRunArtifactsManifestAPI(_RunReadoutBound):
    """Artifact manifest for a run."""

    def get(self) -> ResearchArtifactManifest:
        """Return the artifact manifest describing available run outputs."""
        return self._handle._client.get_run_artifact_manifest(
            self._handle.run_id,
            project_id=self._handle.project_id,
        )


class ResearchRunArtifactsContentAPI(_RunReadoutBound):
    """Download individual run artifacts."""

    def get(
        self,
        artifact_id: str,
        *,
        as_text: bool = True,
    ) -> str | bytes:
        """Return artifact content by id.

        Args:
            artifact_id: Artifact identifier from ``list`` or the manifest.
            as_text: When ``True``, decode as UTF-8 text; otherwise return bytes.
        """
        return self._handle._client.get_artifact_content(
            artifact_id,
            as_text=as_text,
        )


class ResearchRunArtifactsAPI(_RunReadoutBound):
    """Run artifacts listing, manifest, and content download."""

    def __init__(self, handle: Any) -> None:
        super().__init__(handle)
        self._manifest: ResearchRunArtifactsManifestAPI | None = None
        self._content: ResearchRunArtifactsContentAPI | None = None

    @property
    def manifest(self) -> ResearchRunArtifactsManifestAPI:
        """Artifact manifest API."""
        if self._manifest is None:
            self._manifest = ResearchRunArtifactsManifestAPI(self._handle)
        return self._manifest

    @property
    def content(self) -> ResearchRunArtifactsContentAPI:
        """Artifact content download API."""
        if self._content is None:
            self._content = ResearchRunArtifactsContentAPI(self._handle)
        return self._content

    def list(
        self,
        *,
        artifact_type: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> List[ResearchArtifact]:
        """List artifacts for the run with optional type filter."""
        return self._handle._client.list_run_artifacts(
            self._handle.run_id,
            project_id=self._handle.project_id,
            artifact_type=artifact_type,
            limit=limit,
            cursor=cursor,
        )


class ResearchRunResultsAPI(_RunReadoutBound):
    """Final run results and outcome payload."""

    def get(self) -> dict[str, Any]:
        """Return the run results document when execution has finished."""
        return self._handle._client.get_run_results(
            self._handle.project_id,
            self._handle.run_id,
        )


class ResearchRunLogsAPI(_RunReadoutBound):
    """Structured run logs for debugging and audit."""

    def list(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List log records with optional cursor pagination."""
        return self._handle._client.get_run_logs(
            self._handle.project_id,
            self._handle.run_id,
            limit=limit,
            cursor=cursor,
        )


class ResearchRunOrchestratorAPI(_RunReadoutBound):
    """Orchestrator state and routing metadata for a run."""

    def get(self) -> dict[str, Any]:
        """Return orchestrator readouts for the run."""
        return self._handle._client.get_run_orchestrator(
            self._handle.project_id,
            self._handle.run_id,
        )


class ResearchRunWorkspaceAPI(_RunReadoutBound):
    """Run workspace archive download."""

    def download(
        self,
        destination: str,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Download the run workspace archive to a local path.

        Args:
            destination: Local filesystem path for the archive.
            timeout_seconds: Optional download timeout.
        """
        return self._handle._client.download_run_workspace_archive(
            self._handle.project_id,
            self._handle.run_id,
            destination,
            timeout_seconds=timeout_seconds,
        )


class ResearchRunCodeAPI(_RunReadoutBound):
    """Run code archive download."""

    def download(self, path: str) -> dict[str, Any]:
        """Download run code to the given local path."""
        return self._handle.download_code(path)


class ResearchRunActorsAPI(_RunReadoutBound):
    """Runtime actor inventory for a run."""

    def list(self) -> dict[str, Any]:
        """List actors participating in the run."""
        return self._handle.actor_inventory()


class ResearchRunEvidenceAPI(_RunReadoutBound):
    """Operator evidence bundle for debugging run behavior."""

    def get(
        self,
        *,
        runtime_timeline_limit: int | None = None,
        logical_timeline_limit: int | None = None,
        transcript_limit: int | None = None,
        reconciliation_limit: int | None = None,
    ) -> SmrRunOperatorEvidence:
        """Return operator evidence with optional per-section limits."""
        return self._handle.operator_evidence(
            runtime_timeline_limit=runtime_timeline_limit,
            logical_timeline_limit=logical_timeline_limit,
            transcript_limit=transcript_limit,
            reconciliation_limit=reconciliation_limit,
        )


class ResearchRunAuthorityAPI(_RunReadoutBound):
    """Authority and permission readouts for operators."""

    def get(self, *, include_runtime_authority: bool = False) -> Any:
        """Return authority readouts for the run.

        Args:
            include_runtime_authority: Include live runtime authority fields when
                available.
        """
        return self._handle.authority_readouts(
            include_runtime_authority=include_runtime_authority,
        )


class ResearchRunExecutionAPI(_RunReadoutBound):
    """Execution state and worker routing for a run."""

    def get(self, **kwargs: Any) -> Any:
        """Return execution readouts for the run."""
        return self._handle._client.get_run_execution(
            self._handle.project_id,
            self._handle.run_id,
            **kwargs,
        )


class ResearchRunTickingAPI(_RunReadoutBound):
    """Run ticking / heartbeat controls for long-running workloads."""

    def get(self) -> Any:
        """Return current ticking state for the run."""
        return self._handle.ticking()

    def set(self, update: Any = None, **kwargs: Any) -> Any:
        """Update ticking configuration for the run."""
        return self._handle.set_ticking(update, **kwargs)


class ResearchRunReadoutsMixin:
    """Lazy nested readout namespaces on :class:`ResearchRunHandle`.

    Access via ``handle.usage``, ``handle.progress``, ``handle.snapshots``,
    ``handle.transcript``, ``handle.message_queue``, etc.
    """

    _usage_api: ResearchRunUsageAPI | None = None
    _progress_api: ResearchRunProgressAPI | None = None
    _snapshots_api: ResearchRunSnapshotsAPI | None = None
    _events_api: ResearchRunEventsAPI | None = None
    _tasks_api: ResearchRunTasksAPI | None = None
    _message_queue_api: ResearchRunMessageQueueAPI | None = None
    _messages_api: ResearchRunRuntimeMessagesAPI | None = None
    _transcript_api: ResearchRunTranscriptAPI | None = None
    _work_products_api: ResearchRunWorkProductsAPI | None = None
    _hosted_artifact_api: ResearchRunHostedArtifactsAPI | None = None
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
        """Usage, cost, and limit readouts."""
        if self._usage_api is None:
            self._usage_api = ResearchRunUsageAPI(self)  # type: ignore[arg-type]
        return self._usage_api

    @property
    def progress(self) -> ResearchRunProgressAPI:
        """Coarse progress for polling UIs."""
        if self._progress_api is None:
            self._progress_api = ResearchRunProgressAPI(self)  # type: ignore[arg-type]
        return self._progress_api

    @property
    def snapshots(self) -> ResearchRunSnapshotsAPI:
        """Observability snapshots (control or full)."""
        if self._snapshots_api is None:
            self._snapshots_api = ResearchRunSnapshotsAPI(self)  # type: ignore[arg-type]
        return self._snapshots_api

    @property
    def events(self) -> ResearchRunEventsAPI:
        """Structured and streaming runtime events."""
        if self._events_api is None:
            self._events_api = ResearchRunEventsAPI(self)  # type: ignore[arg-type]
        return self._events_api

    @property
    def tasks(self) -> ResearchRunTasksAPI:
        """Task summaries for the run."""
        if self._tasks_api is None:
            self._tasks_api = ResearchRunTasksAPI(self)  # type: ignore[arg-type]
        return self._tasks_api

    @property
    def message_queue(self) -> ResearchRunMessageQueueAPI:
        """Operator message queue threads and outbound messages."""
        if self._message_queue_api is None:
            self._message_queue_api = ResearchRunMessageQueueAPI(self)  # type: ignore[arg-type]
        return self._message_queue_api

    @property
    def messages(self) -> ResearchRunRuntimeMessagesAPI:
        """Runtime messages visible to operators."""
        if self._messages_api is None:
            self._messages_api = ResearchRunRuntimeMessagesAPI(self)  # type: ignore[arg-type]
        return self._messages_api

    @property
    def transcript(self) -> ResearchRunTranscriptAPI:
        """Transcript pages with optional cursor pagination."""
        if self._transcript_api is None:
            self._transcript_api = ResearchRunTranscriptAPI(self)  # type: ignore[arg-type]
        return self._transcript_api

    @property
    def work_products(self) -> ResearchRunWorkProductsAPI:
        """Work products produced by the run."""
        if self._work_products_api is None:
            self._work_products_api = ResearchRunWorkProductsAPI(self)  # type: ignore[arg-type]
        return self._work_products_api

    @property
    def hosted_artifact(self) -> ResearchRunHostedArtifactsAPI:
        """Hosted Open Research artifact receipt for the run."""
        if self._hosted_artifact_api is None:
            self._hosted_artifact_api = ResearchRunHostedArtifactsAPI(self)  # type: ignore[arg-type]
        return self._hosted_artifact_api

    @property
    def trained_models(self) -> ResearchRunTrainedModelsAPI:
        """Trained models registered for the run."""
        if self._trained_models_api is None:
            self._trained_models_api = ResearchRunTrainedModelsAPI(self)  # type: ignore[arg-type]
        return self._trained_models_api

    @property
    def artifacts(self) -> ResearchRunArtifactsAPI:
        """Run artifacts listing and download."""
        if self._artifacts_api is None:
            self._artifacts_api = ResearchRunArtifactsAPI(self)  # type: ignore[arg-type]
        return self._artifacts_api

    @property
    def results(self) -> ResearchRunResultsAPI:
        """Final run results when execution completes."""
        if self._results_api is None:
            self._results_api = ResearchRunResultsAPI(self)  # type: ignore[arg-type]
        return self._results_api

    @property
    def logs(self) -> ResearchRunLogsAPI:
        """Structured run logs."""
        if self._logs_api is None:
            self._logs_api = ResearchRunLogsAPI(self)  # type: ignore[arg-type]
        return self._logs_api

    @property
    def orchestrator(self) -> ResearchRunOrchestratorAPI:
        """Orchestrator readouts for the run."""
        if self._orchestrator_api is None:
            self._orchestrator_api = ResearchRunOrchestratorAPI(self)  # type: ignore[arg-type]
        return self._orchestrator_api

    @property
    def workspace(self) -> ResearchRunWorkspaceAPI:
        """Run workspace archive download."""
        if self._workspace_api is None:
            self._workspace_api = ResearchRunWorkspaceAPI(self)  # type: ignore[arg-type]
        return self._workspace_api

    @property
    def code(self) -> ResearchRunCodeAPI:
        """Run code archive download."""
        if self._code_api is None:
            self._code_api = ResearchRunCodeAPI(self)  # type: ignore[arg-type]
        return self._code_api

    @property
    def actors(self) -> ResearchRunActorsAPI:
        """Runtime actor inventory."""
        if self._actors_api is None:
            self._actors_api = ResearchRunActorsAPI(self)  # type: ignore[arg-type]
        return self._actors_api

    @property
    def evidence(self) -> ResearchRunEvidenceAPI:
        """Operator evidence bundle for debugging."""
        if self._evidence_api is None:
            self._evidence_api = ResearchRunEvidenceAPI(self)  # type: ignore[arg-type]
        return self._evidence_api

    @property
    def authority(self) -> ResearchRunAuthorityAPI:
        """Authority and permission readouts."""
        if self._authority_api is None:
            self._authority_api = ResearchRunAuthorityAPI(self)  # type: ignore[arg-type]
        return self._authority_api

    @property
    def execution(self) -> ResearchRunExecutionAPI:
        """Execution state readouts."""
        if self._execution_api is None:
            self._execution_api = ResearchRunExecutionAPI(self)  # type: ignore[arg-type]
        return self._execution_api

    @property
    def milestones(self) -> ResearchRunMilestonesAPI:
        """Run-scoped milestone readouts."""
        if self._milestones_api is None:
            self._milestones_api = ResearchRunMilestonesAPI(self)  # type: ignore[arg-type]
        return self._milestones_api

    @property
    def ticking(self) -> ResearchRunTickingAPI:
        """Run ticking / heartbeat controls."""
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
