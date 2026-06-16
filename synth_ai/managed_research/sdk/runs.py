"""Run-scoped SDK namespace."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, List, cast

import httpx

from synth_ai.managed_research.errors import SmrApiError
from synth_ai.managed_research.models.canonical_usage import (
    SmrResourceLimitExtension,
    SmrResourceLimitProgress,
    SmrResourceLimits,
    SmrResourceLimitSelector,
    SmrRunUsage,
)
from synth_ai.managed_research.models.checkpoints import Checkpoint
from synth_ai.managed_research.models.run_control import (
    ManagedResearchActorControlAck,
    ManagedResearchActorControlAction,
    ManagedResearchRunControlAck,
)
from synth_ai.managed_research.models.run_diagnostics import (
    SmrRunActorLogs,
    SmrRunActorUsage,
    SmrRunArtifactProgress,
    SmrRunCostSummary,
    SmrRunParticipants,
    SmrRunTraces,
)
from synth_ai.managed_research.models.run_execution import RunExecutionProjection
from synth_ai.managed_research.models.run_launch import (
    EventStream,
    EventStreamRequest,
    RunLaunchRequest,
    RunLaunchResult,
    RunReadRequest,
    RunRef,
    RunSnapshot,
)
from synth_ai.managed_research.models.run_observability import (
    ManagedResearchRunContract,
    RunObservabilitySnapshot,
    RunObservationCursor,
)
from synth_ai.managed_research.models.run_state import ManagedResearchRun
from synth_ai.managed_research.models.run_timeline import (
    SmrAuthorityReadouts,
    SmrBranchMode,
    SmrLogicalTimeline,
    SmrRunBranchResponse,
    SmrRunEventLog,
)
from synth_ai.managed_research.models.runtime_intent import (
    RuntimeIntent,
    RuntimeIntentReceipt,
    RuntimeIntentView,
)
from synth_ai.managed_research.models.smr_host_kinds import SmrHostKind
from synth_ai.managed_research.models.smr_network_topology import SmrNetworkTopology
from synth_ai.managed_research.models.smr_providers import (
    ActorResourceCapability,
    ProviderBinding,
    UsageLimit,
)
from synth_ai.managed_research.models.smr_runbooks import SmrRunbookPreset
from synth_ai.managed_research.models.smr_work_modes import SmrWorkMode
from synth_ai.managed_research.models.types import RunArtifact, RunArtifactManifest
from synth_ai.managed_research.models.work_products import ManagedResearchRunWorkProduct
from synth_ai.managed_research.sdk._base import _ClientNamespace
from synth_ai.managed_research.sdk.config import DEFAULT_MISC_PROJECT_ALIAS

MISC_PROJECT_ID = DEFAULT_MISC_PROJECT_ALIAS


@dataclass(frozen=True, slots=True)
class ProjectSelector:
    """Explicit run launch project selector.

    The backend accepts real project ids plus the default Misc aliases. Project names
    should be resolved by callers before launch so a typo cannot silently create the
    wrong routing decision.
    """

    project_id: str = MISC_PROJECT_ID

    def __post_init__(self) -> None:
        if not str(self.project_id or "").strip():
            raise ValueError("project_id is required")

    @classmethod
    def misc(cls) -> ProjectSelector:
        return cls(MISC_PROJECT_ID)

    @classmethod
    def from_project_id(cls, project_id: str) -> ProjectSelector:
        return cls(str(project_id or "").strip())


def _resolve_project_selector(
    project_id: str | None = None,
    *,
    project: ProjectSelector | str | None = None,
) -> ProjectSelector:
    if project_id is not None and project is not None:
        raise ValueError("pass either project_id or project, not both")
    if isinstance(project, ProjectSelector):
        return project
    if project is not None:
        if not isinstance(project, str):
            raise TypeError("project must be a ProjectSelector or project id string")
        return ProjectSelector.from_project_id(project)
    if project_id is not None:
        return ProjectSelector.from_project_id(project_id)
    return ProjectSelector.misc()


class RunHandle:
    """Project-scoped handle for one managed-research run."""

    def __init__(self, client: Any, project_id: str, run_id: str) -> None:
        project_text = str(project_id or "").strip()
        run_text = str(run_id or "").strip()
        if not project_text:
            raise ValueError("project_id is required")
        if not run_text:
            raise ValueError("run_id is required")
        self._client = client
        self.project_id = project_text
        self.run_id = run_text

    def get(self) -> ManagedResearchRun:
        return ManagedResearchRun.from_wire(
            self._client.get_project_run(self.project_id, self.run_id)
        )

    @property
    def ref(self) -> RunRef:
        return RunRef(project_id=self.project_id, run_id=self.run_id)

    def public_state(self) -> ManagedResearchRun:
        return self._client.get_run_public_state(
            self.run_id,
            project_id=self.project_id,
        )

    def wait(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float = 10.0,
        raise_if_failed: bool = False,
    ) -> ManagedResearchRun:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be greater than 0")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative when provided")
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            try:
                contract = self.contract()
            except httpx.TransportError as exc:
                raise SmrApiError(f"Network error while polling run {self.run_id}: {exc}") from exc
            if contract.terminal:
                if raise_if_failed and contract.public_state.value in {"failed", "blocked"}:
                    msg = self.explain_blocker() or (
                        f"run {self.run_id} ended in state {contract.public_state.value}"
                    )
                    raise SmrApiError(msg, status_code=None)
                return self.get()
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"run {self.run_id} did not complete within {timeout}s")
            time.sleep(poll_interval)

    def contract(self) -> ManagedResearchRunContract:
        return self._client.get_run_contract(self.project_id, self.run_id)

    def wait_terminal(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float = 10.0,
    ) -> ManagedResearchRunContract:
        return self._client.runs.wait_for_run_terminal(
            self.project_id,
            self.run_id,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def explain_blocker(self) -> str | None:
        return self._client.runs.explain_run_blocker(self.project_id, self.run_id)

    @property
    def host_kind(self) -> SmrHostKind | None:
        return self.get().host_kind

    @property
    def resolved_host_kind(self) -> SmrHostKind | None:
        return self.get().resolved_host_kind

    @property
    def work_mode(self) -> SmrWorkMode | None:
        return self.get().work_mode

    @property
    def resolved_work_mode(self) -> SmrWorkMode | None:
        return self.get().resolved_work_mode

    @property
    def runbook(self) -> str | None:
        return self.get().runbook

    @property
    def network_topology(self) -> SmrNetworkTopology | None:
        return self.get().network_topology

    @property
    def network_surfaces(self) -> dict[str, object]:
        return self.get().network_surfaces

    @property
    def providers(self) -> tuple[ProviderBinding, ...]:
        return self.get().providers

    @property
    def capabilities(self) -> frozenset[ActorResourceCapability]:
        return self.get().capabilities

    @property
    def limit(self) -> UsageLimit | None:
        return self.get().limit

    def task_counts(self) -> dict[str, int]:
        return self._client.get_run_observability_snapshot(
            self.project_id,
            self.run_id,
        ).tasks.counts_by_state

    def actor_counts(self) -> dict[str, int]:
        return self._client.get_run_observability_snapshot(
            self.project_id,
            self.run_id,
        ).actors.counts_by_state

    def transcript(
        self,
        *,
        cursor: str | None = None,
        limit: int = 200,
        participant_session_id: str | None = None,
        view: str | None = None,
    ) -> dict[str, Any]:
        return self._client.runs.transcript(
            self.run_id,
            cursor=cursor,
            limit=limit,
            participant_session_id=participant_session_id,
            view=view,
        )

    def stream_events(
        self,
        *,
        transcript_cursor: str | None = None,
        view: str = "operator",
        last_event_id: str | None = None,
        timeout: float | None = None,
    ):
        return self._client.runs.stream_events(
            self.ref,
            transcript_cursor=transcript_cursor,
            view=view,
            last_event_id=last_event_id,
            timeout=timeout,
        )

    def snapshot(self, request: RunReadRequest | None = None) -> RunSnapshot:
        return self._client.runs.get_snapshot(self.ref, request=request)

    def messages(
        self,
        *,
        status: str | None = None,
        viewer_role: str | None = None,
        viewer_target: str | List[str] | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        return self._client.list_project_run_runtime_messages(
            self.project_id,
            self.run_id,
            status=status,
            viewer_role=viewer_role,
            viewer_target=viewer_target,
            limit=limit,
        )

    def publish_message(
        self,
        *,
        intent: str = "queue",
        audience: dict[str, Any] | None = None,
        body: str | None = None,
        payload: dict[str, Any] | None = None,
        message_kind: str = "runtime_message",
        thread_id: str | None = None,
        parent_message_id: str | None = None,
        fallback_policy: str = "block",
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
    ) -> dict[str, Any]:
        return self._client.publish_manderqueue_message(
            self.run_id,
            project_id=self.project_id,
            intent=intent,
            audience=audience,
            body=body,
            payload=payload,
            message_kind=message_kind,
            thread_id=thread_id,
            parent_message_id=parent_message_id,
            fallback_policy=fallback_policy,
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
            causation_id=causation_id,
        )

    def queue_message(self, *, body: str, **kwargs: Any) -> dict[str, Any]:
        return self.publish_message(intent="queue", body=body, **kwargs)

    def send_message(self, *, body: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.send_message(
            self.run_id,
            project_id=self.project_id,
            body=body,
            **kwargs,
        )

    def steer_message(self, *, body: str, **kwargs: Any) -> dict[str, Any]:
        return self.publish_message(intent="steer", body=body, **kwargs)

    def interrupt_message(self, *, body: str, **kwargs: Any) -> dict[str, Any]:
        return self.publish_message(intent="interrupt", body=body, **kwargs)

    def note_message(self, *, body: str, **kwargs: Any) -> dict[str, Any]:
        return self.publish_message(intent="note", body=body, **kwargs)

    def manderqueue_threads(self, *, limit: int | None = None) -> List[dict[str, Any]]:
        return self._client.list_manderqueue_threads(
            self.run_id, project_id=self.project_id, limit=limit
        )

    def manderqueue_messages(
        self,
        *,
        thread_id: str | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        return self._client.list_manderqueue_messages(
            self.run_id,
            project_id=self.project_id,
            thread_id=thread_id,
            limit=limit,
        )

    def manderqueue_interactions(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        return self._client.list_manderqueue_interactions(
            self.run_id,
            project_id=self.project_id,
            status=status,
            limit=limit,
        )

    def respond_to_manderqueue_interaction(
        self,
        interaction_id: str,
        *,
        body: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._client.respond_to_manderqueue_interaction(
            self.run_id,
            interaction_id,
            project_id=self.project_id,
            body=body,
            payload=payload,
        )

    def edit_manderqueue_message(
        self,
        message_id: str,
        *,
        body: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._client.edit_manderqueue_message(
            self.run_id,
            message_id,
            project_id=self.project_id,
            body=body,
            payload=payload,
        )

    def edit_message(
        self,
        message_id: str,
        *,
        body: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._client.edit_message(
            self.run_id,
            message_id,
            project_id=self.project_id,
            body=body,
            payload=payload,
        )

    def retract_manderqueue_message(self, message_id: str) -> dict[str, Any]:
        return self._client.retract_manderqueue_message(
            self.run_id,
            message_id,
            project_id=self.project_id,
        )

    def retract_message(self, message_id: str) -> dict[str, Any]:
        return self._client.retract_message(
            self.run_id,
            message_id,
            project_id=self.project_id,
        )

    def submit_intent(
        self,
        intent: RuntimeIntent | dict[str, Any],
        *,
        mode: str = "queue",
        body: str | None = None,
        causation_id: str | None = None,
    ) -> RuntimeIntentReceipt:
        return self._client.submit_runtime_intent(
            self.run_id,
            intent,
            project_id=self.project_id,
            mode=mode,
            body=body,
            causation_id=causation_id,
        )

    def intents(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> List[RuntimeIntentView]:
        return self._client.list_runtime_intents(
            self.run_id,
            project_id=self.project_id,
            status=status,
            limit=limit,
        )

    def intent(self, runtime_intent_id: str) -> RuntimeIntentView:
        return self._client.get_runtime_intent(
            self.run_id,
            runtime_intent_id,
            project_id=self.project_id,
        )

    def timeline(self) -> SmrLogicalTimeline:
        return self._client.get_run_logical_timeline(self.project_id, self.run_id)

    def execution(
        self,
        *,
        view: str = "summary",
        event_limit: int = 100,
        actor_limit: int = 50,
        task_limit: int = 100,
        message_limit: int = 50,
        work_product_limit: int = 50,
    ) -> RunExecutionProjection:
        return self._client.get_run_execution(
            self.project_id,
            self.run_id,
            view=view,
            event_limit=event_limit,
            actor_limit=actor_limit,
            task_limit=task_limit,
            message_limit=message_limit,
            work_product_limit=work_product_limit,
        )

    def task_events(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._client.list_run_task_events(
            self.project_id,
            self.run_id,
            limit=limit,
            cursor=cursor,
        )

    def objective_events(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._client.list_run_objective_events(
            self.project_id,
            self.run_id,
            limit=limit,
            cursor=cursor,
        )

    def work_graph(self, *, limit: int | None = None) -> dict[str, Any]:
        return self._client.get_run_work_graph(
            self.project_id,
            self.run_id,
            limit=limit,
        )

    def event_log(
        self,
        *,
        sources: List[str] | None = None,
        event_kinds: List[str] | None = None,
        statuses: List[str] | None = None,
        limit: int | None = None,
    ) -> SmrRunEventLog:
        return self._client.get_project_run_event_log(
            self.project_id,
            self.run_id,
            sources=sources,
            event_kinds=event_kinds,
            statuses=statuses,
            limit=limit,
        )

    def authority_readouts(
        self,
        *,
        include_runtime_authority: bool = False,
    ) -> SmrAuthorityReadouts:
        return self._client.get_project_run_authority_readouts(
            self.project_id,
            self.run_id,
            include_runtime_authority=include_runtime_authority,
        )

    def operator_evidence(
        self,
        *,
        runtime_timeline_limit: int | None = None,
        logical_timeline_limit: int | None = None,
        transcript_limit: int | None = None,
        reconciliation_limit: int | None = None,
    ) -> dict[str, Any]:
        return self._client.get_project_run_operator_evidence(
            self.project_id,
            self.run_id,
            runtime_timeline_limit=runtime_timeline_limit,
            logical_timeline_limit=logical_timeline_limit,
            transcript_limit=transcript_limit,
            reconciliation_limit=reconciliation_limit,
        )

    def traces(self) -> SmrRunTraces:
        return self._client.get_project_run_traces(self.project_id, self.run_id)

    def actor_inventory(self) -> dict[str, Any]:
        return self._client.get_project_run_actor_trace_index(self.project_id, self.run_id)

    def actor_trace(self, actor_key: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.get_project_run_actor_trace(
            self.project_id,
            self.run_id,
            actor_key,
            **kwargs,
        )

    def actor_raw_traces(self, actor_key: str) -> List[dict[str, Any]]:
        return self._client.get_project_run_actor_raw_traces(
            self.project_id,
            self.run_id,
            actor_key,
        )

    def raw_trace_events(self, artifact_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.get_project_run_raw_trace_events(
            self.project_id,
            self.run_id,
            artifact_id,
            **kwargs,
        )

    def download_raw_trace(
        self,
        artifact_id: str,
        destination: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._client.download_project_run_raw_trace(
            self.project_id,
            self.run_id,
            artifact_id,
            destination,
            **kwargs,
        )

    def actor_usage(self) -> SmrRunActorUsage:
        return self._client.get_project_run_actor_usage(self.project_id, self.run_id)

    def resource_limits(self) -> SmrResourceLimits:
        return self._client.get_project_run_resource_limits(self.project_id, self.run_id)

    def progress_toward_resource_limits(self) -> SmrResourceLimitProgress:
        return self._client.get_project_run_progress_toward_resource_limits(
            self.project_id,
            self.run_id,
        )

    def extend_resource_limit(
        self,
        *,
        limit_value: float | None = None,
        additional_value: float | None = None,
        reason: str | None = None,
        selector: SmrResourceLimitSelector | Mapping[str, object] | None = None,
        resource_limit_id: str | None = None,
        metric: str = "spend_usd",
        unit: str = "usd",
        resolve_blockers: bool = True,
        resume: bool = True,
        idempotency_key: str | None = None,
    ) -> SmrResourceLimitExtension:
        return self._client.extend_project_run_resource_limit(
            self.project_id,
            self.run_id,
            limit_value=limit_value,
            additional_value=additional_value,
            reason=reason,
            selector=selector,
            resource_limit_id=resource_limit_id,
            metric=metric,
            unit=unit,
            resolve_blockers=resolve_blockers,
            resume=resume,
            idempotency_key=idempotency_key,
        )

    def checkpoints(self) -> List[Checkpoint]:
        return self._client.list_run_checkpoints(
            self.run_id,
            project_id=self.project_id,
        )

    def checkpoint(self, checkpoint_id: str) -> Checkpoint:
        return self._client.get_run_checkpoint(
            self.run_id,
            checkpoint_id,
            project_id=self.project_id,
        )

    def create_checkpoint(
        self,
        *,
        checkpoint_id: str | None = None,
        reason: str | None = None,
        timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 1.0,
    ) -> Checkpoint:
        return self._client.create_run_checkpoint(
            self.run_id,
            project_id=self.project_id,
            checkpoint_id=checkpoint_id,
            reason=reason,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    def request_checkpoint(
        self,
        *,
        checkpoint_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return self._client.request_run_checkpoint(
            self.run_id,
            project_id=self.project_id,
            checkpoint_id=checkpoint_id,
            reason=reason,
        )

    def restore_checkpoint(
        self,
        *,
        checkpoint_id: str | None = None,
        checkpoint_record_id: str | None = None,
        checkpoint_uri: str | None = None,
        reason: str | None = None,
        mode: str = "in_place",
    ) -> dict[str, Any]:
        return self._client.restore_run_checkpoint(
            self.run_id,
            project_id=self.project_id,
            checkpoint_id=checkpoint_id,
            checkpoint_record_id=checkpoint_record_id,
            checkpoint_uri=checkpoint_uri,
            reason=reason,
            mode=mode,
        )

    def branch_from_checkpoint(
        self,
        *,
        checkpoint_id: str | None = None,
        checkpoint_record_id: str | None = None,
        checkpoint_uri: str | None = None,
        mode: SmrBranchMode | str = SmrBranchMode.EXACT,
        message: str | None = None,
        reason: str | None = None,
        title: str | None = None,
        source_node_id: str | None = None,
    ) -> SmrRunBranchResponse:
        return self._client.branch_run_from_checkpoint(
            self.run_id,
            project_id=self.project_id,
            checkpoint_id=checkpoint_id,
            checkpoint_record_id=checkpoint_record_id,
            checkpoint_uri=checkpoint_uri,
            mode=mode,
            message=message,
            reason=reason,
            title=title,
            source_node_id=source_node_id,
        )

    def artifact_manifest(self) -> RunArtifactManifest:
        return self._client.get_run_artifact_manifest(
            self.run_id,
            project_id=self.project_id,
        )

    def artifacts(
        self,
        *,
        artifact_type: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> List[RunArtifact]:
        return self._client.list_run_artifacts(
            self.run_id,
            project_id=self.project_id,
            artifact_type=artifact_type,
            limit=limit,
            cursor=cursor,
        )

    def work_products(self) -> list[ManagedResearchRunWorkProduct]:
        return self._client.work_products.list_for_run(self.project_id, self.run_id)

    def reports(self) -> list[ManagedResearchRunWorkProduct]:
        return [item for item in self.work_products() if item.kind == "report"]

    def final_report(self) -> ManagedResearchRunWorkProduct | None:
        reports = self.reports()
        ready_reports = [
            item
            for item in reports
            if item.status in {"ready", "published", "done"}
            or item.readiness in {"viewable", "downloadable", "importable"}
        ]
        if ready_reports:
            return ready_reports[0]
        return reports[0] if reports else None

    def report_text(self, work_product_id: str | None = None) -> str | None:
        if work_product_id is None:
            report = self.final_report()
            if report is None:
                return None
            work_product_id = report.work_product_id
        content = self._client.work_products.content(work_product_id, as_text=True)
        if isinstance(content, bytes):
            return content.decode("utf-8")
        return content

    def output_file(self, name: str) -> RunArtifact | None:
        wanted = str(name or "").strip().lower()
        if not wanted:
            raise ValueError("name is required")
        for artifact in self.artifact_manifest().output_files:
            candidates = {
                artifact.artifact_id,
                artifact.artifact_type,
                artifact.title,
                artifact.path,
            }
            if artifact.path:
                candidates.add(artifact.path.rsplit("/", 1)[-1])
            if any(str(candidate or "").strip().lower() == wanted for candidate in candidates):
                return artifact
        return None

    def download(self, path: str) -> dict[str, Any]:
        return self._client.download_run_workspace_archive(
            self.project_id,
            self.run_id,
            path,
        )

    def models(self) -> List[dict[str, Any]]:
        return self._client.list_run_models(self.run_id, project_id=self.project_id)

    def datasets(self) -> List[dict[str, Any]]:
        return self._client.list_run_datasets(self.run_id, project_id=self.project_id)

    def participants(self) -> SmrRunParticipants:
        return self._client.list_run_participants(
            self.run_id,
            project_id=self.project_id,
        )

    def artifact_progress(self) -> SmrRunArtifactProgress:
        return self._client.get_run_artifact_progress(
            self.run_id,
            project_id=self.project_id,
        )

    def actor_logs(self, **kwargs: Any) -> SmrRunActorLogs:
        return self._client.list_run_actor_logs(
            self.run_id,
            project_id=self.project_id,
            **kwargs,
        )

    def control_actor(
        self,
        actor_id: str,
        *,
        action: str,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> ManagedResearchActorControlAck:
        return ManagedResearchActorControlAck.from_wire(
            self._client.control_project_run_actor(
                self.project_id,
                self.run_id,
                actor_id,
                action=action,
                reason=reason,
                idempotency_key=idempotency_key,
            )
        )

    def pause_actor(
        self,
        actor_id: str,
        *,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> ManagedResearchActorControlAck:
        return self.control_actor(
            actor_id,
            action=ManagedResearchActorControlAction.PAUSE.value,
            reason=reason,
            idempotency_key=idempotency_key,
        )

    def resume_actor(
        self,
        actor_id: str,
        *,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> ManagedResearchActorControlAck:
        return self.control_actor(
            actor_id,
            action=ManagedResearchActorControlAction.RESUME.value,
            reason=reason,
            idempotency_key=idempotency_key,
        )

    def interrupt_actor(
        self,
        actor_id: str,
        *,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> ManagedResearchActorControlAck:
        return self.control_actor(
            actor_id,
            action=ManagedResearchActorControlAction.INTERRUPT.value,
            reason=reason,
            idempotency_key=idempotency_key,
        )

    def cost_summary(self) -> SmrRunCostSummary:
        return self._client.get_run_cost_summary(self.run_id)

    def stop(self) -> ManagedResearchRunControlAck:
        return ManagedResearchRunControlAck.from_wire(
            self._client.stop_run(self.run_id, project_id=self.project_id)
        )

    def pause(self) -> ManagedResearchRunControlAck:
        return ManagedResearchRunControlAck.from_wire(
            self._client.pause_run(self.run_id, project_id=self.project_id)
        )

    def resume(self) -> ManagedResearchRunControlAck:
        return ManagedResearchRunControlAck.from_wire(
            self._client.resume_run(self.run_id, project_id=self.project_id)
        )


class RunsAPI(_ClientNamespace):
    def _handle(self, project_id: str, run_id: str) -> RunHandle:
        return RunHandle(self._client, project_id, run_id)

    def launch_preflight(
        self,
        project_id: str | None = None,
        *,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if project_id is None and project is None:
            return self._client.get_one_off_launch_preflight(**kwargs)
        selector = _resolve_project_selector(project_id, project=project)
        return self._client.get_launch_preflight(selector.project_id, **kwargs)

    def runbook_presets(self) -> tuple[SmrRunbookPreset, ...]:
        return self._client.list_runbook_presets()

    def trigger(
        self,
        project_id: str | None = None,
        *,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if project_id is None and project is None:
            return self._client.trigger_one_off_run(**kwargs)
        selector = _resolve_project_selector(project_id, project=project)
        return self._client.trigger_run(selector.project_id, **kwargs)

    def trigger_result(
        self,
        project_id: str | None = None,
        *,
        project: ProjectSelector | str | None = None,
        request: RunLaunchRequest,
    ) -> RunLaunchResult:
        selector = _resolve_project_selector(project_id, project=project)
        return self._client.trigger_run_result(selector.project_id, request=request)

    def start_run(
        self,
        project_id: str | None = None,
        *,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if project_id is None and project is None:
            return self._client.trigger_one_off_run(**kwargs)
        selector = _resolve_project_selector(project_id, project=project)
        return self._client.start_run(selector.project_id, **kwargs)

    def start(
        self,
        objective: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> RunHandle:
        objective_text = str(objective or "").strip()
        if not objective_text:
            raise ValueError("objective is required")
        payload = dict(kwargs)
        payload["objective"] = objective_text
        if project_id is None and project is None:
            wire = self._client.trigger_one_off_run(**payload)
        else:
            selector = _resolve_project_selector(project_id, project=project)
            wire = self._client.trigger_run(selector.project_id, **payload)
        run = ManagedResearchRun.from_wire(wire)
        return RunHandle(self._client, run.project_id, run.run_id)

    def launch(
        self,
        objective: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> RunHandle:
        return self.start(objective, project_id=project_id, project=project, **kwargs)

    def list(
        self, project_id: str, *, active_only: bool = False, **kwargs: Any
    ) -> List[dict[str, Any]]:
        return self._client.list_runs(project_id, active_only=active_only, **kwargs)

    def list_active(self, project_id: str) -> List[dict[str, Any]]:
        return self._client.list_active_runs(project_id)

    def get(self, run_id: str, *, project_id: str | None = None) -> ManagedResearchRun:
        return ManagedResearchRun.from_wire(self._client.get_run(run_id, project_id=project_id))

    def public_state(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> ManagedResearchRun:
        return self._client.get_run_public_state(run_id, project_id=project_id)

    def wait(
        self,
        project_id: str,
        run_id: str,
        *,
        timeout: float | None = None,
        poll_interval: float = 10.0,
    ) -> ManagedResearchRun:
        return self._handle(project_id, run_id).wait(
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def get_usage(self, run_id: str) -> SmrRunUsage:
        return self._client.get_run_usage(run_id)

    def get_resource_limits(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> SmrResourceLimits:
        if project_id:
            return self._client.get_project_run_resource_limits(project_id, run_id)
        return self._client.get_run_resource_limits(run_id)

    def get_progress_toward_resource_limits(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> SmrResourceLimitProgress:
        if project_id:
            return self._client.get_project_run_progress_toward_resource_limits(
                project_id,
                run_id,
            )
        return self._client.get_run_progress_toward_resource_limits(run_id)

    def extend_resource_limit(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        limit_value: float | None = None,
        additional_value: float | None = None,
        reason: str | None = None,
        selector: SmrResourceLimitSelector | Mapping[str, object] | None = None,
        resource_limit_id: str | None = None,
        metric: str = "spend_usd",
        unit: str = "usd",
        resolve_blockers: bool = True,
        resume: bool = True,
        idempotency_key: str | None = None,
    ) -> SmrResourceLimitExtension:
        if project_id:
            return self._client.extend_project_run_resource_limit(
                project_id,
                run_id,
                limit_value=limit_value,
                additional_value=additional_value,
                reason=reason,
                selector=selector,
                resource_limit_id=resource_limit_id,
                metric=metric,
                unit=unit,
                resolve_blockers=resolve_blockers,
                resume=resume,
                idempotency_key=idempotency_key,
            )
        return self._client.extend_run_resource_limit(
            run_id,
            limit_value=limit_value,
            additional_value=additional_value,
            reason=reason,
            selector=selector,
            resource_limit_id=resource_limit_id,
            metric=metric,
            unit=unit,
            resolve_blockers=resolve_blockers,
            resume=resume,
            idempotency_key=idempotency_key,
        )

    def get_observability_snapshot(
        self,
        project_id: str,
        run_id: str,
        *,
        detail_level: str = "full",
        event_limit: int = 100,
        actor_limit: int = 25,
        task_limit: int = 50,
        question_limit: int = 25,
        timeline_limit: int = 10,
        message_limit: int = 10,
    ) -> RunObservabilitySnapshot:
        return self._client.get_run_observability_snapshot(
            project_id,
            run_id,
            detail_level=detail_level,
            event_limit=event_limit,
            actor_limit=actor_limit,
            task_limit=task_limit,
            question_limit=question_limit,
            timeline_limit=timeline_limit,
            message_limit=message_limit,
        )

    def snapshot(self, request: RunReadRequest) -> RunSnapshot:
        snapshot = self.get_observability_snapshot(
            request.run_ref.project_id,
            request.run_ref.run_id,
            event_limit=request.event_limit or 100,
            actor_limit=request.actor_limit or 25,
            task_limit=request.task_limit or 50,
            question_limit=request.question_limit or 25,
            timeline_limit=request.timeline_limit or 10,
            message_limit=request.message_limit or 10,
        )
        return RunSnapshot.from_observability(
            run_ref=request.run_ref,
            snapshot=snapshot,
        )

    def get_snapshot(
        self,
        run_ref: RunRef,
        *,
        request: RunReadRequest | None = None,
    ) -> RunSnapshot:
        effective_request = request or RunReadRequest(run_ref=run_ref)
        if effective_request.run_ref != run_ref:
            raise ValueError("request.run_ref must match run_ref")
        return self.snapshot(effective_request)

    def get_observability_snapshot_control(
        self,
        project_id: str,
        run_id: str,
        *,
        event_limit: int = 40,
        actor_limit: int = 25,
        task_limit: int = 40,
        question_limit: int = 10,
        timeline_limit: int = 10,
        message_limit: int = 8,
    ) -> RunObservabilitySnapshot:
        return self._client.get_run_observability_snapshot_control(
            project_id,
            run_id,
            event_limit=event_limit,
            actor_limit=actor_limit,
            task_limit=task_limit,
            question_limit=question_limit,
            timeline_limit=timeline_limit,
            message_limit=message_limit,
        )

    def get_observability_snapshot_full(
        self,
        project_id: str,
        run_id: str,
        *,
        event_limit: int = 100,
        actor_limit: int = 25,
        task_limit: int = 50,
        question_limit: int = 25,
        timeline_limit: int = 10,
        message_limit: int = 10,
    ) -> RunObservabilitySnapshot:
        return self._client.get_run_observability_snapshot_full(
            project_id,
            run_id,
            event_limit=event_limit,
            actor_limit=actor_limit,
            task_limit=task_limit,
            question_limit=question_limit,
            timeline_limit=timeline_limit,
            message_limit=message_limit,
        )

    def get_execution(
        self,
        project_id: str,
        run_id: str,
        *,
        view: str = "summary",
        event_limit: int = 100,
        actor_limit: int = 50,
        task_limit: int = 100,
        message_limit: int = 50,
        work_product_limit: int = 50,
    ) -> RunExecutionProjection:
        return self._client.get_run_execution(
            project_id,
            run_id,
            view=view,
            event_limit=event_limit,
            actor_limit=actor_limit,
            task_limit=task_limit,
            message_limit=message_limit,
            work_product_limit=work_product_limit,
        )

    def get_actors(
        self,
        project_id: str,
        run_id: str,
    ) -> List[dict[str, Any]]:
        return self._client.get_project_run_actors(project_id, run_id)

    def list_task_events(
        self,
        project_id: str,
        run_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._client.list_run_task_events(
            project_id,
            run_id,
            limit=limit,
            cursor=cursor,
        )

    def list_objective_events(
        self,
        project_id: str,
        run_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._client.list_run_objective_events(
            project_id,
            run_id,
            limit=limit,
            cursor=cursor,
        )

    def get_work_graph(
        self,
        project_id: str,
        run_id: str,
        *,
        limit: int | None = None,
    ) -> dict[str, Any]:
        return self._client.get_run_work_graph(
            project_id,
            run_id,
            limit=limit,
        )

    def get_run_contract(
        self,
        project_id: str,
        run_id: str,
    ) -> ManagedResearchRunContract:
        return self._client.get_run_contract(project_id, run_id)

    def wait_for_run_terminal(
        self,
        project_id: str,
        run_id: str,
        *,
        timeout: float | None = None,
        poll_interval: float = 10.0,
    ) -> ManagedResearchRunContract:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be greater than 0")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative when provided")
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            contract = self.get_run_contract(project_id, run_id)
            if contract.terminal:
                return contract
            if deadline is not None and time.monotonic() >= deadline:
                blocker = self.explain_run_blocker(project_id, run_id)
                suffix = f": {blocker}" if blocker else ""
                raise TimeoutError(f"run {run_id} did not complete within {timeout}s{suffix}")
            time.sleep(poll_interval)

    def explain_run_blocker(self, project_id: str, run_id: str) -> str | None:
        contract = self.get_run_contract(project_id, run_id)
        invariant = next(
            iter(contract.diagnostics.lifecycle_invariants),
            None,
        )
        if invariant is not None:
            detail = str(invariant.get("detail") or "").strip()
            code = str(invariant.get("code") or "").strip()
            return detail or code or "run lifecycle invariant failed"
        failure = contract.diagnostics.failure_classification
        if failure is not None:
            code = str(failure.get("code") or "").strip()
            detail = str(failure.get("detail") or "").strip()
            route = failure.get("route")
            route_mapping = cast(Mapping[str, Any], route) if isinstance(route, Mapping) else {}
            model = str(route_mapping.get("model") or "").strip()
            suffix = f" model={model}" if model else ""
            if detail:
                return detail
            if code:
                return f"{code}{suffix}"
            return f"run failure{suffix}"
        if contract.incidents.unresolved:
            return (
                f"{contract.incidents.unresolved} unresolved incident(s); "
                f"recovery={contract.recovery.status}"
            )
        if contract.recovery.status not in {"none", "closed"}:
            reason = f": {contract.recovery.reason}" if contract.recovery.reason else ""
            return f"recovery {contract.recovery.status}{reason}"
        if contract.finalization.status in {"required", "in_review", "blocked"}:
            reason = f": {contract.finalization.reason}" if contract.finalization.reason else ""
            return f"finalization {contract.finalization.status}{reason}"
        if contract.tasks.nonterminal:
            return f"{contract.tasks.nonterminal} nonterminal task(s)"
        if contract.artifacts.readiness == "not_ready":
            return "terminal artifacts are not ready"
        return None

    def poll_observability_snapshot(
        self,
        project_id: str,
        run_id: str,
        *,
        cursor: RunObservationCursor | dict[str, Any] | None = None,
        detail_level: str = "full",
        event_limit: int = 100,
        actor_limit: int = 25,
        task_limit: int = 50,
        question_limit: int = 25,
        timeline_limit: int = 10,
        message_limit: int = 10,
    ) -> RunObservabilitySnapshot:
        return self._client.poll_run_observability_snapshot(
            project_id,
            run_id,
            cursor=cursor,
            detail_level=detail_level,
            event_limit=event_limit,
            actor_limit=actor_limit,
            task_limit=task_limit,
            question_limit=question_limit,
            timeline_limit=timeline_limit,
            message_limit=message_limit,
        )

    def get_actor_counts(self, project_id: str, run_id: str) -> dict[str, int]:
        return self.get_observability_snapshot(project_id, run_id).actors.counts_by_state

    def get_task_counts(self, project_id: str, run_id: str) -> dict[str, int]:
        return self.get_observability_snapshot(project_id, run_id).tasks.counts_by_state

    def get_terminal_classifier(self, project_id: str, run_id: str) -> str:
        snapshot = self.get_observability_snapshot(project_id, run_id)
        return snapshot.public_state.value

    def stop(self, run_id: str, *, project_id: str | None = None) -> ManagedResearchRunControlAck:
        return ManagedResearchRunControlAck.from_wire(
            self._client.stop_run(run_id, project_id=project_id)
        )

    def pause(self, run_id: str, *, project_id: str | None = None) -> ManagedResearchRunControlAck:
        return ManagedResearchRunControlAck.from_wire(
            self._client.pause_run(run_id, project_id=project_id)
        )

    def resume(self, run_id: str, *, project_id: str | None = None) -> ManagedResearchRunControlAck:
        return ManagedResearchRunControlAck.from_wire(
            self._client.resume_run(run_id, project_id=project_id)
        )

    def control_actor(
        self,
        project_id: str,
        run_id: str,
        actor_id: str,
        *,
        action: str,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> ManagedResearchActorControlAck:
        return ManagedResearchActorControlAck.from_wire(
            self._client.control_project_run_actor(
                project_id,
                run_id,
                actor_id,
                action=action,
                reason=reason,
                idempotency_key=idempotency_key,
            )
        )

    def pause_actor(
        self,
        project_id: str,
        run_id: str,
        actor_id: str,
        *,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> ManagedResearchActorControlAck:
        return self.control_actor(
            project_id,
            run_id,
            actor_id,
            action=ManagedResearchActorControlAction.PAUSE.value,
            reason=reason,
            idempotency_key=idempotency_key,
        )

    def resume_actor(
        self,
        project_id: str,
        run_id: str,
        actor_id: str,
        *,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> ManagedResearchActorControlAck:
        return self.control_actor(
            project_id,
            run_id,
            actor_id,
            action=ManagedResearchActorControlAction.RESUME.value,
            reason=reason,
            idempotency_key=idempotency_key,
        )

    def interrupt_actor(
        self,
        project_id: str,
        run_id: str,
        actor_id: str,
        *,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> ManagedResearchActorControlAck:
        return self.control_actor(
            project_id,
            run_id,
            actor_id,
            action=ManagedResearchActorControlAction.INTERRUPT.value,
            reason=reason,
            idempotency_key=idempotency_key,
        )

    def submit_intent(
        self,
        run_id: str,
        intent: RuntimeIntent | dict[str, Any],
        *,
        project_id: str | None = None,
        mode: str = "queue",
        body: str | None = None,
        causation_id: str | None = None,
    ) -> RuntimeIntentReceipt:
        return self._client.submit_runtime_intent(
            run_id,
            intent,
            project_id=project_id,
            mode=mode,
            body=body,
            causation_id=causation_id,
        )

    def intents(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> List[RuntimeIntentView]:
        return self._client.list_runtime_intents(
            run_id,
            project_id=project_id,
            status=status,
            limit=limit,
        )

    def intent(
        self,
        run_id: str,
        runtime_intent_id: str,
        *,
        project_id: str | None = None,
    ) -> RuntimeIntentView:
        return self._client.get_runtime_intent(
            run_id,
            runtime_intent_id,
            project_id=project_id,
        )

    def list_questions(
        self, run_id: str, *, project_id: str | None = None, **kwargs: Any
    ) -> List[dict[str, Any]]:
        return self._client.list_run_questions(run_id, project_id=project_id, **kwargs)

    def respond_to_question(
        self,
        run_id: str,
        question_id: str,
        *,
        response_text: str,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        return self._client.respond_to_run_question(
            run_id,
            question_id,
            project_id=project_id,
            response_text=response_text,
        )

    def create_checkpoint(
        self, run_id: str, *, project_id: str | None = None, **kwargs: Any
    ) -> Checkpoint:
        return self._client.create_run_checkpoint(run_id, project_id=project_id, **kwargs)

    def list_checkpoints(self, run_id: str, *, project_id: str | None = None) -> List[Checkpoint]:
        return self._client.list_run_checkpoints(run_id, project_id=project_id)

    def checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
        *,
        project_id: str | None = None,
    ) -> Checkpoint | None:
        return self._client.get_run_checkpoint(
            run_id,
            checkpoint_id,
            project_id=project_id,
        )

    def request_checkpoint(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._client.request_run_checkpoint(run_id, project_id=project_id, **kwargs)

    def artifact_manifest(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> RunArtifactManifest:
        return self._client.get_run_artifact_manifest(run_id, project_id=project_id)

    def artifacts(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        artifact_type: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> List[RunArtifact]:
        return self._client.list_run_artifacts(
            run_id,
            project_id=project_id,
            artifact_type=artifact_type,
            limit=limit,
            cursor=cursor,
        )

    def output_file(
        self,
        run_id: str,
        name: str,
        *,
        project_id: str | None = None,
    ) -> RunArtifact | None:
        wanted = str(name or "").strip().lower()
        if not wanted:
            raise ValueError("name is required")
        for artifact in self.artifact_manifest(
            run_id,
            project_id=project_id,
        ).output_files:
            candidates = {
                artifact.artifact_id,
                artifact.artifact_type,
                artifact.title,
                artifact.path,
            }
            if artifact.path:
                candidates.add(artifact.path.rsplit("/", 1)[-1])
            if any(str(candidate or "").strip().lower() == wanted for candidate in candidates):
                return artifact
        return None

    def download(
        self,
        project_id: str,
        run_id: str,
        path: str,
    ) -> dict[str, Any]:
        return self._client.download_run_workspace_archive(project_id, run_id, path)

    def models(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> List[dict[str, Any]]:
        return self._client.list_run_models(run_id, project_id=project_id)

    def datasets(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> List[dict[str, Any]]:
        return self._client.list_run_datasets(run_id, project_id=project_id)

    def restore_checkpoint(
        self, run_id: str, *, project_id: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return self._client.restore_run_checkpoint(run_id, project_id=project_id, **kwargs)

    def get_logical_timeline(self, project_id: str, run_id: str) -> SmrLogicalTimeline:
        return self._client.get_run_logical_timeline(project_id, run_id)

    def get_event_log(
        self,
        project_id: str,
        run_id: str,
        *,
        sources: List[str] | None = None,
        event_kinds: List[str] | None = None,
        statuses: List[str] | None = None,
        limit: int | None = None,
    ) -> SmrRunEventLog:
        return self._client.get_project_run_event_log(
            project_id,
            run_id,
            sources=sources,
            event_kinds=event_kinds,
            statuses=statuses,
            limit=limit,
        )

    def get_authority_readouts(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        include_runtime_authority: bool = False,
    ) -> SmrAuthorityReadouts:
        if project_id:
            return self._client.get_project_run_authority_readouts(
                project_id,
                run_id,
                include_runtime_authority=include_runtime_authority,
            )
        return self._client.get_run_authority_readouts(
            run_id,
            include_runtime_authority=include_runtime_authority,
        )

    def get_operator_evidence(
        self,
        project_id: str,
        run_id: str,
        *,
        runtime_timeline_limit: int | None = None,
        logical_timeline_limit: int | None = None,
        transcript_limit: int | None = None,
        reconciliation_limit: int | None = None,
    ) -> dict[str, Any]:
        return self._client.get_project_run_operator_evidence(
            project_id,
            run_id,
            runtime_timeline_limit=runtime_timeline_limit,
            logical_timeline_limit=logical_timeline_limit,
            transcript_limit=transcript_limit,
            reconciliation_limit=reconciliation_limit,
        )

    def get_traces(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> SmrRunTraces:
        if project_id:
            return self._client.get_project_run_traces(project_id, run_id)
        return self._client.get_run_traces(run_id)

    def get_actor_usage(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> SmrRunActorUsage:
        if project_id:
            return self._client.get_project_run_actor_usage(project_id, run_id)
        return self._client.get_run_actor_usage(run_id)

    def participants(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> SmrRunParticipants:
        return self._client.list_run_participants(run_id, project_id=project_id)

    def artifact_progress(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> SmrRunArtifactProgress:
        return self._client.get_run_artifact_progress(run_id, project_id=project_id)

    def actor_logs(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        **kwargs: Any,
    ) -> SmrRunActorLogs:
        return self._client.list_run_actor_logs(
            run_id,
            project_id=project_id,
            **kwargs,
        )

    def actor_trace(
        self,
        project_id: str,
        run_id: str,
        actor_key: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._client.get_project_run_actor_trace(
            project_id,
            run_id,
            actor_key,
            **kwargs,
        )

    def actor_inventory(self, project_id: str, run_id: str) -> dict[str, Any]:
        return self._client.get_project_run_actor_trace_index(project_id, run_id)

    def actor_raw_traces(
        self,
        project_id: str,
        run_id: str,
        actor_key: str,
    ) -> List[dict[str, Any]]:
        return self._client.get_project_run_actor_raw_traces(
            project_id,
            run_id,
            actor_key,
        )

    def raw_trace_events(
        self,
        project_id: str,
        run_id: str,
        artifact_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._client.get_project_run_raw_trace_events(
            project_id,
            run_id,
            artifact_id,
            **kwargs,
        )

    def raw_trace_download_url(
        self,
        project_id: str,
        run_id: str,
        artifact_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._client.create_project_run_raw_trace_download_url(
            project_id,
            run_id,
            artifact_id,
            **kwargs,
        )

    def download_raw_trace(
        self,
        project_id: str,
        run_id: str,
        artifact_id: str,
        destination: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._client.download_project_run_raw_trace(
            project_id,
            run_id,
            artifact_id,
            destination,
            **kwargs,
        )

    def cost_summary(self, run_id: str) -> SmrRunCostSummary:
        return self._client.get_run_cost_summary(run_id)

    def branch_from_checkpoint(
        self,
        run_id: str | None = None,
        *,
        project_id: str | None = None,
        checkpoint_id: str | None = None,
        checkpoint_record_id: str | None = None,
        checkpoint_uri: str | None = None,
        mode: SmrBranchMode | str = SmrBranchMode.EXACT,
        message: str | None = None,
        reason: str | None = None,
        title: str | None = None,
        source_node_id: str | None = None,
    ) -> SmrRunBranchResponse:
        return self._client.branch_run_from_checkpoint(
            run_id,
            project_id=project_id,
            checkpoint_id=checkpoint_id,
            checkpoint_record_id=checkpoint_record_id,
            checkpoint_uri=checkpoint_uri,
            mode=mode,
            message=message,
            reason=reason,
            title=title,
            source_node_id=source_node_id,
        )

    def list_runtime_messages(
        self,
        run_id: str,
        *,
        status: str | None = None,
        viewer_role: str | None = None,
        viewer_target: str | List[str] | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        return self._client.list_runtime_messages(
            run_id,
            status=status,
            viewer_role=viewer_role,
            viewer_target=viewer_target,
            limit=limit,
        )

    def enqueue_runtime_message(self, run_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.enqueue_runtime_message(run_id, **kwargs)

    def publish_manderqueue_message(
        self,
        run_id: str,
        *,
        project_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._client.publish_manderqueue_message(run_id, project_id=project_id, **kwargs)

    def send_message(
        self,
        run_id: str,
        *,
        project_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._client.send_message(run_id, project_id=project_id, **kwargs)

    def list_manderqueue_messages(
        self,
        run_id: str,
        *,
        project_id: str,
        **kwargs: Any,
    ) -> List[dict[str, Any]]:
        return self._client.list_manderqueue_messages(run_id, project_id=project_id, **kwargs)

    def list_messages(
        self,
        run_id: str,
        *,
        project_id: str,
        **kwargs: Any,
    ) -> List[dict[str, Any]]:
        return self._client.list_messages(run_id, project_id=project_id, **kwargs)

    def list_tasks(
        self,
        project_id: str,
        *,
        run_id: str | None = None,
        objective_id: str | None = None,
        kind: str | None = None,
        limit: int | None = None,
    ) -> List[dict[str, Any]]:
        return self._client.list_tasks(
            project_id,
            run_id=run_id,
            objective_id=objective_id,
            kind=kind,
            limit=limit,
        )

    def create_task(
        self,
        run_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        project_id: str,
        **kwargs: Any,
    ) -> RuntimeIntentReceipt:
        return self._client.create_task(run_id, payload, project_id=project_id, **kwargs)

    def update_task(
        self,
        run_id: str,
        task_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        project_id: str,
        **kwargs: Any,
    ) -> RuntimeIntentReceipt:
        return self._client.update_task(
            run_id,
            task_id,
            payload,
            project_id=project_id,
            **kwargs,
        )

    def cancel_task(
        self,
        run_id: str,
        task_id: str,
        *,
        project_id: str,
        **kwargs: Any,
    ) -> RuntimeIntentReceipt:
        return self._client.cancel_task(
            run_id,
            task_id,
            project_id=project_id,
            **kwargs,
        )

    def reassign_task(
        self,
        run_id: str,
        task_id: str,
        *,
        project_id: str,
        assignee: str,
        **kwargs: Any,
    ) -> RuntimeIntentReceipt:
        return self._client.reassign_task(
            run_id,
            task_id,
            project_id=project_id,
            assignee=assignee,
            **kwargs,
        )

    def list_manderqueue_threads(
        self,
        run_id: str,
        *,
        project_id: str,
        **kwargs: Any,
    ) -> List[dict[str, Any]]:
        return self._client.list_manderqueue_threads(run_id, project_id=project_id, **kwargs)

    def list_manderqueue_interactions(
        self,
        run_id: str,
        *,
        project_id: str,
        **kwargs: Any,
    ) -> List[dict[str, Any]]:
        return self._client.list_manderqueue_interactions(run_id, project_id=project_id, **kwargs)

    def edit_message(
        self,
        run_id: str,
        message_id: str,
        *,
        project_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._client.edit_message(
            run_id,
            message_id,
            project_id=project_id,
            **kwargs,
        )

    def retract_message(
        self,
        run_id: str,
        message_id: str,
        *,
        project_id: str,
    ) -> dict[str, Any]:
        return self._client.retract_message(
            run_id,
            message_id,
            project_id=project_id,
        )

    def transcript(
        self,
        run_id: str,
        *,
        cursor: str | None = None,
        limit: int = 200,
        participant_session_id: str | None = None,
        view: str | None = None,
    ) -> dict[str, Any]:
        """Fetch one page of transcript events.

        Returns a dict with ``events`` (list), ``next_cursor``, and
        ``live_resume_cursor``. Pass ``cursor=page["next_cursor"]`` to page
        forward. ``next_cursor`` is ``None`` when no more persisted events exist.

        For live runs, use ``live_resume_cursor`` as the starting cursor on
        subsequent calls to pick up new events as they arrive.
        """
        return self._client.get_run_transcript(
            run_id,
            cursor=cursor,
            limit=limit,
            participant_session_id=participant_session_id,
            view=view,
        )

    def stream_events(
        self,
        run_id: str | RunRef | EventStreamRequest,
        *,
        transcript_cursor: str | None = None,
        view: str = "operator",
        last_event_id: str | None = None,
        timeout: float | None = None,
    ):
        """Stream live runtime events for a run over backend SSE.

        Yields typed ``RunRuntimeStreamEvent`` instances. Transcript payloads are
        already projected by backend policy for the requested view.
        """
        if isinstance(run_id, EventStreamRequest):
            return self.stream(run_id, view=view, timeout=timeout)
        if isinstance(run_id, RunRef):
            return self.stream(
                EventStreamRequest(
                    run_ref=run_id,
                    transcript_cursor=transcript_cursor,
                    last_event_id=last_event_id,
                ),
                view=view,
                timeout=timeout,
            )
        return self._client.stream_run_events(
            run_id,
            transcript_cursor=transcript_cursor,
            view=view,
            last_event_id=last_event_id,
            timeout=timeout,
        )

    def stream(
        self,
        request: EventStreamRequest,
        *,
        view: str = "operator",
        timeout: float | None = None,
    ) -> EventStream:
        """Return a context-managed typed event stream for a run."""

        events = self._client.stream_run_events(
            request.run_ref.run_id,
            transcript_cursor=request.transcript_cursor,
            view=view,
            last_event_id=request.last_event_id,
            timeout=timeout or float(request.heartbeat_timeout_seconds),
        )
        return EventStream(events, request=request)

    def stream_transcript(
        self,
        run_id: str,
        *,
        cursor: str | None = None,
        page_size: int = 200,
        participant_session_id: str | None = None,
        view: str | None = None,
    ):
        """Iterate over all persisted transcript events for a run.

        Yields individual event dicts. Fetches pages until ``next_cursor`` is
        exhausted. Suitable for completed runs; for live runs use
        :meth:`transcript` in a poll loop with the returned ``live_resume_cursor``.
        """
        current_cursor = cursor
        while True:
            page = self._client.get_run_transcript(
                run_id,
                cursor=current_cursor,
                limit=page_size,
                participant_session_id=participant_session_id,
                view=view,
            )
            yield from page.get("events") or []
            next_cursor = page.get("next_cursor")
            if not next_cursor:
                break
            current_cursor = next_cursor


__all__ = ["MISC_PROJECT_ID", "ProjectSelector", "RunHandle", "RunsAPI"]
