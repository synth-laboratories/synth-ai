"""``client.research.runs`` — alpha run surface."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, List

from synth_ai.managed_research.models.canonical_usage import (
    SmrResourceLimitProgress,
    SmrResourceLimits,
    SmrRunUsage,
)
from synth_ai.managed_research.models.checkpoints import Checkpoint
from synth_ai.managed_research.models.run_control import (
    ManagedResearchActorControlAck,
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
from synth_ai.managed_research.models.run_events import RunRuntimeStreamEvent
from synth_ai.managed_research.models.run_execution import RunExecutionProjection
from synth_ai.managed_research.models.run_observability import (
    ManagedResearchRunContract,
    RunObservabilitySnapshot,
)
from synth_ai.managed_research.models.run_timeline import (
    SmrAuthorityReadouts,
    SmrLogicalTimeline,
    SmrRunEventLog,
)
from synth_ai.managed_research.models.types import RunArtifact, RunArtifactManifest
from synth_ai.managed_research.sdk.client import ManagedResearchClient
from synth_ai.managed_research.sdk.runs import ProjectSelector, RunHandle
from synth_ai.research.models import ResearchRun, ResearchRunbookPreset, ResearchWorkProduct


def _text(value: object) -> str:
    return str(value or "").strip()


def _first_text(payload: dict[str, Any], *names: str) -> str:
    for name in names:
        value = _text(payload.get(name))
        if value:
            return value
    return ""


def _normalize_directed_outcome(value: object) -> object:
    if not isinstance(value, dict):
        return value
    payload = dict(value)
    outcome_text = _first_text(
        payload,
        "outcome_text",
        "outcome",
        "target",
        "description",
        "title",
    )
    if not outcome_text:
        raise ValueError(
            "directed_outcome requires outcome_text, outcome, target, description, or title"
        )
    title = _first_text(payload, "title", "name") or outcome_text[:120]
    description = _first_text(payload, "description", "summary", "context") or outcome_text
    scope = _first_text(payload, "scope", "context", "description") or description
    payload.setdefault("title", title)
    payload.setdefault("description", description)
    payload.setdefault("scope", scope)
    payload.setdefault("outcome_text", outcome_text)
    return payload


def _normalize_open_ended_question(value: object) -> object:
    if not isinstance(value, dict):
        return value
    payload = dict(value)
    question_text = _first_text(
        payload,
        "question_text",
        "question",
        "prompt",
        "description",
        "title",
    )
    if not question_text:
        raise ValueError(
            "open_ended_question requires question_text, question, prompt, description, or title"
        )
    title = _first_text(payload, "title", "name") or question_text[:120]
    description = _first_text(payload, "description", "summary", "context") or question_text
    scope = _first_text(payload, "scope", "context", "description") or description
    payload.setdefault("title", title)
    payload.setdefault("description", description)
    payload.setdefault("scope", scope)
    payload.setdefault("question_text", question_text)
    return payload


def _research_run_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    payload = dict(kwargs)
    directed_outcome = payload.pop("directed_outcome", None)
    open_ended_question = payload.get("open_ended_question")
    if open_ended_question is not None:
        payload["open_ended_question"] = _normalize_open_ended_question(open_ended_question)
    if directed_outcome is not None:
        if payload.get("directed_effort_outcome") is not None:
            raise ValueError("pass either directed_outcome or directed_effort_outcome, not both")
        payload["directed_effort_outcome"] = _normalize_directed_outcome(directed_outcome)
    elif payload.get("directed_effort_outcome") is not None:
        payload["directed_effort_outcome"] = _normalize_directed_outcome(
            payload["directed_effort_outcome"]
        )
    return payload


class ResearchRunHandle:
    """Handle for one Research run (wait, readback, workspace archive)."""

    def __init__(self, handle: RunHandle) -> None:
        self._handle = handle

    @property
    def project_id(self) -> str:
        return self._handle.project_id

    @property
    def run_id(self) -> str:
        return self._handle.run_id

    def get(self) -> ResearchRun:
        return self._handle.get()

    def public_state(self) -> ResearchRun:
        return self._handle.public_state()

    def wait(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float = 10.0,
        raise_if_failed: bool = False,
    ) -> ResearchRun:
        return self._handle.wait(
            timeout=timeout,
            poll_interval=poll_interval,
            raise_if_failed=raise_if_failed,
        )

    def contract(self) -> ManagedResearchRunContract:
        return self._handle.contract()

    def wait_terminal(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float = 10.0,
    ) -> ManagedResearchRunContract:
        return self._handle.wait_terminal(
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def explain_blocker(self) -> str | None:
        return self._handle.explain_blocker()

    def progress(
        self,
        *,
        detail_level: str = "control",
        event_limit: int = 40,
        actor_limit: int = 25,
        task_limit: int = 40,
        question_limit: int = 10,
        timeline_limit: int = 10,
        message_limit: int = 8,
    ) -> RunObservabilitySnapshot:
        return self._handle._client.get_run_observability_snapshot(
            self.project_id,
            self.run_id,
            detail_level=detail_level,
            event_limit=event_limit,
            actor_limit=actor_limit,
            task_limit=task_limit,
            question_limit=question_limit,
            timeline_limit=timeline_limit,
            message_limit=message_limit,
        )

    def full_progress(self) -> RunObservabilitySnapshot:
        return self._handle._client.get_run_observability_snapshot_full(
            self.project_id,
            self.run_id,
        )

    def timeline(self) -> SmrLogicalTimeline:
        return self._handle.timeline()

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
        return self._handle.execution(
            view=view,
            event_limit=event_limit,
            actor_limit=actor_limit,
            task_limit=task_limit,
            message_limit=message_limit,
            work_product_limit=work_product_limit,
        )

    def transcript(
        self,
        *,
        cursor: str | None = None,
        limit: int = 200,
        participant_session_id: str | None = None,
        view: str | None = None,
    ) -> dict[str, Any]:
        return self._handle.transcript(
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
    ) -> Iterator[RunRuntimeStreamEvent]:
        return self._handle.stream_events(
            transcript_cursor=transcript_cursor,
            view=view,
            last_event_id=last_event_id,
            timeout=timeout,
        )

    def stream_transcript(
        self,
        *,
        cursor: str | None = None,
        page_size: int = 200,
        participant_session_id: str | None = None,
        view: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        return self._handle._client.runs.stream_transcript(
            self.run_id,
            cursor=cursor,
            page_size=page_size,
            participant_session_id=participant_session_id,
            view=view,
        )

    def messages(
        self,
        *,
        status: str | None = None,
        viewer_role: str | None = None,
        viewer_target: str | list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self._handle.messages(
            status=status,
            viewer_role=viewer_role,
            viewer_target=viewer_target,
            limit=limit,
        )

    def task_events(
        self,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._handle.task_events(limit=limit, cursor=cursor)

    def objective_events(
        self,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._handle.objective_events(limit=limit, cursor=cursor)

    def work_graph(self, *, limit: int | None = None) -> dict[str, Any]:
        return self._handle.work_graph(limit=limit)

    def event_log(
        self,
        *,
        sources: list[str] | None = None,
        event_kinds: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int | None = None,
    ) -> SmrRunEventLog:
        return self._handle.event_log(
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
        return self._handle.authority_readouts(
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
        return self._handle.operator_evidence(
            runtime_timeline_limit=runtime_timeline_limit,
            logical_timeline_limit=logical_timeline_limit,
            transcript_limit=transcript_limit,
            reconciliation_limit=reconciliation_limit,
        )

    def traces(self) -> SmrRunTraces:
        return self._handle.traces()

    def participants(self) -> SmrRunParticipants:
        return self._handle.participants()

    def artifact_progress(self) -> SmrRunArtifactProgress:
        return self._handle.artifact_progress()

    def actor_logs(self, **kwargs: Any) -> SmrRunActorLogs:
        return self._handle.actor_logs(**kwargs)

    def checkpoints(self) -> list[Checkpoint]:
        return self._handle.checkpoints()

    def checkpoint(self, checkpoint_id: str) -> Checkpoint:
        return self._handle.checkpoint(checkpoint_id)

    def stop(self) -> ManagedResearchRunControlAck:
        return self._handle.stop()

    def pause(self) -> ManagedResearchRunControlAck:
        return self._handle.pause()

    def resume(self) -> ManagedResearchRunControlAck:
        return self._handle.resume()

    def control_actor(
        self,
        actor_id: str,
        *,
        action: str,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> ManagedResearchActorControlAck:
        return self._handle.control_actor(
            actor_id,
            action=action,
            reason=reason,
            idempotency_key=idempotency_key,
        )

    def pause_actor(
        self,
        actor_id: str,
        *,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> ManagedResearchActorControlAck:
        return self._handle.pause_actor(
            actor_id,
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
        return self._handle.resume_actor(
            actor_id,
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
        return self._handle.interrupt_actor(
            actor_id,
            reason=reason,
            idempotency_key=idempotency_key,
        )

    def work_products(self) -> list[ResearchWorkProduct]:
        return self._handle.work_products()

    def reports(self) -> list[ResearchWorkProduct]:
        return self._handle.reports()

    def final_report(self) -> ResearchWorkProduct | None:
        return self._handle.final_report()

    def report_text(self, work_product_id: str | None = None) -> str | None:
        return self._handle.report_text(work_product_id)

    def work_product_content(
        self,
        work_product_id: str,
        *,
        as_text: bool = True,
    ) -> str | bytes:
        return self._handle._client.work_products.content(
            work_product_id,
            as_text=as_text,
        )

    def usage(self) -> SmrRunUsage:
        return self._handle._client.get_run_usage(self.run_id)

    def actor_usage(self) -> SmrRunActorUsage:
        return self._handle.actor_usage()

    def cost_summary(self) -> SmrRunCostSummary:
        return self._handle.cost_summary()

    def resource_limits(self) -> SmrResourceLimits:
        return self._handle.resource_limits()

    def progress_toward_resource_limits(self) -> SmrResourceLimitProgress:
        return self._handle.progress_toward_resource_limits()

    def artifact_manifest(self) -> RunArtifactManifest:
        return self._handle.artifact_manifest()

    def artifacts(
        self,
        *,
        artifact_type: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[RunArtifact]:
        return self._handle.artifacts(
            artifact_type=artifact_type,
            limit=limit,
            cursor=cursor,
        )

    def download_workspace_archive(
        self,
        destination: str,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return self._handle._client.download_run_workspace_archive(
            self.project_id,
            self.run_id,
            destination,
            timeout_seconds=timeout_seconds,
        )

    def list_artifacts(
        self,
        *,
        artifact_type: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            artifact.__dict__
            for artifact in self._handle._client.list_run_artifacts(
                self.run_id,
                project_id=self.project_id,
                artifact_type=artifact_type,
                limit=limit,
                cursor=cursor,
            )
        ]


class ResearchRunsAPI:
    """Public Research run methods (alpha must-have)."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def runbook_presets(self) -> tuple[ResearchRunbookPreset, ...]:
        return self._session.runs.runbook_presets()

    def launch_preflight(
        self,
        project_id: str | None = None,
        *,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._session.runs.launch_preflight(
            project_id,
            project=project,
            **_research_run_kwargs(kwargs),
        )

    def start(
        self,
        objective: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> ResearchRunHandle:
        """Start a run with a primary objective message (canonical name)."""
        handle = self._session.runs.start(
            objective,
            project_id=project_id,
            project=project,
            **_research_run_kwargs(kwargs),
        )
        return ResearchRunHandle(handle)

    def launch(
        self,
        objective: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> ResearchRunHandle:
        return self.start(
            objective,
            project_id=project_id,
            project=project,
            **kwargs,
        )

    def trigger(
        self,
        project_id: str | None = None,
        *,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compatibility alias for existing ReportBench drivers."""
        return self._session.runs.trigger(
            project_id,
            project=project,
            **_research_run_kwargs(kwargs),
        )

    def start_run(
        self,
        project_id: str | None = None,
        *,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._session.runs.start_run(
            project_id,
            project=project,
            **_research_run_kwargs(kwargs),
        )

    def get(
        self,
        *args: str,
        run_id: str | None = None,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> ResearchRunHandle:
        if len(args) > 2:
            raise TypeError("get() accepts at most two positional arguments")
        if len(args) == 1:
            if run_id is not None:
                raise TypeError("run_id was provided both positionally and by keyword")
            run_id = args[0]
        elif len(args) == 2:
            if project_id is not None or run_id is not None:
                raise TypeError("project_id/run_id were provided both positionally and by keyword")
            project_id, run_id = args
        if run_id is None:
            raise ValueError("run_id is required")
        if project is not None:
            if project_id is not None:
                raise ValueError("pass either project_id or project, not both")
            project_id = (
                project.project_id
                if isinstance(project, ProjectSelector)
                else ProjectSelector.from_project_id(project).project_id
            )
        if project_id is None:
            run = self._session.runs.get(run_id)
            project_id = run.project_id
        return ResearchRunHandle(self._session.run(project_id, run_id))

    def public_state(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> ResearchRun:
        return self.get(run_id=run_id, project_id=project_id, project=project).public_state()

    def list(
        self,
        project_id: str,
        *,
        active_only: bool = False,
        **kwargs: Any,
    ) -> List[dict[str, Any]]:
        return self._session.runs.list(project_id, active_only=active_only, **kwargs)

    def wait(
        self,
        project_id: str | None = None,
        run_id: str | None = None,
        *,
        project: ProjectSelector | str | None = None,
        timeout: float | None = None,
        poll_interval: float = 10.0,
        raise_if_failed: bool = False,
    ) -> ResearchRun:
        return self.get(project_id, run_id, project=project).wait(
            timeout=timeout,
            poll_interval=poll_interval,
            raise_if_failed=raise_if_failed,
        )

    def transcript(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
        cursor: str | None = None,
        limit: int = 200,
        participant_session_id: str | None = None,
        view: str | None = None,
    ) -> dict[str, Any]:
        return self.get(run_id=run_id, project_id=project_id, project=project).transcript(
            cursor=cursor,
            limit=limit,
            participant_session_id=participant_session_id,
            view=view,
        )

    def stream_events(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
        transcript_cursor: str | None = None,
        view: str = "operator",
        last_event_id: str | None = None,
        timeout: float | None = None,
    ) -> Iterator[RunRuntimeStreamEvent]:
        return self.get(run_id=run_id, project_id=project_id, project=project).stream_events(
            transcript_cursor=transcript_cursor,
            view=view,
            last_event_id=last_event_id,
            timeout=timeout,
        )

    def resource_limits(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> SmrResourceLimits:
        return self.get(run_id=run_id, project_id=project_id, project=project).resource_limits()

    def progress_toward_resource_limits(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> SmrResourceLimitProgress:
        return self.get(
            run_id=run_id,
            project_id=project_id,
            project=project,
        ).progress_toward_resource_limits()

    def stop(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> ManagedResearchRunControlAck:
        return self.get(run_id=run_id, project_id=project_id, project=project).stop()

    def pause(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> ManagedResearchRunControlAck:
        return self.get(run_id=run_id, project_id=project_id, project=project).pause()

    def resume(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> ManagedResearchRunControlAck:
        return self.get(run_id=run_id, project_id=project_id, project=project).resume()


__all__ = ["ResearchRunHandle", "ResearchRunsAPI"]
