"""``client.research.runs`` — launch and lifecycle for Managed Research runs.

**Status:** alpha
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any, List, cast

from synth_ai.managed_research.models.canonical_usage import (
    SmrResourceLimitProgress,
    SmrResourceLimits,
)
from synth_ai.managed_research.models.run_control import (
    ManagedResearchRunControlAck,
)
from synth_ai.managed_research.models.run_events import RunRuntimeStreamEvent
from synth_ai.managed_research.models.run_observability import (
    RunObservabilitySnapshot,
)
from synth_ai.managed_research.sdk.client import ManagedResearchClient
from synth_ai.managed_research.sdk.runs import ProjectSelector, RunHandle
from synth_ai.research.models import ResearchRun, ResearchRunbookPreset
from synth_ai.research.run_readouts import ResearchRunReadoutsMixin, _deprecated_method
from synth_ai.sdk.pagination import SyncPage


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


class ResearchRunHandle(ResearchRunReadoutsMixin, RunHandle):
    """Run-scoped readouts and lifecycle (public hero session type).

    Prefer ``ResearchRunSession`` in type hints — ``RunHandle`` is not part of
    the public hero surface.
    """

    def __init__(self, handle: RunHandle) -> None:
        super().__init__(handle._client, handle.project_id, handle.run_id)

    def progress_snapshot(
        self,
        *,
        detail_level: str = "control",
        event_limit: int = 40,
        actor_limit: int = 25,
        task_limit: int = 40,
        question_limit: int = 10,
        timeline_limit: int = 10,
        message_limit: int = 8,
    ) -> Any:
        """Deprecated alias for ``snapshots.get(detail=...)``."""
        _deprecated_method(
            "ResearchRunHandle.progress_snapshot()",
            "handle.snapshots.get(detail=...)",
        )
        return self.snapshots.get(
            detail=detail_level,
            event_limit=event_limit,
            actor_limit=actor_limit,
            task_limit=task_limit,
            question_limit=question_limit,
            timeline_limit=timeline_limit,
            message_limit=message_limit,
        )

    def full_progress(self) -> RunObservabilitySnapshot:
        """Return the full observability snapshot (deprecated path — use ``snapshots.get(detail='full')``)."""
        return self.snapshots.get(detail="full")

    def stream_transcript(
        self,
        *,
        cursor: str | None = None,
        page_size: int = 200,
        participant_session_id: str | None = None,
        view: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream transcript event pages for the run."""
        return self._client.runs.stream_transcript(
            self.run_id,
            cursor=cursor,
            page_size=page_size,
            participant_session_id=participant_session_id,
            view=view,
        )

    def work_product_content(
        self,
        work_product_id: str,
        *,
        as_text: bool = True,
    ) -> str | bytes:
        """Download work product bytes or text by id."""
        return self.work_products.content.get(work_product_id, as_text=as_text)

    def download_workspace_archive(
        self,
        destination: str,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Deprecated alias for ``workspace.download``."""
        _deprecated_method(
            "ResearchRunHandle.download_workspace_archive()",
            "handle.workspace.download(...)",
        )
        return self.workspace.download(destination, timeout_seconds=timeout_seconds)

    def list_artifacts(
        self,
        *,
        artifact_type: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> List[dict[str, Any]]:
        """Deprecated alias for ``artifacts.list``."""
        _deprecated_method(
            "ResearchRunHandle.list_artifacts()",
            "handle.artifacts.list(...)",
        )
        return [
            artifact.__dict__ if hasattr(artifact, "__dict__") else dict(artifact)
            for artifact in self.artifacts.list(
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
        """Return supported runbook presets for ``runs.create``."""
        return self._session.runs.runbook_presets()

    def check_preflight(
        self,
        project_id: str | None = None,
        *,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Validate a launch request before starting a run.

        Call after ``projects.setup.prepare`` to surface blockers (missing repo,
        secrets, budget caps) without creating a run record.

        Returns:
            Preflight payload with ``allowed`` flag and structured denials.

        Example:
            research.projects.setup.prepare(project_id)
            preflight = research.runs.check_preflight(project_id, work_mode="directed_effort")
            if not preflight.get("allowed"):
                raise RuntimeError(preflight)
        """
        return self._session.runs.launch_preflight(
            project_id,
            project=project,
            **_research_run_kwargs(kwargs),
        )

    def launch_preflight(
        self,
        project_id: str | None = None,
        *,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Deprecated alias for ``check_preflight``."""
        warnings.warn(
            "runs.launch_preflight is deprecated; use runs.check_preflight instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.check_preflight(
            project_id,
            project=project,
            **kwargs,
        )

    def create(
        self,
        project_id: str | None = None,
        *,
        project: ProjectSelector | str | None = None,
        objective: str | None = None,
        **kwargs: Any,
    ) -> ResearchRunHandle:
        """Launch a Managed Research run.

        Always returns a :class:`ResearchRunHandle`. When ``objective`` is omitted,
        the backend starts the project's configured run and the hero client parses
        the response into the same typed handle. Legacy callers that require the
        raw trigger response can use the deprecated :meth:`trigger` method.

        Args:
            project_id: Owning project id.
            objective: Primary operator message for the run (preferred launch path).

        Returns:
            A typed :class:`ResearchRunHandle` for both launch forms.

        Example:
            handle = research.runs.create(
                project_id,
                objective="Audit the repo for security issues",
            )
            research.runs.wait(project_id, handle.run_id)
        """
        run_kwargs = _research_run_kwargs(kwargs)
        if objective is not None:
            handle = self._session.runs.start(
                objective,
                project_id=project_id,
                project=project,
                **run_kwargs,
            )
            return ResearchRunHandle(handle)
        wire = self._session.runs.trigger(
            project_id,
            project=project,
            **run_kwargs,
        )
        run = ResearchRun.from_wire(wire)
        return ResearchRunHandle(self._session.run(run.project_id, run.run_id))

    def start(
        self,
        objective: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> ResearchRunHandle:
        """Start a run with a primary objective message (deprecated — use ``create``)."""
        warnings.warn(
            "runs.start is deprecated; use runs.create instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        """Deprecated alias for ``create`` with a required objective."""
        warnings.warn(
            "runs.launch is deprecated; use runs.create instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        warnings.warn(
            "runs.trigger is deprecated; use runs.create instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        """Deprecated alias for configured-run launch (prefer ``create``)."""
        warnings.warn(
            "runs.start_run is deprecated; use runs.create instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        """Open a run-scoped session handle for readouts and lifecycle control.

        Accepts ``(project_id, run_id)``, ``(run_id,)`` when project is implied,
        or keyword forms. Prefer nested readouts on the returned handle:

        ``handle.usage.get()``, ``handle.snapshots.get()``, ``handle.transcript.get()``.
        """
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

    def open(
        self,
        *args: str,
        run_id: str | None = None,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> ResearchRunHandle:
        """Open a run session (alias for ``get``)."""
        return self.get(
            *args,
            run_id=run_id,
            project_id=project_id,
            project=project,
        )

    def state(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> ResearchRun:
        """Return the public run model without opening a full session handle."""
        return self.public_state(
            run_id,
            project_id=project_id,
            project=project,
        )

    def public_state(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> ResearchRun:
        """Return the public run model without opening a full session handle."""
        return self.get(run_id=run_id, project_id=project_id, project=project).public_state()

    def list(
        self,
        project_id: str,
        *,
        active_only: bool = False,
        **kwargs: Any,
    ) -> List[dict[str, Any]]:
        """List runs for a project (newest first)."""
        return self._session.runs.list(project_id, active_only=active_only, **kwargs)

    def list_active(self, project_id: str, **kwargs: Any) -> List[dict[str, Any]]:
        """Return active runs for a project (eval-compat name)."""
        return self.list(project_id, active_only=True, **kwargs)

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
        """Block until a run reaches a terminal state.

        Args:
            timeout: Max seconds to wait (``None`` waits indefinitely).
            poll_interval: Seconds between status polls.
            raise_if_failed: Raise when the run ends in a failed state.

        Returns:
            Final ``ResearchRun`` public state model.
        """
        return self.get(run_id=run_id, project_id=project_id, project=project).wait(
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
        """Fetch a transcript page for a run (prefer ``handle.transcript.get``)."""
        return self.get(run_id=run_id, project_id=project_id, project=project).transcript.get(
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
        """Stream runtime events for a run (prefer ``handle.events.stream``)."""
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
        """Return configured resource limits for the run."""
        return self.get(run_id=run_id, project_id=project_id, project=project).resource_limits()

    def progress_toward_resource_limits(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> SmrResourceLimitProgress:
        """Return progress toward resource limits for the run."""
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
        """Request graceful stop for a run."""
        return self.get(run_id=run_id, project_id=project_id, project=project).stop()

    def pause(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> ManagedResearchRunControlAck:
        """Pause a running run."""
        return self.get(run_id=run_id, project_id=project_id, project=project).pause()

    def resume(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> ManagedResearchRunControlAck:
        """Resume a paused run."""
        return self.get(run_id=run_id, project_id=project_id, project=project).resume()

    def results(
        self,
        project_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        """Return final run results when execution completes."""
        return self._session.get_run_results(project_id, run_id)

    def logs(
        self,
        project_id: str,
        run_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List structured log records for a run."""
        return self._session.get_run_logs(
            project_id,
            run_id,
            limit=limit,
            cursor=cursor,
        )

    def logs_page(
        self,
        project_id: str,
        run_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> SyncPage[dict[str, Any]]:
        """Fetch a paginated page of run logs.

        Returns:
            ``SyncPage`` with ``items``, ``next_cursor``, and ``has_more`` for
            cursor-based iteration without hand-parsing wire payloads.
        """
        from synth_ai.sdk.pagination import page_from_wire

        payload = self.logs(project_id, run_id, limit=limit, cursor=cursor)
        raw_items, next_cursor, has_more = page_from_wire(payload)
        if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
            raw_items = payload["entries"]
        elif isinstance(payload, dict) and isinstance(payload.get("logs"), list):
            raw_items = payload["logs"]
        elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
            raw_items = payload["records"]
        normalized = [cast(dict[str, Any], item) for item in raw_items if isinstance(item, dict)]
        return SyncPage(items=normalized, next_cursor=next_cursor, has_more=has_more)

    def execution(
        self,
        project_id: str,
        run_id: str,
        **kwargs: Any,
    ) -> Any:
        """Return orchestrator execution metadata for a run."""
        return self._session.get_run_execution(project_id, run_id, **kwargs)

    def orchestrator(
        self,
        project_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        """Return orchestrator state for a run (actors, phases, checkpoints)."""
        return self._session.get_run_orchestrator(project_id, run_id)


ResearchRunSession = ResearchRunHandle

__all__ = ["ResearchRunHandle", "ResearchRunSession", "ResearchRunsAPI"]
