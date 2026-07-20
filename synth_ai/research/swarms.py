"""``client.research.swarms`` — launch and lifecycle for Managed Swarms.

A Managed Swarm is one bounded multi-agent execution under the Managed
Research (SMR) umbrella. Swarms launch directly against a project, or are
composed by Managed Factories through Efforts. The wire protocol still uses
``run``/``run_id``; this module is the public noun layer over it.

**Status:** alpha
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, List, Literal, cast

from synth_ai.managed_research.errors import (
    SmrApiError,
    SmrConcurrentRunLimitExceededError,
)
from synth_ai.managed_research.models.canonical_usage import (
    SmrResourceLimitProgress,
    SmrResourceLimits,
    SmrRunUsage,
)
from synth_ai.managed_research.models.run_control import (
    ManagedResearchRunControlAck,
)
from synth_ai.managed_research.models.run_diagnostics import SmrRunCostSummary
from synth_ai.managed_research.models.run_events import RunRuntimeStreamEvent
from synth_ai.managed_research.models.run_observability import (
    RunObservabilitySnapshot,
)
from synth_ai.managed_research.models.run_state import RunState
from synth_ai.managed_research.sdk.client import ManagedResearchClient
from synth_ai.managed_research.sdk.runs import ProjectSelector, RunHandle
from synth_ai.research.models import (
    ResearchRunbookPreset,
    ResearchSwarm,
    ResearchWorkProduct,
)
from synth_ai.research.swarm_readouts import ResearchSwarmReadoutsMixin, _deprecated_method
from synth_ai.sdk.pagination import SyncPage


def _resolve_swarm_id(swarm_id: str | None, run_id: str | None) -> str | None:
    if run_id is None:
        return swarm_id
    warnings.warn(
        "run_id= is deprecated on swarms methods; pass swarm_id= instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    if swarm_id is not None and swarm_id != run_id:
        raise ValueError("pass either swarm_id or run_id, not both")
    return run_id


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


def _swallow_readout_errors(fetch: Any) -> Any:
    """Best-effort readout fetch for failed swarms only; returns None on error."""
    try:
        return fetch()
    except Exception:
        return None


def swarm_state_is_terminal(state: str) -> bool:
    """Return whether a public swarm state string is terminal.

    Reuses :class:`RunState` (the same terminal-state source the low-level
    wait/contract path is projected from) instead of hand-authoring a string
    set. Unknown states are non-terminal.
    """
    try:
        parsed = RunState(str(state or "").strip().lower())
    except ValueError:
        return False
    return parsed.is_terminal


def classify_event_kind(kind: str) -> str:
    """Classify a wire event ``kind`` into a coarse forward-compatible category.

    Returns one of ``"tool"``, ``"message"``, ``"turn"``, ``"usage"``,
    ``"status"``, or ``"other"``. Unknown kinds always classify as ``"other"``
    so new backend event kinds never break consumers.
    """
    text = str(kind or "").strip().lower()
    if not text:
        return "other"
    if text.startswith("tool"):
        return "tool"
    if text.startswith(("message", "operator.message", "runtime.message", "reasoning")):
        return "message"
    if text.startswith("turn"):
        return "turn"
    if text.startswith(("token", "usage")):
        return "usage"
    if ".state.changed" in text or text in {"heartbeat", "snapshot"}:
        return "status"
    return "other"


class SwarmPreflightBlockedError(SmrApiError):
    """Raised when ``launch_and_wait`` preflight is not clear to trigger.

    Carries the structured ``blockers`` list and the full preflight payload so
    callers can act on the class of denial rather than a bare message.
    """

    def __init__(
        self,
        message: str,
        *,
        blockers: list[dict[str, Any]],
        preflight: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.blockers = blockers
        self.preflight = preflight


class SwarmLaunchBackpressureError(SmrApiError):
    """Raised when swarm launch exhausts its bounded backpressure retries."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_error: SmrApiError,
    ) -> None:
        super().__init__(message, status_code=last_error.status_code)
        self.attempts = attempts
        self.last_error = last_error


_LAUNCH_RETRY_MAX_ATTEMPTS = 5
_LAUNCH_RETRY_MAX_SLEEP_SECONDS = 30.0
_LAUNCH_RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})


def _is_retryable_launch_error(exc: SmrApiError) -> bool:
    if isinstance(exc, SmrConcurrentRunLimitExceededError):
        return True
    return exc.status_code in _LAUNCH_RETRYABLE_STATUS_CODES


@dataclass(frozen=True)
class SwarmResult:
    """Typed outcome of :meth:`ResearchSwarmsAPI.launch_and_wait`."""

    swarm: ResearchSwarm
    swarm_id: str
    project_id: str
    status: str
    is_success: bool
    usage: SmrRunUsage | None
    cost: SmrRunCostSummary | None
    work_products: List[ResearchWorkProduct]
    handle: ResearchSwarmHandle

    @property
    def is_terminal(self) -> bool:
        return swarm_state_is_terminal(self.status)


@dataclass(frozen=True)
class SwarmRetryResult:
    """Typed outcome of :meth:`ResearchSwarmHandle.retry`.

    Provenance (which swarm this retry came from and why) lives here
    client-side; the launch wire has no metadata field to carry it.
    """

    source_swarm_id: str
    new_swarm_id: str
    mode: str
    reason: str | None
    checkpoint_id: str | None
    handle: ResearchSwarmHandle


class ResearchSwarmHandle(ResearchSwarmReadoutsMixin, RunHandle):
    """Swarm-scoped readouts and lifecycle (public hero session type).

    A Managed Swarm is one bounded multi-agent execution. Prefer
    ``ResearchSwarmSession`` in type hints — ``RunHandle`` is not part of the
    public hero surface.
    """

    def __init__(self, handle: RunHandle) -> None:
        super().__init__(handle._client, handle.project_id, handle.run_id)

    @property
    def swarm_id(self) -> str:
        """Public swarm identifier (wire transport still calls this ``run_id``)."""
        return self.run_id

    def wait_until_terminal(
        self,
        timeout: float,
        poll_interval: float = 10.0,
    ) -> ResearchSwarm:
        """Block until the swarm reaches a terminal state (thin over ``wait``).

        Args:
            timeout: Max seconds to wait; the operator owns the wall clock.
            poll_interval: Seconds between status polls.

        Returns:
            Final :class:`ResearchSwarm` public state model.
        """
        return self.wait(timeout=timeout, poll_interval=poll_interval)

    def is_terminal(self) -> bool:
        """Return whether the swarm has reached a terminal state.

        Uses the backend run contract's ``terminal`` flag — the same authority
        the low-level wait loop polls.
        """
        return self.contract().terminal

    def retry(
        self,
        mode: Literal["from_checkpoint", "fresh"] = "fresh",
        *,
        reason: str | None = None,
        checkpoint_id: str | None = None,
    ) -> SwarmRetryResult:
        """Retry a terminal swarm, either fresh or branched from a checkpoint.

        Args:
            mode: ``"fresh"`` re-launches the project's configured swarm;
                ``"from_checkpoint"`` branches from an existing checkpoint.
            reason: Optional operator reason, recorded client-side on the
                result (and on the branch request for checkpoint retries).
            checkpoint_id: Required when ``mode="from_checkpoint"``.

        Returns:
            :class:`SwarmRetryResult` with the new swarm id and handle.

        Raises:
            ValueError: If the source swarm is not terminal, or the mode /
                checkpoint arguments are inconsistent.
        """
        if mode not in ("from_checkpoint", "fresh"):
            raise ValueError(f"mode must be 'from_checkpoint' or 'fresh', got {mode!r}")
        state = self.get().public_state
        if not state.is_terminal:
            raise ValueError(
                f"swarm {self.swarm_id} is not terminal (state {state.value}); "
                "retry requires a terminal source swarm"
            )
        if mode == "from_checkpoint":
            if checkpoint_id is None:
                raise ValueError("checkpoint_id is required when mode is 'from_checkpoint'")
            branch = self.branch_from_checkpoint(
                checkpoint_id=checkpoint_id,
                reason=reason,
            )
            new_swarm_id = branch.child_run_id
        else:
            if checkpoint_id is not None:
                raise ValueError("checkpoint_id is only valid when mode is 'from_checkpoint'")
            wire = self._client.runs.trigger(self.project_id)
            new_swarm_id = ResearchSwarm.from_wire(wire).run_id
        new_handle = ResearchSwarmHandle(self._client.run(self.project_id, new_swarm_id))
        return SwarmRetryResult(
            source_swarm_id=self.swarm_id,
            new_swarm_id=new_swarm_id,
            mode=mode,
            reason=reason,
            checkpoint_id=checkpoint_id,
            handle=new_handle,
        )

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
            "ResearchSwarmHandle.progress_snapshot()",
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
        """Stream transcript event pages for the swarm."""
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
            "ResearchSwarmHandle.download_workspace_archive()",
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
            "ResearchSwarmHandle.list_artifacts()",
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


class ResearchSwarmsAPI:
    """Public Managed Swarm methods (alpha must-have)."""

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def runbook_presets(self) -> tuple[ResearchRunbookPreset, ...]:
        """Return supported runbook presets for ``swarms.create``."""
        return self._session.runs.runbook_presets()

    def check_preflight(
        self,
        project_id: str | None = None,
        *,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Validate a launch request before starting a swarm.

        Call after ``projects.setup.prepare`` to surface blockers (missing repo,
        secrets, budget caps) without creating a swarm record.

        Returns:
            Preflight payload with ``allowed`` flag and structured denials.

        Example:
            research.projects.setup.prepare(project_id)
            preflight = research.swarms.check_preflight(project_id, work_mode="directed_effort")
            if not preflight.get("allowed"):
                raise RuntimeError(preflight)
        """
        return self._session.runs.launch_preflight(
            project_id,
            project=project,
            **_research_run_kwargs(kwargs),
        )

    def launch_and_wait(
        self,
        project_id: str | None = None,
        *,
        project: ProjectSelector | str | None = None,
        objective: str | None = None,
        timeout: float,
        poll_interval: float = 10.0,
        raise_if_failed: bool = False,
        **launch_kwargs: Any,
    ) -> SwarmResult:
        """Preflight, launch, and wait for a swarm in one call (hero flow).

        Runs ``check_preflight`` first and fails immediately with a typed
        :class:`SwarmPreflightBlockedError` when not clear to trigger. Launch
        is wrapped in a bounded backpressure retry (HTTP 502/503/504 and
        concurrent-run-limit backpressure, max 5 attempts, capped
        backoff); other errors raise immediately. Then blocks until terminal
        and assembles a typed :class:`SwarmResult`.

        Args:
            project_id: Owning project id.
            objective: Primary operator message (objective launch path). When
                omitted, the project's configured swarm is triggered.
            timeout: REQUIRED max seconds to wait for terminal state — the
                operator owns the wall clock; there is no wait-forever default.
            poll_interval: Seconds between status polls.
            raise_if_failed: Raise when the swarm ends failed/blocked.
            **launch_kwargs: Any launch-request field the backend accepts
                (``execution_target``, ``local_execution``, ``limit``, ...).

        Returns:
            :class:`SwarmResult` with final state, usage, cost, work products,
            and a live handle for further readouts.
        """
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        run_kwargs = _research_run_kwargs(launch_kwargs)
        preflight_kwargs = {
            key: value for key, value in run_kwargs.items() if key != "run_kind"
        }
        preflight = self._session.runs.launch_preflight(
            project_id,
            project=project,
            objective=objective,
            **preflight_kwargs,
        )
        clear = preflight.get("clear_to_trigger")
        if clear is None:
            clear = preflight.get("allowed")
        if not clear:
            blockers_payload = preflight.get("blockers") or preflight.get("checks") or []
            blockers = [item for item in blockers_payload if isinstance(item, dict)]
            names = ", ".join(
                str(item.get("blocker") or item.get("check") or item.get("kind") or "unnamed")
                for item in blockers
            )
            raise SwarmPreflightBlockedError(
                "swarm launch preflight is not clear to trigger"
                + (f"; blockers: {names}" if names else ""),
                blockers=blockers,
                preflight=preflight,
            )
        handle = self._launch_with_backpressure_retry(
            project_id,
            project=project,
            objective=objective,
            run_kwargs=run_kwargs,
        )
        swarm = handle.wait(
            timeout=timeout,
            poll_interval=poll_interval,
            raise_if_failed=raise_if_failed,
        )
        status = swarm.public_state.value
        is_success = swarm.public_state is RunState.DONE
        if is_success:
            usage: SmrRunUsage | None = handle.usage.get()
            cost: SmrRunCostSummary | None = handle.usage.cost.get()
            work_products = list(handle.work_products.list())
        else:
            usage = _swallow_readout_errors(handle.usage.get)
            cost = _swallow_readout_errors(handle.usage.cost.get)
            work_products = _swallow_readout_errors(handle.work_products.list) or []
        return SwarmResult(
            swarm=swarm,
            swarm_id=handle.swarm_id,
            project_id=handle.project_id,
            status=status,
            is_success=is_success,
            usage=usage,
            cost=cost,
            work_products=list(work_products),
            handle=handle,
        )

    def _launch_handle(
        self,
        project_id: str | None,
        *,
        project: ProjectSelector | str | None,
        objective: str | None,
        run_kwargs: dict[str, Any],
    ) -> ResearchSwarmHandle:
        if objective is not None:
            handle = self._session.runs.start(
                objective,
                project_id=project_id,
                project=project,
                **run_kwargs,
            )
            return ResearchSwarmHandle(handle)
        wire = self._session.runs.trigger(
            project_id,
            project=project,
            **run_kwargs,
        )
        run = ResearchSwarm.from_wire(wire)
        return ResearchSwarmHandle(self._session.run(run.project_id, run.run_id))

    def _launch_with_backpressure_retry(
        self,
        project_id: str | None,
        *,
        project: ProjectSelector | str | None,
        objective: str | None,
        run_kwargs: dict[str, Any],
    ) -> ResearchSwarmHandle:
        for attempt in range(1, _LAUNCH_RETRY_MAX_ATTEMPTS + 1):
            try:
                return self._launch_handle(
                    project_id,
                    project=project,
                    objective=objective,
                    run_kwargs=run_kwargs,
                )
            except SmrApiError as exc:
                if not _is_retryable_launch_error(exc):
                    raise
                if attempt == _LAUNCH_RETRY_MAX_ATTEMPTS:
                    raise SwarmLaunchBackpressureError(
                        f"swarm launch exhausted {_LAUNCH_RETRY_MAX_ATTEMPTS} attempts on "
                        f"retryable backpressure ({type(exc).__name__}, "
                        f"status_code={exc.status_code}): {exc}",
                        attempts=_LAUNCH_RETRY_MAX_ATTEMPTS,
                        last_error=exc,
                    ) from exc
                time.sleep(min(2.0**attempt, _LAUNCH_RETRY_MAX_SLEEP_SECONDS))
        raise RuntimeError("unreachable: launch retry loop must return or raise")

    def launch_preflight(
        self,
        project_id: str | None = None,
        *,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Deprecated alias for ``check_preflight``."""
        warnings.warn(
            "swarms.launch_preflight is deprecated; use swarms.check_preflight instead.",
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
    ) -> ResearchSwarmHandle | dict[str, Any]:
        """Launch a Managed Swarm.

        When ``objective`` is provided, returns a :class:`ResearchSwarmHandle`.
        Objective-less calls retain their historical raw response. Use
        :meth:`create_configured` for a typed configured-swarm launch.

        Args:
            project_id: Owning project id.
            objective: Primary operator message for the swarm (preferred launch path).

        Returns:
            A typed handle for objective launches, otherwise the legacy raw payload.

        Example:
            handle = research.swarms.create(
                project_id,
                objective="Audit the repo for security issues",
            )
            research.swarms.wait(project_id, handle.swarm_id)
        """
        run_kwargs = _research_run_kwargs(kwargs)
        if objective is not None:
            handle = self._session.runs.start(
                objective,
                project_id=project_id,
                project=project,
                **run_kwargs,
            )
            return ResearchSwarmHandle(handle)
        return self._session.runs.trigger(
            project_id,
            project=project,
            **run_kwargs,
        )

    def create_configured(
        self,
        project_id: str | None = None,
        *,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> ResearchSwarmHandle:
        """Launch the project's configured swarm and return a typed handle."""
        wire = self._session.runs.trigger(
            project_id,
            project=project,
            **_research_run_kwargs(kwargs),
        )
        run = ResearchSwarm.from_wire(wire)
        return ResearchSwarmHandle(self._session.run(run.project_id, run.run_id))

    def start(
        self,
        objective: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> ResearchSwarmHandle:
        """Start a swarm with a primary objective message (deprecated — use ``create``)."""
        warnings.warn(
            "swarms.start is deprecated; use swarms.create instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        handle = self._session.runs.start(
            objective,
            project_id=project_id,
            project=project,
            **_research_run_kwargs(kwargs),
        )
        return ResearchSwarmHandle(handle)

    def launch(
        self,
        objective: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
        **kwargs: Any,
    ) -> ResearchSwarmHandle:
        """Deprecated alias for ``create`` with a required objective."""
        warnings.warn(
            "swarms.launch is deprecated; use swarms.create instead.",
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
            "swarms.trigger is deprecated; use swarms.create instead.",
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
        """Deprecated alias for configured-swarm launch (prefer ``create``)."""
        warnings.warn(
            "swarms.start_run is deprecated; use swarms.create instead.",
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
        swarm_id: str | None = None,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
        run_id: str | None = None,
    ) -> ResearchSwarmHandle:
        """Open a swarm-scoped session handle for readouts and lifecycle control.

        Accepts ``(project_id, swarm_id)``, ``(swarm_id,)`` when project is implied,
        or keyword forms. Prefer nested readouts on the returned handle:

        ``handle.usage.get()``, ``handle.snapshots.get()``, ``handle.transcript.get()``.
        """
        swarm_id = _resolve_swarm_id(swarm_id, run_id)
        if len(args) > 2:
            raise TypeError("get() accepts at most two positional arguments")
        if len(args) == 1:
            if swarm_id is not None:
                raise TypeError("swarm_id was provided both positionally and by keyword")
            swarm_id = args[0]
        elif len(args) == 2:
            if project_id is not None or swarm_id is not None:
                raise TypeError("project_id/swarm_id were provided both positionally and by keyword")
            project_id, swarm_id = args
        if swarm_id is None:
            raise ValueError("swarm_id is required")
        if project is not None:
            if project_id is not None:
                raise ValueError("pass either project_id or project, not both")
            project_id = (
                project.project_id
                if isinstance(project, ProjectSelector)
                else ProjectSelector.from_project_id(project).project_id
            )
        if project_id is None:
            run = self._session.runs.get(swarm_id)
            project_id = run.project_id
        return ResearchSwarmHandle(self._session.run(project_id, swarm_id))

    def open(
        self,
        *args: str,
        swarm_id: str | None = None,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
        run_id: str | None = None,
    ) -> ResearchSwarmHandle:
        """Open a swarm session (alias for ``get``)."""
        return self.get(
            *args,
            swarm_id=_resolve_swarm_id(swarm_id, run_id),
            project_id=project_id,
            project=project,
        )

    def state(
        self,
        swarm_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> ResearchSwarm:
        """Return the public swarm state model without opening a full session handle."""
        return self.public_state(
            swarm_id,
            project_id=project_id,
            project=project,
        )

    def public_state(
        self,
        swarm_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> ResearchSwarm:
        """Return the public swarm state model without opening a full session handle."""
        return self.get(swarm_id=swarm_id, project_id=project_id, project=project).public_state()

    def list(
        self,
        project_id: str,
        *,
        active_only: bool = False,
        **kwargs: Any,
    ) -> List[dict[str, Any]]:
        """List swarms for a project (newest first)."""
        return self._session.runs.list(project_id, active_only=active_only, **kwargs)

    def list_active(self, project_id: str, **kwargs: Any) -> List[dict[str, Any]]:
        """Return active swarms for a project (eval-compat name)."""
        return self.list(project_id, active_only=True, **kwargs)

    def wait(
        self,
        project_id: str | None = None,
        swarm_id: str | None = None,
        *,
        project: ProjectSelector | str | None = None,
        timeout: float | None = None,
        poll_interval: float = 10.0,
        raise_if_failed: bool = False,
        run_id: str | None = None,
    ) -> ResearchSwarm:
        """Block until a swarm reaches a terminal state.

        Args:
            timeout: Max seconds to wait (``None`` waits indefinitely).
            poll_interval: Seconds between status polls.
            raise_if_failed: Raise when the swarm ends in a failed state.

        Returns:
            Final ``ResearchSwarm`` public state model.
        """
        swarm_id = _resolve_swarm_id(swarm_id, run_id)
        return self.get(swarm_id=swarm_id, project_id=project_id, project=project).wait(
            timeout=timeout,
            poll_interval=poll_interval,
            raise_if_failed=raise_if_failed,
        )

    def transcript(
        self,
        swarm_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
        cursor: str | None = None,
        limit: int = 200,
        participant_session_id: str | None = None,
        view: str | None = None,
    ) -> dict[str, Any]:
        """Fetch a transcript page for a swarm (prefer ``handle.transcript.get``)."""
        return self.get(swarm_id=swarm_id, project_id=project_id, project=project).transcript.get(
            cursor=cursor,
            limit=limit,
            participant_session_id=participant_session_id,
            view=view,
        )

    def stream_events(
        self,
        swarm_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
        transcript_cursor: str | None = None,
        view: str = "operator",
        last_event_id: str | None = None,
        timeout: float | None = None,
    ) -> Iterator[RunRuntimeStreamEvent]:
        """Stream runtime events for a swarm (prefer ``handle.events.stream``)."""
        return self.get(swarm_id=swarm_id, project_id=project_id, project=project).stream_events(
            transcript_cursor=transcript_cursor,
            view=view,
            last_event_id=last_event_id,
            timeout=timeout,
        )

    def resource_limits(
        self,
        swarm_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> SmrResourceLimits:
        """Return configured resource limits for the swarm."""
        return self.get(swarm_id=swarm_id, project_id=project_id, project=project).resource_limits()

    def progress_toward_resource_limits(
        self,
        swarm_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> SmrResourceLimitProgress:
        """Return progress toward resource limits for the swarm."""
        return self.get(
            swarm_id=swarm_id,
            project_id=project_id,
            project=project,
        ).progress_toward_resource_limits()

    def stop(
        self,
        swarm_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> ManagedResearchRunControlAck:
        """Request graceful stop for a swarm."""
        return self.get(swarm_id=swarm_id, project_id=project_id, project=project).stop()

    def pause(
        self,
        swarm_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> ManagedResearchRunControlAck:
        """Pause an active swarm."""
        return self.get(swarm_id=swarm_id, project_id=project_id, project=project).pause()

    def resume(
        self,
        swarm_id: str,
        *,
        project_id: str | None = None,
        project: ProjectSelector | str | None = None,
    ) -> ManagedResearchRunControlAck:
        """Resume a paused swarm."""
        return self.get(swarm_id=swarm_id, project_id=project_id, project=project).resume()

    def results(
        self,
        project_id: str,
        swarm_id: str,
    ) -> dict[str, Any]:
        """Return final swarm results when execution completes."""
        return self._session.get_run_results(project_id, swarm_id)

    def logs(
        self,
        project_id: str,
        swarm_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List structured log records for a swarm."""
        return self._session.get_run_logs(
            project_id,
            swarm_id,
            limit=limit,
            cursor=cursor,
        )

    def logs_page(
        self,
        project_id: str,
        swarm_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> SyncPage[dict[str, Any]]:
        """Fetch a paginated page of swarm logs.

        Returns:
            ``SyncPage`` with ``items``, ``next_cursor``, and ``has_more`` for
            cursor-based iteration without hand-parsing wire payloads.
        """
        from synth_ai.sdk.pagination import page_from_wire

        payload = self.logs(project_id, swarm_id, limit=limit, cursor=cursor)
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
        swarm_id: str,
        **kwargs: Any,
    ) -> Any:
        """Return orchestrator execution metadata for a swarm."""
        return self._session.get_run_execution(project_id, swarm_id, **kwargs)

    def orchestrator(
        self,
        project_id: str,
        swarm_id: str,
    ) -> dict[str, Any]:
        """Return orchestrator state for a swarm (actors, phases, checkpoints)."""
        return self._session.get_run_orchestrator(project_id, swarm_id)


ResearchSwarmSession = ResearchSwarmHandle

__all__ = [
    "ResearchSwarmHandle",
    "ResearchSwarmSession",
    "ResearchSwarmsAPI",
    "SwarmLaunchBackpressureError",
    "SwarmPreflightBlockedError",
    "SwarmResult",
    "SwarmRetryResult",
    "classify_event_kind",
    "swarm_state_is_terminal",
]
