"""Internal run-authority operations mixed into the Managed Research client."""

from __future__ import annotations

import json as _json
import os
import re
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import httpx

from synth_ai.managed_research.errors import SmrApiError
from synth_ai.managed_research.models import Checkpoint
from synth_ai.managed_research.models.operator_evidence import SmrRunOperatorEvidence
from synth_ai.managed_research.models.run_authority import ManagedResearchRunTask
from synth_ai.managed_research.models.run_control import ManagedResearchRunControlError
from synth_ai.managed_research.models.run_diagnostics import (
    SmrRunActorLogs,
    SmrRunActorUsage,
    SmrRunArtifactProgress,
    SmrRunCostSummary,
    SmrRunParticipants,
    SmrRunTraces,
)
from synth_ai.managed_research.models.run_events import RunRuntimeStreamEvent
from synth_ai.managed_research.models.run_observability import (
    MessageQueueInteraction,
    MessageQueueMessage,
    MessageQueueThread,
    RunObservabilitySnapshot,
    RunObservationCursor,
    TaskSummary,
)
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
from synth_ai.managed_research.sdk._client_helpers import (
    _coerce_branch_request,
    _coerce_dict,
    _coerce_dict_list,
    _optional_mapping,
    _require_non_empty_string,
)
from synth_ai.managed_research.transport.http import _raise_for_error_response
from synth_ai.managed_research.transport.pagination import build_query_params


class ManagedResearchRunAuthorityMixin:
    """Run authority, observability, control, trace, and messaging operations."""

    def list_run_task_events(
        self,
        project_id: str,
        run_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = int(limit)
        if cursor and cursor.strip():
            params["cursor"] = cursor.strip()
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/task-events",
                params=params or None,
            ),
            label="list_run_task_events",
        )

    def list_run_objective_events(
        self,
        project_id: str,
        run_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = int(limit)
        if cursor and cursor.strip():
            params["cursor"] = cursor.strip()
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/objective-events",
                params=params or None,
            ),
            label="list_run_objective_events",
        )

    def get_run_progress(
        self,
        project_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/progress",
            ),
            label="get_run_progress",
        )

    def list_run_primary_parent_milestones(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Eval-compat shim: milestones embedded in run progress payload."""
        resolved_project_id = str(project_id or "").strip()
        if not resolved_project_id:
            run_payload = self.get_run(run_id)
            resolved_project_id = str(run_payload.get("project_id") or "").strip()
        if not resolved_project_id:
            raise ValueError(f"run_id {run_id!r} missing project_id")
        progress = self.get_run_progress(resolved_project_id, run_id)
        milestones = progress.get("primary_parent_milestones")
        if isinstance(milestones, list):
            return [item for item in milestones if isinstance(item, dict)]
        return []

    def get_run_results(
        self,
        project_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/results",
            ),
            label="get_run_results",
        )

    def get_run_logs(
        self,
        project_id: str,
        run_id: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/logs",
                params=build_query_params(limit=limit, cursor=cursor),
            ),
            label="get_run_logs",
        )

    def get_run_orchestrator(
        self,
        project_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/orchestrator",
            ),
            label="get_run_orchestrator",
        )

    def get_run_work_graph(
        self,
        project_id: str,
        run_id: str,
        *,
        limit: int | None = None,
    ) -> dict[str, Any]:
        params = {"limit": int(limit)} if limit is not None else None
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/work-graph",
                params=params,
            ),
            label="get_run_work_graph",
        )

    def poll_run_observability_snapshot(
        self,
        project_id: str,
        run_id: str,
        *,
        cursor: RunObservationCursor | Mapping[str, Any] | None = None,
        detail_level: str = "full",
        event_limit: int = 100,
        actor_limit: int = 25,
        task_limit: int = 50,
        question_limit: int = 25,
        timeline_limit: int = 10,
        message_limit: int = 10,
    ) -> RunObservabilitySnapshot:
        resolved_cursor = (
            cursor
            if isinstance(cursor, RunObservationCursor)
            else RunObservationCursor.from_wire(cursor or {})
        )
        return self.get_run_observability_snapshot(
            project_id,
            run_id,
            since_event_seq=resolved_cursor.latest_event_seq,
            latest_runtime_message_seq=resolved_cursor.latest_runtime_message_seq,
            latest_runtime_event_id=resolved_cursor.latest_runtime_event_id,
            detail_level=detail_level,
            event_limit=event_limit,
            actor_limit=actor_limit,
            task_limit=task_limit,
            question_limit=question_limit,
            timeline_limit=timeline_limit,
            message_limit=message_limit,
        )

    def get_run_transcript(
        self,
        run_id: str,
        *,
        cursor: str | None = None,
        limit: int = 200,
        participant_session_id: str | None = None,
        view: str | None = None,
    ) -> dict[str, Any]:
        params = build_query_params(
            cursor=cursor,
            limit=limit,
            participant_session_id=participant_session_id,
            view=view,
        )
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/runs/{run_id}/runtime/transcript",
                params=params,
            ),
            label="get_run_transcript",
        )

    def stream_run_events(
        self,
        run_id: str,
        *,
        transcript_cursor: str | None = None,
        view: str = "operator",
        last_event_id: str | None = None,
        timeout: float | None = None,
    ):
        params = build_query_params(
            transcript_cursor=transcript_cursor,
            view=view,
        )
        for event in self._stream_sse(
            f"/smr/runs/{run_id}/runtime/stream",
            params=params,
            last_event_id=last_event_id,
            timeout=timeout,
        ):
            yield RunRuntimeStreamEvent.from_sse(event)

    def list_objectives(
        self,
        project_id: str,
        *,
        kind: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/objectives",
                params=build_query_params(kind=kind, run_id=run_id, limit=limit),
            ),
            label="list_objectives",
        )

    def list_directed_effort_outcomes(
        self,
        project_id: str,
        *,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Compatibility alias for eval harnesses (use ``list_objectives``)."""
        return self.list_objectives(
            project_id,
            kind="directed_effort_outcome",
            run_id=run_id,
            limit=limit,
        )

    def get_objective_status(
        self,
        project_id: str,
        objective_id: str,
        *,
        kind: str | None = None,
        task_limit: int | None = None,
        claim_limit: int | None = None,
        event_limit: int | None = None,
        milestone_limit: int | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/objectives/{objective_id}/status",
                params=build_query_params(
                    kind=kind,
                    task_limit=task_limit,
                    claim_limit=claim_limit,
                    event_limit=event_limit,
                    milestone_limit=milestone_limit,
                ),
            ),
            label="get_objective_status",
        )

    def get_objective_progress(
        self,
        project_id: str,
        objective_id: str,
        *,
        kind: str | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/objectives/{objective_id}/progress",
                params=build_query_params(kind=kind),
            ),
            label="get_objective_progress",
        )

    def list_objective_progress_claims(
        self,
        project_id: str,
        objective_id: str,
        *,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/objectives/{objective_id}/claims",
                params=build_query_params(kind=kind, limit=limit),
            ),
            label="list_objective_progress_claims",
        )

    def create_objective_progress_claim(
        self,
        project_id: str,
        objective_id: str,
        *,
        payload: Mapping[str, Any],
        kind: str | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/objectives/{objective_id}/claims",
                params=build_query_params(kind=kind),
                json_body=dict(payload),
            ),
            label="create_objective_progress_claim",
        )

    def list_milestones(
        self,
        project_id: str,
        *,
        run_id: str | None = None,
        parent_kind: str | None = None,
        parent_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/milestones",
                params=build_query_params(
                    run_id=run_id,
                    parent_kind=parent_kind,
                    parent_id=parent_id,
                    limit=limit,
                ),
            ),
            label="list_milestones",
        )

    def list_project_milestones(
        self,
        project_id: str,
        *,
        run_id: str | None = None,
        parent_kind: str | None = None,
        parent_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Compatibility alias for eval harnesses (use ``list_milestones``)."""
        return self.list_milestones(
            project_id,
            run_id=run_id,
            parent_kind=parent_kind,
            parent_id=parent_id,
            limit=limit,
        )

    def create_milestone(
        self,
        project_id: str,
        *,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/milestones",
                json_body=dict(payload),
            ),
            label="create_milestone",
        )

    def get_milestone(self, project_id: str, milestone_id: str) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/milestones/{milestone_id}",
            ),
            label="get_milestone",
        )

    def patch_milestone(
        self,
        project_id: str,
        milestone_id: str,
        *,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "PATCH",
                f"/smr/projects/{project_id}/milestones/{milestone_id}",
                json_body=dict(payload),
            ),
            label="patch_milestone",
        )

    def transition_milestone(
        self,
        project_id: str,
        milestone_id: str,
        *,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/milestones/{milestone_id}/transition",
                json_body=dict(payload),
            ),
            label="transition_milestone",
        )

    def list_tasks(
        self,
        project_id: str,
        *,
        run_id: str | None = None,
        objective_id: str | None = None,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[ManagedResearchRunTask]:
        """Return project-scoped task DTOs from the backend owner route."""

        if not run_id:
            if objective_id:
                raise SmrApiError(
                    "Objective-scoped task listing is not available in the current "
                    "Managed Research backend contract; use run-scoped tasks or "
                    "run objective events instead.",
                    failure_class="unsupported_backend_contract",
                )
            raise ValueError("run_id or objective_id is required")
        tasks = [
            ManagedResearchRunTask.from_wire(item)
            for item in self.list_tasks_raw(
                project_id,
                run_id=run_id,
                kind=kind,
                limit=limit,
            )
        ]
        for task in tasks:
            if task.project_id != project_id or task.run_id != run_id:
                raise SmrApiError(
                    "Task owner-route identity does not match the requested project/run",
                    failure_class="backend_contract_mismatch",
                )
        return tasks

    def list_tasks_raw(
        self,
        project_id: str,
        *,
        run_id: str,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Legacy serialized task rows; canonical SDK callers use ``list_tasks``."""

        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/tasks",
                params=build_query_params(kind=kind, limit=limit),
            ),
            label="list_tasks",
        )

    def list_task_summaries(
        self,
        project_id: str,
        *,
        run_id: str | None = None,
        objective_id: str | None = None,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[TaskSummary]:
        payload = self._request_json(
            "GET",
            f"/smr/projects/{project_id}/sublinear/tasks",
            params=build_query_params(limit=limit),
        )
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, Mapping):
            raw_items = payload.get("tasks") or payload.get("items") or []
            if not isinstance(raw_items, list):
                raise SmrApiError("list_task_summaries response missing tasks list")
            items = raw_items
        else:
            raise SmrApiError("list_task_summaries expected an object or list response")
        summaries = [dict(item) for item in items if isinstance(item, Mapping)]
        if run_id:
            summaries = [
                item for item in summaries if str(item.get("run_id") or run_id).strip() == run_id
            ]
        if objective_id:
            summaries = [
                item
                for item in summaries
                if str(item.get("objective_id") or objective_id).strip() == objective_id
            ]
        if kind:
            summaries = [item for item in summaries if str(item.get("kind") or "").strip() == kind]
        return [TaskSummary.from_wire(item) for item in summaries]

    def create_task(
        self,
        run_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        project_id: str,
        mode: str = "queue",
        body: str | None = None,
    ) -> RuntimeIntentReceipt:
        return self.submit_runtime_intent(
            run_id,
            RuntimeIntent.plan_tasks(tasks=[dict(payload)]),
            project_id=project_id,
            mode=mode,
            body=body,
        )

    def update_task(
        self,
        run_id: str,
        task_id: str,
        payload: Mapping[str, Any] | dict[str, Any],
        *,
        project_id: str,
        mode: str = "queue",
        body: str | None = None,
    ) -> RuntimeIntentReceipt:
        task_payload = {"task_id": task_id, **dict(payload)}
        return self.submit_runtime_intent(
            run_id,
            RuntimeIntent.plan_tasks(tasks=[task_payload]),
            project_id=project_id,
            mode=mode,
            body=body,
        )

    def cancel_task(
        self,
        run_id: str,
        task_id: str,
        *,
        project_id: str,
        reason: str | None = None,
        mode: str = "queue",
        body: str | None = None,
    ) -> RuntimeIntentReceipt:
        return self.submit_runtime_intent(
            run_id,
            RuntimeIntent.set_task_state(
                task_id=task_id,
                state="stopped",
                reason=reason,
            ),
            project_id=project_id,
            mode=mode,
            body=body,
        )

    def reassign_task(
        self,
        run_id: str,
        task_id: str,
        *,
        project_id: str,
        assignee: str,
        mode: str = "queue",
        body: str | None = None,
    ) -> RuntimeIntentReceipt:
        return self.update_task(
            run_id,
            task_id,
            {"assignee": assignee},
            project_id=project_id,
            mode=mode,
            body=body,
        )

    def _run_lifecycle_control(
        self,
        *,
        method: str,
        path: str,
        label: str,
    ) -> dict[str, Any]:
        """POST a lifecycle control and translate 409 bodies to typed errors.

        The backend contract for pause/resume/stop returns HTTP 409 with a
        ``detail`` mapping of
        ``{error_code, message, retryable, current_state, run_id}`` when
        the transition is rejected. We surface that as
        :class:`ManagedResearchRunControlError` so callers can discriminate
        auth / config / transient failure modes without re-parsing strings.
        Any other error status is left to the transport's existing mapping.
        """

        try:
            return _coerce_dict(
                self._request_json(method, path),
                label=label,
            )
        except SmrApiError as exc:
            if exc.status_code != 409:
                raise
            response_text = exc.response_text
            if response_text is None or not response_text.strip():
                raise ValueError(
                    f"{label}: HTTP 409 but response body was empty; "
                    "expected detail mapping with error_code/message/retryable/current_state/run_id"
                ) from exc
            try:
                payload = _json.loads(response_text)
            except ValueError as parse_exc:
                raise ValueError(
                    f"{label}: HTTP 409 body was not valid JSON: {response_text!r}"
                ) from parse_exc
            raise ManagedResearchRunControlError.from_response(
                payload=payload,
                status_code=exc.status_code,
                response_text=response_text,
            ) from exc

    def stop_run(self, run_id: str, *, project_id: str | None = None) -> dict[str, Any]:
        if project_id:
            return self._run_lifecycle_control(
                method="POST",
                path=f"/smr/projects/{project_id}/runs/{run_id}/stop",
                label="stop_project_run",
            )
        return self._run_lifecycle_control(
            method="POST",
            path=f"/smr/runs/{run_id}/stop",
            label="stop_run",
        )

    def pause_run(self, run_id: str, *, project_id: str | None = None) -> dict[str, Any]:
        if project_id:
            return self._run_lifecycle_control(
                method="POST",
                path=f"/smr/projects/{project_id}/runs/{run_id}/pause",
                label="pause_project_run",
            )
        return self._run_lifecycle_control(
            method="POST",
            path=f"/smr/runs/{run_id}/pause",
            label="pause_run",
        )

    def resume_run(self, run_id: str, *, project_id: str | None = None) -> dict[str, Any]:
        if project_id:
            return self._run_lifecycle_control(
                method="POST",
                path=f"/smr/projects/{project_id}/runs/{run_id}/resume",
                label="resume_project_run",
            )
        return self._run_lifecycle_control(
            method="POST",
            path=f"/smr/runs/{run_id}/resume",
            label="resume_run",
        )

    def control_project_run_actor(
        self,
        project_id: str,
        run_id: str,
        actor_id: str,
        *,
        action: str,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        body = {
            "action": action,
            "reason": reason,
            "idempotency_key": idempotency_key,
        }
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/actors/{actor_id}/control",
                json_body={key: value for key, value in body.items() if value is not None},
            ),
            label="control_project_run_actor",
        )

    def list_run_questions(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        status_filter: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        params = build_query_params(status_filter=status_filter, limit=limit, cursor=cursor)
        if project_id:
            scoped = self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/questions",
                params=params,
                allow_not_found=True,
            )
            if scoped is not None:
                return _coerce_dict_list(scoped, label="list_project_run_questions")
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/runs/{run_id}/questions", params=params),
            label="list_run_questions",
        )

    def respond_to_run_question(
        self,
        run_id: str,
        question_id: str,
        *,
        response_text: str,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "response_text": _require_non_empty_string(response_text, field_name="response_text")
        }
        if project_id:
            scoped = self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/questions/{question_id}/respond",
                json_body=payload,
                allow_not_found=True,
            )
            if scoped is not None:
                return _coerce_dict(scoped, label="respond_project_run_question")
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/runs/{run_id}/questions/{question_id}/respond",
                json_body=payload,
            ),
            label="respond_run_question",
        )

    def list_run_approvals(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        status_filter: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        params = build_query_params(status_filter=status_filter, limit=limit, cursor=cursor)
        if project_id:
            scoped = self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/approvals",
                params=params,
                allow_not_found=True,
            )
            if scoped is not None:
                return _coerce_dict_list(scoped, label="list_project_run_approvals")
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/runs/{run_id}/approvals", params=params),
            label="list_run_approvals",
        )

    def approve_run_approval(
        self,
        run_id: str,
        approval_id: str,
        *,
        project_id: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        payload = build_query_params(comment=comment) or {}
        if project_id:
            scoped = self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/approvals/{approval_id}/approve",
                json_body=payload,
                allow_not_found=True,
            )
            if scoped is not None:
                return _coerce_dict(scoped, label="approve_project_run_approval")
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/runs/{run_id}/approvals/{approval_id}/approve",
                json_body=payload,
            ),
            label="approve_run_approval",
        )

    def deny_run_approval(
        self,
        run_id: str,
        approval_id: str,
        *,
        project_id: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        payload = build_query_params(comment=comment) or {}
        if project_id:
            scoped = self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/approvals/{approval_id}/deny",
                json_body=payload,
                allow_not_found=True,
            )
            if scoped is not None:
                return _coerce_dict(scoped, label="deny_project_run_approval")
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/runs/{run_id}/approvals/{approval_id}/deny",
                json_body=payload,
            ),
            label="deny_run_approval",
        )

    def _run_checkpoint_path(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        checkpoint_id: str | None = None,
    ) -> str:
        if project_id:
            base = f"/smr/projects/{project_id}/runs/{run_id}/checkpoints"
        else:
            base = f"/smr/runs/{run_id}/checkpoints"
        if checkpoint_id and checkpoint_id.strip():
            return f"{base}/{checkpoint_id.strip()}"
        return base

    def request_run_checkpoint(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        checkpoint_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        payload = build_query_params(checkpoint_id=checkpoint_id, reason=reason)
        path = self._run_checkpoint_path(run_id, project_id=project_id)
        label = (
            "request_project_run_checkpoint" if project_id is not None else "request_run_checkpoint"
        )
        return _coerce_dict(
            self._request_json("POST", path, json_body=payload or {}),
            label=label,
        )

    def get_run_checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
        *,
        project_id: str | None = None,
        allow_not_found: bool = False,
    ) -> Checkpoint | None:
        path = self._run_checkpoint_path(
            run_id,
            project_id=project_id,
            checkpoint_id=checkpoint_id,
        )
        label = "get_project_run_checkpoint" if project_id is not None else "get_run_checkpoint"
        payload = self._request_json("GET", path, allow_not_found=allow_not_found)
        if payload is None:
            return None
        return Checkpoint.from_wire(_coerce_dict(payload, label=label))

    def wait_for_run_checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
        *,
        project_id: str | None = None,
        timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 1.0,
    ) -> Checkpoint:
        checkpoint_id_text = _require_non_empty_string(
            checkpoint_id,
            field_name="checkpoint_id",
        )
        timeout = max(0.1, float(timeout_seconds))
        poll_interval = max(0.1, float(poll_interval_seconds))
        deadline = time.monotonic() + timeout
        last_state: str | None = None
        while True:
            checkpoint = self.get_run_checkpoint(
                run_id,
                checkpoint_id_text,
                project_id=project_id,
                allow_not_found=True,
            )
            if checkpoint is not None:
                state = str(checkpoint.state).strip().lower()
                if state in {"ready", "failed", "pruned"}:
                    return checkpoint
                last_state = checkpoint.state
            now = time.monotonic()
            if now >= deadline:
                break
            time.sleep(min(poll_interval, deadline - now))
        last_state_suffix = f" (last_state={last_state})" if last_state else ""
        raise SmrApiError(
            f"Timed out waiting for checkpoint '{checkpoint_id_text}' to materialize{last_state_suffix}"
        )

    def create_run_checkpoint(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        checkpoint_id: str | None = None,
        reason: str | None = None,
        timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 1.0,
    ) -> Checkpoint:
        control_ack = self.request_run_checkpoint(
            run_id,
            project_id=project_id,
            checkpoint_id=checkpoint_id,
            reason=reason,
        )
        resolved_checkpoint_id = _require_non_empty_string(
            str(control_ack.get("checkpoint_id") or checkpoint_id or ""),
            field_name="checkpoint_id",
        )
        return self.wait_for_run_checkpoint(
            run_id,
            resolved_checkpoint_id,
            project_id=project_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    def list_run_checkpoints(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> list[Checkpoint]:
        path = self._run_checkpoint_path(run_id, project_id=project_id)
        label = "list_project_run_checkpoints" if project_id is not None else "list_run_checkpoints"
        return [
            Checkpoint.from_wire(item)
            for item in _coerce_dict_list(
                self._request_json("GET", path),
                label=label,
            )
        ]

    def restore_run_checkpoint(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        checkpoint_id: str | None = None,
        checkpoint_record_id: str | None = None,
        checkpoint_uri: str | None = None,
        reason: str | None = None,
        mode: str = "in_place",
    ) -> dict[str, Any]:
        payload = build_query_params(
            checkpoint_id=checkpoint_id,
            checkpoint_record_id=checkpoint_record_id,
            checkpoint_uri=checkpoint_uri,
            reason=reason,
            mode=mode,
        )
        if project_id:
            return _coerce_dict(
                self._request_json(
                    "POST",
                    f"/smr/projects/{project_id}/runs/{run_id}/restore",
                    json_body=payload or {},
                ),
                label="restore_project_run_checkpoint",
            )
        return _coerce_dict(
            self._request_json("POST", f"/smr/runs/{run_id}/restore", json_body=payload or {}),
            label="restore_run_checkpoint",
        )

    def get_run_logical_timeline(self, project_id: str, run_id: str) -> SmrLogicalTimeline:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/timeline",
            ),
            label="get_run_logical_timeline",
        )
        return SmrLogicalTimeline.from_wire(payload)

    def get_project_run_event_log(
        self,
        project_id: str,
        run_id: str,
        *,
        sources: list[str] | None = None,
        event_kinds: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int | None = None,
    ) -> SmrRunEventLog:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/event-log",
                params=build_query_params(
                    sources=sources,
                    event_kinds=event_kinds,
                    statuses=statuses,
                    limit=limit,
                ),
            ),
            label="get_project_run_event_log",
        )
        return SmrRunEventLog.from_wire(payload)

    def get_run_authority_readouts(
        self,
        run_id: str,
        *,
        include_runtime_authority: bool = False,
    ) -> SmrAuthorityReadouts:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/runs/{run_id}/authority-readouts",
                params=build_query_params(include_runtime_authority=include_runtime_authority),
            ),
            label="get_run_authority_readouts",
        )
        return SmrAuthorityReadouts.from_wire(payload)

    def get_project_run_authority_readouts(
        self,
        project_id: str,
        run_id: str,
        *,
        include_runtime_authority: bool = False,
    ) -> SmrAuthorityReadouts:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/authority-readouts",
                params=build_query_params(include_runtime_authority=include_runtime_authority),
            ),
            label="get_project_run_authority_readouts",
        )
        return SmrAuthorityReadouts.from_wire(payload)

    def get_project_run_operator_evidence(
        self,
        project_id: str,
        run_id: str,
        *,
        runtime_timeline_limit: int | None = None,
        logical_timeline_limit: int | None = None,
        transcript_limit: int | None = None,
        reconciliation_limit: int | None = None,
    ) -> SmrRunOperatorEvidence:
        """Return the canonical typed operator-evidence owner response."""

        return SmrRunOperatorEvidence.from_wire(
            self.get_project_run_operator_evidence_raw(
                project_id,
                run_id,
                runtime_timeline_limit=runtime_timeline_limit,
                logical_timeline_limit=logical_timeline_limit,
                transcript_limit=transcript_limit,
                reconciliation_limit=reconciliation_limit,
            )
        )

    def get_project_run_operator_evidence_raw(
        self,
        project_id: str,
        run_id: str,
        *,
        runtime_timeline_limit: int | None = None,
        logical_timeline_limit: int | None = None,
        transcript_limit: int | None = None,
        reconciliation_limit: int | None = None,
    ) -> dict[str, Any]:
        """Legacy serialized operator evidence; canonical callers use the DTO."""

        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/operator-evidence",
                params=build_query_params(
                    runtime_timeline_limit=runtime_timeline_limit,
                    logical_timeline_limit=logical_timeline_limit,
                    transcript_limit=transcript_limit,
                    reconciliation_limit=reconciliation_limit,
                ),
            ),
            label="get_project_run_operator_evidence",
        )

    def get_project_run_operator_evidence_typed(
        self,
        project_id: str,
        run_id: str,
        *,
        runtime_timeline_limit: int | None = None,
        logical_timeline_limit: int | None = None,
        transcript_limit: int | None = None,
        reconciliation_limit: int | None = None,
    ) -> SmrRunOperatorEvidence:
        """Compatibility alias for the canonical typed operator-evidence read."""

        return self.get_project_run_operator_evidence(
            project_id,
            run_id,
            runtime_timeline_limit=runtime_timeline_limit,
            logical_timeline_limit=logical_timeline_limit,
            transcript_limit=transcript_limit,
            reconciliation_limit=reconciliation_limit,
        )

    def get_run_traces(self, run_id: str) -> SmrRunTraces:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/runs/{run_id}/traces",
            ),
            label="get_run_traces",
        )
        return SmrRunTraces.from_wire(payload)

    def get_project_run_traces(self, project_id: str, run_id: str) -> SmrRunTraces:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/traces",
            ),
            label="get_project_run_traces",
        )
        return SmrRunTraces.from_wire(payload)

    def get_project_run_actor_trace(
        self,
        project_id: str,
        run_id: str,
        actor_key: str,
        *,
        cursor: str | None = None,
        live_cursor: str | None = None,
        limit: int | None = None,
        include_live: bool | None = None,
        include_traces: bool | None = None,
    ) -> dict[str, Any]:
        params = build_query_params(
            cursor=cursor,
            live_cursor=live_cursor,
            limit=limit,
            include_live=include_live,
            include_traces=include_traces,
        )
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/actors/{actor_key}/trace",
                params=params,
            ),
            label="get_project_run_actor_trace",
        )

    def get_project_run_actor_trace_index(
        self,
        project_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/actors/trace-index",
            ),
            label="get_project_run_actor_trace_index",
        )

    def get_project_run_actors(
        self,
        project_id: str,
        run_id: str,
    ) -> list[dict[str, Any]]:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/actors",
            ),
            label="get_project_run_actors",
        )
        actors = payload.get("actors")
        if not isinstance(actors, list):
            raise SmrApiError("get_project_run_actors response missing actors list")
        return [dict(item) for item in actors if isinstance(item, Mapping)]

    def get_project_run_actor_raw_traces(
        self,
        project_id: str,
        run_id: str,
        actor_key: str,
    ) -> list[dict[str, Any]]:
        payload = self._request_json(
            "GET",
            f"/smr/projects/{project_id}/runs/{run_id}/actors/{actor_key}/traces",
        )
        if not isinstance(payload, list):
            raise ValueError("get_project_run_actor_raw_traces expected a list response")
        return [dict(item) for item in payload if isinstance(item, Mapping)]

    def get_project_run_raw_trace_events(
        self,
        project_id: str,
        run_id: str,
        artifact_id: str,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        redaction_mode: str | None = None,
        reconstruct: bool | None = None,
        category: str | list[str] | None = None,
        method: str | list[str] | None = None,
    ) -> dict[str, Any]:
        params = build_query_params(
            cursor=cursor,
            limit=limit,
            redaction_mode=redaction_mode,
            reconstruct=reconstruct,
            category=category,
            method=method,
        )
        return _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/traces/{artifact_id}/events",
                params=params,
            ),
            label="get_project_run_raw_trace_events",
        )

    def create_project_run_raw_trace_download_url(
        self,
        project_id: str,
        run_id: str,
        artifact_id: str,
        *,
        expires_in: int | None = None,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/traces/{artifact_id}/download-url",
                params=build_query_params(expires_in=expires_in),
            ),
            label="create_project_run_raw_trace_download_url",
        )

    def download_project_run_raw_trace(
        self,
        project_id: str,
        run_id: str,
        artifact_id: str,
        destination: str | os.PathLike[str],
        *,
        expires_in: int | None = None,
    ) -> dict[str, Any]:
        url_payload = self.create_project_run_raw_trace_download_url(
            project_id,
            run_id,
            artifact_id,
            expires_in=expires_in,
        )
        url = str(url_payload.get("url") or "").strip()
        if not url:
            raise ValueError("download URL response did not include url")
        response = httpx.get(url, timeout=self.timeout_seconds, follow_redirects=True)
        if response.is_error:
            _raise_for_error_response(response)
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(response.content)
        result = dict(url_payload)
        result["destination"] = str(destination_path)
        result["size_bytes"] = len(response.content)
        return result

    def get_run_actor_usage(self, run_id: str) -> SmrRunActorUsage:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/runs/{run_id}/actors/usage",
            ),
            label="get_run_actor_usage",
        )
        return SmrRunActorUsage.from_wire(payload)

    def get_project_run_actor_usage(
        self,
        project_id: str,
        run_id: str,
    ) -> SmrRunActorUsage:
        payload = _coerce_dict(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/actors/usage",
            ),
            label="get_project_run_actor_usage",
        )
        return SmrRunActorUsage.from_wire(payload)

    def list_run_participants(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> SmrRunParticipants:
        path = (
            f"/smr/projects/{project_id}/runs/{run_id}/participants"
            if project_id
            else f"/smr/runs/{run_id}/participants"
        )
        payload = _coerce_dict(
            self._request_json("GET", path),
            label="list_run_participants",
        )
        return SmrRunParticipants.from_wire(payload)

    def get_run_artifact_progress(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> SmrRunArtifactProgress:
        path = (
            f"/smr/projects/{project_id}/runs/{run_id}/artifact-progress"
            if project_id
            else f"/smr/runs/{run_id}/artifact-progress"
        )
        payload = _coerce_dict(
            self._request_json("GET", path),
            label="get_run_artifact_progress",
        )
        return SmrRunArtifactProgress.from_wire(payload)

    def list_run_actor_logs(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        actor_id: str | None = None,
        turn_id: str | None = None,
        kind: str | None = None,
        since: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> SmrRunActorLogs:
        path = (
            f"/smr/projects/{project_id}/runs/{run_id}/actor-logs"
            if project_id
            else f"/smr/runs/{run_id}/actor-logs"
        )
        params = build_query_params(
            actor_id=actor_id,
            turn_id=turn_id,
            kind=kind,
            since=since,
            cursor=cursor,
            limit=limit,
        )
        payload = _coerce_dict(
            self._request_json("GET", path, params=params),
            label="list_run_actor_logs",
        )
        return SmrRunActorLogs.from_wire(payload)

    def get_run_cost_summary(self, run_id: str) -> SmrRunCostSummary:
        payload = _coerce_dict(
            self.run_cost.summary(run_id),
            label="get_run_cost_summary",
        )
        return SmrRunCostSummary.from_wire(payload)

    def branch_run_from_checkpoint(
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
        request = _coerce_branch_request(
            checkpoint_id=checkpoint_id,
            checkpoint_record_id=checkpoint_record_id,
            checkpoint_uri=checkpoint_uri,
            mode=mode,
            message=message,
            reason=reason,
            title=title,
            source_node_id=source_node_id,
        )
        if project_id is not None and run_id is None:
            raise ValueError("run_id is required when project_id is provided")
        if project_id and run_id:
            path = f"/smr/projects/{project_id}/runs/{run_id}/branches"
            label = "branch_project_run_from_checkpoint"
        elif run_id:
            path = f"/smr/runs/{run_id}/branches"
            label = "branch_run_from_checkpoint"
        else:
            path = "/smr/checkpoints/branches"
            label = "branch_checkpoint_reference"
        payload = _coerce_dict(
            self._request_json("POST", path, json_body=request.to_wire()),
            label=label,
        )
        return SmrRunBranchResponse.from_wire(payload)

    def list_runtime_messages(
        self,
        run_id: str,
        *,
        status: str | None = None,
        viewer_role: str | None = None,
        viewer_target: str | Iterable[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if status and status.strip():
            params["status"] = status.strip()
        if viewer_role and viewer_role.strip():
            params["viewer_role"] = viewer_role.strip()
        if limit is not None:
            params["limit"] = int(limit)
        if isinstance(viewer_target, str) and viewer_target.strip():
            params["viewer_target"] = [viewer_target.strip()]
        elif viewer_target is not None:
            cleaned_targets = [str(item).strip() for item in viewer_target if str(item).strip()]
            if cleaned_targets:
                params["viewer_target"] = cleaned_targets
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/runs/{run_id}/runtime/messages",
                params=params or None,
            ),
            label="list_runtime_messages",
        )

    def _list_runtime_messages(
        self,
        run_id: str,
        *,
        status: str | None = None,
        viewer_role: str | None = None,
        viewer_target: str | Iterable[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.list_runtime_messages(
            run_id,
            status=status,
            viewer_role=viewer_role,
            viewer_target=viewer_target,
            limit=limit,
        )

    def list_project_run_runtime_messages(
        self,
        project_id: str,
        run_id: str,
        *,
        status: str | None = None,
        viewer_role: str | None = None,
        viewer_target: str | Iterable[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if status and status.strip():
            params["status"] = status.strip()
        if viewer_role and viewer_role.strip():
            params["viewer_role"] = viewer_role.strip()
        if limit is not None:
            params["limit"] = int(limit)
        if isinstance(viewer_target, str) and viewer_target.strip():
            params["viewer_target"] = [viewer_target.strip()]
        elif viewer_target is not None:
            cleaned_targets = [str(item).strip() for item in viewer_target if str(item).strip()]
            if cleaned_targets:
                params["viewer_target"] = cleaned_targets
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/runtime/messages",
                params=params or None,
            ),
            label="list_project_run_runtime_messages",
        )

    def submit_runtime_intent(
        self,
        run_id: str,
        intent: RuntimeIntent | Mapping[str, Any] | dict[str, Any],
        *,
        project_id: str | None = None,
        mode: str = "queue",
        body: str | None = None,
        causation_id: str | None = None,
    ) -> RuntimeIntentReceipt:
        if isinstance(intent, RuntimeIntent):
            intent_payload = intent.to_wire()
        elif isinstance(intent, Mapping):
            intent_payload = dict(intent)
        else:
            raise ValueError("intent must be a RuntimeIntent or mapping")
        json_body: dict[str, Any] = {
            "intent": intent_payload,
            "mode": str(mode or "queue").strip().lower(),
        }
        if body and body.strip():
            json_body["body"] = body.strip()
        if causation_id and causation_id.strip():
            json_body["causation_id"] = causation_id.strip()
        path = (
            f"/smr/projects/{project_id}/runs/{run_id}/runtime/intents"
            if project_id
            else f"/smr/runs/{run_id}/runtime/intents"
        )
        return RuntimeIntentReceipt.from_wire(
            _coerce_dict(
                self._request_json("POST", path, json_body=json_body),
                label="submit_runtime_intent",
            )
        )

    def list_runtime_intents(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[RuntimeIntentView]:
        params: dict[str, Any] = {}
        if status and status.strip():
            params["status"] = status.strip()
        if limit is not None:
            params["limit"] = int(limit)
        path = (
            f"/smr/projects/{project_id}/runs/{run_id}/runtime/intents"
            if project_id
            else f"/smr/runs/{run_id}/runtime/intents"
        )
        return [
            RuntimeIntentView.from_wire(item)
            for item in _coerce_dict_list(
                self._request_json("GET", path, params=params or None),
                label="list_runtime_intents",
            )
        ]

    def get_runtime_intent(
        self,
        run_id: str,
        runtime_intent_id: str,
        *,
        project_id: str | None = None,
    ) -> RuntimeIntentView:
        path = (
            f"/smr/projects/{project_id}/runs/{run_id}/runtime/intents/{runtime_intent_id}"
            if project_id
            else f"/smr/runs/{run_id}/runtime/intents/{runtime_intent_id}"
        )
        return RuntimeIntentView.from_wire(
            _coerce_dict(
                self._request_json("GET", path),
                label="get_runtime_intent",
            )
        )

    def enqueue_runtime_message(
        self,
        run_id: str,
        *,
        topic: str | None = None,
        causation_id: str | None = None,
        mode: str | None = None,
        spawn_policy: str | None = None,
        sender: str | None = None,
        target: str | None = None,
        participant_session_id: str | None = None,
        action: str | None = None,
        body: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        json_body: dict[str, Any] = {}
        for key, value in (
            ("topic", topic),
            ("causation_id", causation_id),
            ("mode", mode),
            ("spawn_policy", spawn_policy),
            ("sender", sender),
            ("target", target),
            ("participant_session_id", participant_session_id),
            ("action", action),
            ("body", body),
        ):
            if value and value.strip():
                json_body[key] = value.strip()
        normalized_payload = _optional_mapping(payload, field_name="payload")
        if normalized_payload:
            json_body["payload"] = normalized_payload
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/runs/{run_id}/runtime/messages",
                json_body=json_body,
            ),
            label="enqueue_runtime_message",
        )

    def publish_manderqueue_message(
        self,
        run_id: str,
        *,
        project_id: str,
        intent: str = "queue",
        audience: Mapping[str, Any] | dict[str, Any] | None = None,
        body: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
        message_kind: str = "runtime_message",
        thread_id: str | None = None,
        parent_message_id: str | None = None,
        fallback_policy: str = "block",
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
    ) -> dict[str, Any]:
        json_body: dict[str, Any] = {
            "intent": str(intent or "queue").strip(),
            "audience": dict(audience or {"kind": "run"}),
            "message_kind": str(message_kind or "runtime_message").strip(),
            "fallback_policy": str(fallback_policy or "block").strip(),
        }
        for key, value in (
            ("body", body),
            ("thread_id", thread_id),
            ("parent_message_id", parent_message_id),
            ("idempotency_key", idempotency_key),
            ("correlation_id", correlation_id),
            ("causation_id", causation_id),
        ):
            if value and str(value).strip():
                json_body[key] = str(value).strip()
        normalized_payload = _optional_mapping(payload, field_name="payload")
        if normalized_payload:
            json_body["payload"] = normalized_payload
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/manderqueue/messages",
                json_body=json_body,
            ),
            label="publish_manderqueue_message",
        )

    def send_message(
        self,
        run_id: str,
        *,
        project_id: str,
        intent: str = "queue",
        audience: Mapping[str, Any] | dict[str, Any] | None = None,
        body: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
        message_kind: str = "runtime_message",
        thread_id: str | None = None,
        parent_message_id: str | None = None,
        fallback_policy: str = "block",
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
    ) -> dict[str, Any]:
        """Send a product-level message queue message for a run."""

        return self.publish_manderqueue_message(
            run_id,
            project_id=project_id,
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

    def publish_message_queue_message(
        self,
        run_id: str,
        *,
        project_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Publish a product-level message queue message for a run."""

        return self.publish_manderqueue_message(run_id, project_id=project_id, **kwargs)

    def list_manderqueue_threads(
        self,
        run_id: str,
        *,
        project_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params = {"limit": int(limit)} if limit is not None else None
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/manderqueue/threads",
                params=params,
            ),
            label="list_manderqueue_threads",
        )

    def list_message_queue_threads(
        self,
        run_id: str,
        *,
        project_id: str,
        limit: int | None = None,
    ) -> list[MessageQueueThread]:
        return [
            MessageQueueThread.from_wire(item)
            for item in self.list_manderqueue_threads(
                run_id,
                project_id=project_id,
                limit=limit,
            )
        ]

    def list_manderqueue_messages(
        self,
        run_id: str,
        *,
        project_id: str,
        thread_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if thread_id and thread_id.strip():
            params["thread_id"] = thread_id.strip()
        if limit is not None:
            params["limit"] = int(limit)
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/manderqueue/messages",
                params=params or None,
            ),
            label="list_manderqueue_messages",
        )

    def list_message_queue_messages(
        self,
        run_id: str,
        *,
        project_id: str,
        thread_id: str | None = None,
        limit: int | None = None,
    ) -> list[MessageQueueMessage]:
        return [
            MessageQueueMessage.from_wire(item)
            for item in self.list_manderqueue_messages(
                run_id,
                project_id=project_id,
                thread_id=thread_id,
                limit=limit,
            )
        ]

    def list_messages(
        self,
        run_id: str,
        *,
        project_id: str,
        thread_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List product-level message queue messages for a run."""

        return self.list_manderqueue_messages(
            run_id,
            project_id=project_id,
            thread_id=thread_id,
            limit=limit,
        )

    def list_manderqueue_interactions(
        self,
        run_id: str,
        *,
        project_id: str,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if status and status.strip():
            params["status"] = status.strip()
        if limit is not None:
            params["limit"] = int(limit)
        return _coerce_dict_list(
            self._request_json(
                "GET",
                f"/smr/projects/{project_id}/runs/{run_id}/manderqueue/interactions",
                params=params or None,
            ),
            label="list_manderqueue_interactions",
        )

    def list_message_queue_interactions(
        self,
        run_id: str,
        *,
        project_id: str,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[MessageQueueInteraction]:
        return [
            MessageQueueInteraction.from_wire(item)
            for item in self.list_manderqueue_interactions(
                run_id,
                project_id=project_id,
                status=status,
                limit=limit,
            )
        ]

    def respond_to_manderqueue_interaction(
        self,
        run_id: str,
        interaction_id: str,
        *,
        project_id: str,
        body: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        json_body: dict[str, Any] = {}
        if body and body.strip():
            json_body["body"] = body.strip()
        normalized_payload = _optional_mapping(payload, field_name="payload")
        if normalized_payload:
            json_body["payload"] = normalized_payload
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/manderqueue/interactions/{interaction_id}/responses",
                json_body=json_body,
            ),
            label="respond_to_manderqueue_interaction",
        )

    def respond_to_message_queue_interaction(
        self,
        run_id: str,
        interaction_id: str,
        *,
        project_id: str,
        body: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.respond_to_manderqueue_interaction(
            run_id,
            interaction_id,
            project_id=project_id,
            body=body,
            payload=payload,
        )

    def edit_manderqueue_message(
        self,
        run_id: str,
        message_id: str,
        *,
        project_id: str,
        body: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        json_body: dict[str, Any] = {}
        if body and body.strip():
            json_body["body"] = body.strip()
        normalized_payload = _optional_mapping(payload, field_name="payload")
        if normalized_payload:
            json_body["payload"] = normalized_payload
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/manderqueue/messages/{message_id}/edit",
                json_body=json_body,
            ),
            label="edit_manderqueue_message",
        )

    def edit_message_queue_message(
        self,
        run_id: str,
        message_id: str,
        *,
        project_id: str,
        body: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.edit_manderqueue_message(
            run_id,
            message_id,
            project_id=project_id,
            body=body,
            payload=payload,
        )

    def edit_message(
        self,
        run_id: str,
        message_id: str,
        *,
        project_id: str,
        body: str | None = None,
        payload: Mapping[str, Any] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Edit a product-level message queue message for a run."""

        return self.edit_manderqueue_message(
            run_id,
            message_id,
            project_id=project_id,
            body=body,
            payload=payload,
        )

    def retract_manderqueue_message(
        self,
        run_id: str,
        message_id: str,
        *,
        project_id: str,
    ) -> dict[str, Any]:
        return _coerce_dict(
            self._request_json(
                "POST",
                f"/smr/projects/{project_id}/runs/{run_id}/manderqueue/messages/{message_id}/retract",
                json_body={},
            ),
            label="retract_manderqueue_message",
        )

    def retract_message_queue_message(
        self,
        run_id: str,
        message_id: str,
        *,
        project_id: str,
    ) -> dict[str, Any]:
        return self.retract_manderqueue_message(
            run_id,
            message_id,
            project_id=project_id,
        )

    def retract_message(
        self,
        run_id: str,
        message_id: str,
        *,
        project_id: str,
    ) -> dict[str, Any]:
        """Retract a product-level message queue message for a run."""

        return self.retract_manderqueue_message(
            run_id,
            message_id,
            project_id=project_id,
        )

    def _list_run_log_archives(
        self,
        project_id: str,
        run_id: str,
    ) -> list[dict[str, Any]]:
        return _coerce_dict_list(
            self._request_json("GET", f"/smr/projects/{project_id}/runs/{run_id}/logs/archives"),
            label="list_run_log_archives",
        )

