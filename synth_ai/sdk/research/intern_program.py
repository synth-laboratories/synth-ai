"""``client.intern.program`` -- the Effort-primary Intern research program.

Effort is the Async organizer. Objectives, milestones, tasks, progress claims,
and links hang beneath one Effort, and the Effort is a path segment on every
create call, so a row cannot be written without its binding.

The Intern is an MCP client of the swarm plane, not its owner. Nothing in this
namespace plans a swarm task graph or writes swarm run tasks. Kicking a run off
and polling it stays on the existing Factory / swarm surfaces
(``research.factories``, ``research.swarms``, ``research.runs``); the Intern
side of the loop is :meth:`InternProgramAPI.create_progress_claim`, which folds
a finished run's result back into the objective that motivated it.

# See: backend app/api/v1/managed_research/intern_program.py (WP6 SS4)
"""

from __future__ import annotations

from typing import cast

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.sdk.research.contracts._wire import array_value
from synth_ai.sdk.research.contracts.intern_program import (
    InternEffortBoardResponse,
    InternEffortDetailResponse,
    InternEffortTaskCreateRequest,
    InternEffortTaskPatchRequest,
    InternEffortTaskResponse,
    InternEffortTaskState,
    InternMemoryHitResponse,
    InternMemorySearchResponse,
    InternMilestoneCreateRequest,
    InternMilestoneResponse,
    InternMilestoneState,
    InternMilestoneTransitionRequest,
    InternObjectiveCreateRequest,
    InternObjectiveKind,
    InternObjectiveLinkCreateRequest,
    InternObjectiveLinkResponse,
    InternObjectivePatchRequest,
    InternObjectiveResponse,
    InternObjectiveStatus,
    InternProgressClaimCreateRequest,
    InternProgressClaimResponse,
)
from synth_ai.sdk.research.operations import research_operation

_EFFORTS = "/smr/research-intern/efforts"
_OBJECTIVES = "/smr/research-intern/objectives"
_MILESTONES = "/smr/research-intern/milestones"
_TASKS = "/smr/research-intern/tasks"
_MEMORY = "/smr/research-intern/memory"

#: Server-side caps. Asking for more is not an error; the backend clamps. The
#: SDK sends what the caller asked for so a raised backend cap needs no release.
_EFFORT_LIMIT_DEFAULT = 25
_OBJECTIVE_LIMIT_DEFAULT = 50
_MILESTONE_LIMIT_DEFAULT = 50
_TASK_LIMIT_DEFAULT = 100
_CLAIM_LIMIT_DEFAULT = 50
_LINK_LIMIT_DEFAULT = 50
_MEMORY_LIMIT_DEFAULT = 20


def _request(
    operation_id: str,
    path: str,
    *,
    query: JsonObject | None = None,
    body: JsonObject | None = None,
) -> HttpRequest:
    return HttpRequest(
        research_operation(operation_id),
        path,
        query=query or {},
        body=body,
    )


def _objectives(value: JsonValue) -> tuple[InternObjectiveResponse, ...]:
    return tuple(
        InternObjectiveResponse.from_wire(item)
        for item in array_value(value, operation_id="list_intern_effort_objectives")
    )


def _milestones(value: JsonValue) -> tuple[InternMilestoneResponse, ...]:
    return tuple(
        InternMilestoneResponse.from_wire(item)
        for item in array_value(value, operation_id="list_intern_effort_milestones")
    )


def _tasks(value: JsonValue) -> tuple[InternEffortTaskResponse, ...]:
    return tuple(
        InternEffortTaskResponse.from_wire(item)
        for item in array_value(value, operation_id="list_intern_effort_tasks")
    )


def _claims(value: JsonValue) -> tuple[InternProgressClaimResponse, ...]:
    return tuple(
        InternProgressClaimResponse.from_wire(item)
        for item in array_value(value, operation_id="list_intern_effort_progress_claims")
    )


def _links(value: JsonValue) -> tuple[InternObjectiveLinkResponse, ...]:
    return tuple(
        InternObjectiveLinkResponse.from_wire(item)
        for item in array_value(value, operation_id="list_intern_effort_objective_links")
    )


def _effort_board_query(
    *,
    factory_id: str | None,
    status: str | None,
    limit: int,
    cursor: str | None,
) -> JsonObject:
    query: JsonObject = {"limit": limit}
    if factory_id is not None:
        query["factory_id"] = factory_id
    if status is not None:
        query["status"] = status
    if cursor is not None:
        query["cursor"] = cursor
    return query


def _objective_query(status: InternObjectiveStatus | str | None, limit: int) -> JsonObject:
    query: JsonObject = {"limit": limit}
    if status is not None:
        query["status"] = str(status)
    return query


def _milestone_query(state: InternMilestoneState | str | None, limit: int) -> JsonObject:
    query: JsonObject = {"limit": limit}
    if state is not None:
        query["state"] = str(state)
    return query


def _task_query(
    state: InternEffortTaskState | str | None,
    milestone_id: str | None,
    limit: int,
) -> JsonObject:
    query: JsonObject = {"limit": limit}
    if state is not None:
        query["state"] = str(state)
    if milestone_id is not None:
        query["milestone_id"] = milestone_id
    return query


def _memory_query(query: str, kinds: tuple[str, ...] | None, limit: int) -> JsonObject:
    search: JsonObject = {"query": query, "limit": limit}
    if kinds:
        search["kinds"] = list(kinds)
    return search


class InternProgramAPI:
    """Effort board, Effort detail, and the Effort-bound Intern planner store."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    # -- Effort board -------------------------------------------------------

    def list_efforts(
        self,
        *,
        factory_id: str | None = None,
        status: str | None = None,
        limit: int = _EFFORT_LIMIT_DEFAULT,
        cursor: str | None = None,
    ) -> InternEffortBoardResponse:
        """List Efforts as board rows: the Effort plus its open-work counts.

        Args:
            factory_id: Restrict to one Factory's Efforts.
            status: Effort status filter.
            limit: Page size; the backend clamps it.
            cursor: Opaque continuation from a previous ``next_cursor``.

        Returns:
            ``InternEffortBoardResponse``. Board rows carry counts only --
            fetch :meth:`get_effort` for the selected Effort's bodies.
        """
        return InternEffortBoardResponse.from_wire(
            self._transport.execute(
                _request(
                    "list_intern_efforts",
                    _EFFORTS,
                    query=_effort_board_query(
                        factory_id=factory_id,
                        status=status,
                        limit=limit,
                        cursor=cursor,
                    ),
                )
            )
        )

    def get_effort(self, effort_id: str) -> InternEffortDetailResponse:
        """Fetch one Effort's rollup: progress, results, experiments, knowledge.

        Args:
            effort_id: Effort to roll up.

        Returns:
            ``InternEffortDetailResponse``. The ``runtime`` block is secondary
            ops chrome; the four rollups are the product surface.
        """
        return InternEffortDetailResponse.from_wire(
            self._transport.execute(_request("get_intern_effort_detail", f"{_EFFORTS}/{effort_id}"))
        )

    # -- Objectives ---------------------------------------------------------

    def list_objectives(
        self,
        effort_id: str,
        *,
        status: InternObjectiveStatus | str | None = None,
        limit: int = _OBJECTIVE_LIMIT_DEFAULT,
    ) -> tuple[InternObjectiveResponse, ...]:
        """List the Intern objectives opened under one Effort."""
        return _objectives(
            self._transport.execute(
                _request(
                    "list_intern_effort_objectives",
                    f"{_EFFORTS}/{effort_id}/objectives",
                    query=_objective_query(status, limit),
                )
            )
        )

    def create_objective(
        self,
        effort_id: str,
        request: InternObjectiveCreateRequest,
    ) -> InternObjectiveResponse:
        """Open one objective under an Effort.

        Args:
            effort_id: The owning Effort. This is the binding; the request body
                carries no ``effort_id`` of its own to contradict it.
            request: Typed objective specification.

        Returns:
            The created ``InternObjectiveResponse``.
        """
        return InternObjectiveResponse.from_wire(
            self._transport.execute(
                _request(
                    "create_intern_effort_objective",
                    f"{_EFFORTS}/{effort_id}/objectives",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    def get_objective(
        self,
        objective_id: str,
        *,
        objective_kind: InternObjectiveKind | str,
    ) -> InternObjectiveResponse:
        """Fetch one objective.

        Args:
            objective_id: Objective to fetch.
            objective_kind: Required -- open-ended questions and directed
                effort outcomes live in different tables, so the kind selects
                which one to read.
        """
        return InternObjectiveResponse.from_wire(
            self._transport.execute(
                _request(
                    "get_intern_objective",
                    f"{_OBJECTIVES}/{objective_id}",
                    query={"objective_kind": str(objective_kind)},
                )
            )
        )

    def update_objective(
        self,
        objective_id: str,
        request: InternObjectivePatchRequest,
    ) -> InternObjectiveResponse:
        """Revise one objective.

        ``request.objective_kind`` selects the backing table. The Effort binding
        is immutable and is not patchable.
        """
        return InternObjectiveResponse.from_wire(
            self._transport.execute(
                _request(
                    "patch_intern_objective",
                    f"{_OBJECTIVES}/{objective_id}",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    # -- Milestones ---------------------------------------------------------

    def list_milestones(
        self,
        effort_id: str,
        *,
        state: InternMilestoneState | str | None = None,
        limit: int = _MILESTONE_LIMIT_DEFAULT,
    ) -> tuple[InternMilestoneResponse, ...]:
        """List the milestones under one Effort."""
        return _milestones(
            self._transport.execute(
                _request(
                    "list_intern_effort_milestones",
                    f"{_EFFORTS}/{effort_id}/milestones",
                    query=_milestone_query(state, limit),
                )
            )
        )

    def create_milestone(
        self,
        effort_id: str,
        request: InternMilestoneCreateRequest,
    ) -> InternMilestoneResponse:
        """Break one objective into a subquestion or suboutcome milestone."""
        return InternMilestoneResponse.from_wire(
            self._transport.execute(
                _request(
                    "create_intern_effort_milestone",
                    f"{_EFFORTS}/{effort_id}/milestones",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    def transition_milestone(
        self,
        milestone_id: str,
        request: InternMilestoneTransitionRequest,
    ) -> InternMilestoneResponse:
        """Move a milestone to its next lifecycle state.

        Legality is decided by the shared milestone state machine, so an
        illegal move is refused rather than silently applied.
        """
        return InternMilestoneResponse.from_wire(
            self._transport.execute(
                _request(
                    "transition_intern_milestone",
                    f"{_MILESTONES}/{milestone_id}/transition",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    # -- Tasks --------------------------------------------------------------

    def list_tasks(
        self,
        effort_id: str,
        *,
        state: InternEffortTaskState | str | None = None,
        milestone_id: str | None = None,
        limit: int = _TASK_LIMIT_DEFAULT,
    ) -> tuple[InternEffortTaskResponse, ...]:
        """List the Intern planner checklist for one Effort.

        These are the Intern's own tasks. They are not swarm run tasks and have
        no relationship to a run's task graph.
        """
        return _tasks(
            self._transport.execute(
                _request(
                    "list_intern_effort_tasks",
                    f"{_EFFORTS}/{effort_id}/tasks",
                    query=_task_query(state, milestone_id, limit),
                )
            )
        )

    def create_task(
        self,
        effort_id: str,
        request: InternEffortTaskCreateRequest,
    ) -> InternEffortTaskResponse:
        """Add one task to the Intern planner checklist for an Effort."""
        return InternEffortTaskResponse.from_wire(
            self._transport.execute(
                _request(
                    "create_intern_effort_task",
                    f"{_EFFORTS}/{effort_id}/tasks",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    def update_task(
        self,
        intern_task_id: str,
        request: InternEffortTaskPatchRequest,
    ) -> InternEffortTaskResponse:
        """Revise one planner task; ``state`` moves validate against the state machine."""
        return InternEffortTaskResponse.from_wire(
            self._transport.execute(
                _request(
                    "patch_intern_effort_task",
                    f"{_TASKS}/{intern_task_id}",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    # -- Progress claims and links -----------------------------------------

    def list_progress_claims(
        self,
        effort_id: str,
        *,
        limit: int = _CLAIM_LIMIT_DEFAULT,
    ) -> tuple[InternProgressClaimResponse, ...]:
        """List the append-only progress claims folded into one Effort."""
        return _claims(
            self._transport.execute(
                _request(
                    "list_intern_effort_progress_claims",
                    f"{_EFFORTS}/{effort_id}/progress-claims",
                    query={"limit": limit},
                )
            )
        )

    def create_progress_claim(
        self,
        effort_id: str,
        request: InternProgressClaimCreateRequest,
    ) -> InternProgressClaimResponse:
        """Fold a result into an objective.

        This is the Intern half of kickoff-and-poll: a swarm run finishes on
        the SMR surfaces, and the claim records what it proved about the
        objective. ``smr_run_id`` on the request is a read-only reference.
        """
        return InternProgressClaimResponse.from_wire(
            self._transport.execute(
                _request(
                    "create_intern_effort_progress_claim",
                    f"{_EFFORTS}/{effort_id}/progress-claims",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    def list_objective_links(
        self,
        effort_id: str,
        *,
        limit: int = _LINK_LIMIT_DEFAULT,
    ) -> tuple[InternObjectiveLinkResponse, ...]:
        """List the references from this Effort's objectives to SMR-owned rows."""
        return _links(
            self._transport.execute(
                _request(
                    "list_intern_effort_objective_links",
                    f"{_EFFORTS}/{effort_id}/links",
                    query={"limit": limit},
                )
            )
        )

    def create_objective_link(
        self,
        objective_id: str,
        request: InternObjectiveLinkCreateRequest,
    ) -> InternObjectiveLinkResponse:
        """Reference an SMR-owned row from an Intern objective.

        The link is a reference. It writes one Intern row and never touches the
        target, so it grants no authority over the linked run, report, or
        work product.
        """
        return InternObjectiveLinkResponse.from_wire(
            self._transport.execute(
                _request(
                    "create_intern_objective_link",
                    f"{_OBJECTIVES}/{objective_id}/links",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    # -- Memory -------------------------------------------------------------

    def search_memory(
        self,
        query: str,
        *,
        kinds: tuple[str, ...] | None = None,
        limit: int = _MEMORY_LIMIT_DEFAULT,
    ) -> InternMemorySearchResponse:
        """Search the Intern's own history by keyword.

        The result is shaped exactly like the agent-facing memory tool result,
        so what an operator reads here is what the Intern read.

        Args:
            query: Keyword query.
            kinds: Optional record kinds to restrict to.
            limit: Maximum hits; the backend clamps it.
        """
        return InternMemorySearchResponse.from_wire(
            self._transport.execute(
                _request(
                    "search_intern_memory",
                    _MEMORY,
                    query=_memory_query(query, kinds, limit),
                )
            )
        )

    def get_memory_hit(self, hit_id: str) -> InternMemoryHitResponse:
        """Fetch one memory hit by the stable ``kind:id`` handle search returned."""
        return InternMemoryHitResponse.from_wire(
            self._transport.execute(_request("get_intern_memory_hit", f"{_MEMORY}/{hit_id}"))
        )


class AsyncInternProgramAPI:
    """Native asynchronous peer of :class:`InternProgramAPI`."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    # -- Effort board -------------------------------------------------------

    async def list_efforts(
        self,
        *,
        factory_id: str | None = None,
        status: str | None = None,
        limit: int = _EFFORT_LIMIT_DEFAULT,
        cursor: str | None = None,
    ) -> InternEffortBoardResponse:
        """List Efforts as board rows: the Effort plus its open-work counts."""
        return InternEffortBoardResponse.from_wire(
            await self._transport.execute(
                _request(
                    "list_intern_efforts",
                    _EFFORTS,
                    query=_effort_board_query(
                        factory_id=factory_id,
                        status=status,
                        limit=limit,
                        cursor=cursor,
                    ),
                )
            )
        )

    async def get_effort(self, effort_id: str) -> InternEffortDetailResponse:
        """Fetch one Effort's rollup: progress, results, experiments, knowledge."""
        return InternEffortDetailResponse.from_wire(
            await self._transport.execute(
                _request("get_intern_effort_detail", f"{_EFFORTS}/{effort_id}")
            )
        )

    # -- Objectives ---------------------------------------------------------

    async def list_objectives(
        self,
        effort_id: str,
        *,
        status: InternObjectiveStatus | str | None = None,
        limit: int = _OBJECTIVE_LIMIT_DEFAULT,
    ) -> tuple[InternObjectiveResponse, ...]:
        """List the Intern objectives opened under one Effort."""
        return _objectives(
            await self._transport.execute(
                _request(
                    "list_intern_effort_objectives",
                    f"{_EFFORTS}/{effort_id}/objectives",
                    query=_objective_query(status, limit),
                )
            )
        )

    async def create_objective(
        self,
        effort_id: str,
        request: InternObjectiveCreateRequest,
    ) -> InternObjectiveResponse:
        """Open one objective under an Effort; the path segment is the binding."""
        return InternObjectiveResponse.from_wire(
            await self._transport.execute(
                _request(
                    "create_intern_effort_objective",
                    f"{_EFFORTS}/{effort_id}/objectives",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    async def get_objective(
        self,
        objective_id: str,
        *,
        objective_kind: InternObjectiveKind | str,
    ) -> InternObjectiveResponse:
        """Fetch one objective; ``objective_kind`` selects the backing table."""
        return InternObjectiveResponse.from_wire(
            await self._transport.execute(
                _request(
                    "get_intern_objective",
                    f"{_OBJECTIVES}/{objective_id}",
                    query={"objective_kind": str(objective_kind)},
                )
            )
        )

    async def update_objective(
        self,
        objective_id: str,
        request: InternObjectivePatchRequest,
    ) -> InternObjectiveResponse:
        """Revise one objective; ``request.objective_kind`` selects the table."""
        return InternObjectiveResponse.from_wire(
            await self._transport.execute(
                _request(
                    "patch_intern_objective",
                    f"{_OBJECTIVES}/{objective_id}",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    # -- Milestones ---------------------------------------------------------

    async def list_milestones(
        self,
        effort_id: str,
        *,
        state: InternMilestoneState | str | None = None,
        limit: int = _MILESTONE_LIMIT_DEFAULT,
    ) -> tuple[InternMilestoneResponse, ...]:
        """List the milestones under one Effort."""
        return _milestones(
            await self._transport.execute(
                _request(
                    "list_intern_effort_milestones",
                    f"{_EFFORTS}/{effort_id}/milestones",
                    query=_milestone_query(state, limit),
                )
            )
        )

    async def create_milestone(
        self,
        effort_id: str,
        request: InternMilestoneCreateRequest,
    ) -> InternMilestoneResponse:
        """Break one objective into a subquestion or suboutcome milestone."""
        return InternMilestoneResponse.from_wire(
            await self._transport.execute(
                _request(
                    "create_intern_effort_milestone",
                    f"{_EFFORTS}/{effort_id}/milestones",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    async def transition_milestone(
        self,
        milestone_id: str,
        request: InternMilestoneTransitionRequest,
    ) -> InternMilestoneResponse:
        """Move a milestone to its next lifecycle state; illegal moves are refused."""
        return InternMilestoneResponse.from_wire(
            await self._transport.execute(
                _request(
                    "transition_intern_milestone",
                    f"{_MILESTONES}/{milestone_id}/transition",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    # -- Tasks --------------------------------------------------------------

    async def list_tasks(
        self,
        effort_id: str,
        *,
        state: InternEffortTaskState | str | None = None,
        milestone_id: str | None = None,
        limit: int = _TASK_LIMIT_DEFAULT,
    ) -> tuple[InternEffortTaskResponse, ...]:
        """List the Intern planner checklist for one Effort (not swarm run tasks)."""
        return _tasks(
            await self._transport.execute(
                _request(
                    "list_intern_effort_tasks",
                    f"{_EFFORTS}/{effort_id}/tasks",
                    query=_task_query(state, milestone_id, limit),
                )
            )
        )

    async def create_task(
        self,
        effort_id: str,
        request: InternEffortTaskCreateRequest,
    ) -> InternEffortTaskResponse:
        """Add one task to the Intern planner checklist for an Effort."""
        return InternEffortTaskResponse.from_wire(
            await self._transport.execute(
                _request(
                    "create_intern_effort_task",
                    f"{_EFFORTS}/{effort_id}/tasks",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    async def update_task(
        self,
        intern_task_id: str,
        request: InternEffortTaskPatchRequest,
    ) -> InternEffortTaskResponse:
        """Revise one planner task; ``state`` moves validate against the state machine."""
        return InternEffortTaskResponse.from_wire(
            await self._transport.execute(
                _request(
                    "patch_intern_effort_task",
                    f"{_TASKS}/{intern_task_id}",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    # -- Progress claims and links -----------------------------------------

    async def list_progress_claims(
        self,
        effort_id: str,
        *,
        limit: int = _CLAIM_LIMIT_DEFAULT,
    ) -> tuple[InternProgressClaimResponse, ...]:
        """List the append-only progress claims folded into one Effort."""
        return _claims(
            await self._transport.execute(
                _request(
                    "list_intern_effort_progress_claims",
                    f"{_EFFORTS}/{effort_id}/progress-claims",
                    query={"limit": limit},
                )
            )
        )

    async def create_progress_claim(
        self,
        effort_id: str,
        request: InternProgressClaimCreateRequest,
    ) -> InternProgressClaimResponse:
        """Fold a finished run's result into the objective that motivated it."""
        return InternProgressClaimResponse.from_wire(
            await self._transport.execute(
                _request(
                    "create_intern_effort_progress_claim",
                    f"{_EFFORTS}/{effort_id}/progress-claims",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    async def list_objective_links(
        self,
        effort_id: str,
        *,
        limit: int = _LINK_LIMIT_DEFAULT,
    ) -> tuple[InternObjectiveLinkResponse, ...]:
        """List the references from this Effort's objectives to SMR-owned rows."""
        return _links(
            await self._transport.execute(
                _request(
                    "list_intern_effort_objective_links",
                    f"{_EFFORTS}/{effort_id}/links",
                    query={"limit": limit},
                )
            )
        )

    async def create_objective_link(
        self,
        objective_id: str,
        request: InternObjectiveLinkCreateRequest,
    ) -> InternObjectiveLinkResponse:
        """Reference an SMR-owned row from an Intern objective, writing only the reference."""
        return InternObjectiveLinkResponse.from_wire(
            await self._transport.execute(
                _request(
                    "create_intern_objective_link",
                    f"{_OBJECTIVES}/{objective_id}/links",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )

    # -- Memory -------------------------------------------------------------

    async def search_memory(
        self,
        query: str,
        *,
        kinds: tuple[str, ...] | None = None,
        limit: int = _MEMORY_LIMIT_DEFAULT,
    ) -> InternMemorySearchResponse:
        """Search the Intern's own history by keyword."""
        return InternMemorySearchResponse.from_wire(
            await self._transport.execute(
                _request(
                    "search_intern_memory",
                    _MEMORY,
                    query=_memory_query(query, kinds, limit),
                )
            )
        )

    async def get_memory_hit(self, hit_id: str) -> InternMemoryHitResponse:
        """Fetch one memory hit by the stable ``kind:id`` handle search returned."""
        return InternMemoryHitResponse.from_wire(
            await self._transport.execute(_request("get_intern_memory_hit", f"{_MEMORY}/{hit_id}"))
        )


__all__ = [
    "AsyncInternProgramAPI",
    "InternProgramAPI",
]
