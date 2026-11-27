"""Progress notifications (`notifications/progress`) and logging for stdio MCP (SYN-6057)."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from typing import TYPE_CHECKING

from synth_ai._generated.models import (
    ObjectiveProgressPayload,
    RunApprovalRequestedPayload,
    RunEvent,
    RunMilestone,
    RunWait,
    RunWaitOutcome,
)
from synth_ai.errors import SwarmsAPIError

from .elicitation import client_supports_elicitation, elicit_approval_decision

if TYPE_CHECKING:
    from mcp.server.session import ServerSession
    from mcp.types import LoggingLevel, ProgressToken, RequestId

    from synth_ai.aclient import AsyncSynthClient

__all__ = [
    "LOG_LEVEL_ORDER",
    "format_event_progress",
    "wait_run_with_progress",
]

LOG_LEVEL_ORDER: dict[str, int] = {
    "debug": 10,
    "info": 20,
    "notice": 25,
    "warning": 30,
    "error": 40,
    "critical": 50,
    "alert": 60,
    "emergency": 70,
}

MILESTONE_LEVELS: dict[str, int] = {
    "execution_closed": 1,
    "accounting_final": 2,
    "cleanup_complete": 3,
}

TERMINAL_RUN_STATES = frozenset({"completed", "partial", "failed", "cancelled"})


def should_emit_log(level: LoggingLevel, minimum: LoggingLevel | None) -> bool:
    if minimum is None:
        return True
    return LOG_LEVEL_ORDER.get(level, 20) >= LOG_LEVEL_ORDER.get(minimum, 20)


def format_event_progress(
    item: RunEvent | RunMilestone,
    step: int,
) -> tuple[float, float | None, str, LoggingLevel]:
    """Translate a RunEvent or RunMilestone into progress + log notification fields."""
    if isinstance(item, RunMilestone):
        level = MILESTONE_LEVELS.get(str(item.milestone), 1)
        return (
            float(level),
            3.0,
            f"Milestone {item.milestone} (state={item.state})",
            "info",
        )
    if isinstance(item.payload, ObjectiveProgressPayload):
        return (
            float(item.payload.percent),
            100.0,
            f"Objective {item.payload.name}: {item.payload.percent}%",
            "info",
        )
    if isinstance(item.payload, RunApprovalRequestedPayload):
        return (
            float(step),
            None,
            f"Approval requested ({item.payload.approval_id}): {item.payload.summary}",
            "notice",
        )
    if item.kind == "limit.warning":
        return (float(step), None, f"Run event: {item.kind}", "warning")
    return (float(step), None, f"Run event: {item.kind}", "info")


async def emit_progress_and_log(
    session: ServerSession,
    *,
    progress_token: ProgressToken | None,
    progress: float,
    total: float | None,
    message: str,
    level: LoggingLevel,
    minimum_log_level: LoggingLevel | None,
    related_request_id: RequestId | None,
) -> None:
    if progress_token is not None:
        await session.send_progress_notification(
            progress_token=progress_token,
            progress=progress,
            total=total,
            message=message,
            related_request_id=str(related_request_id) if related_request_id is not None else None,
        )
    if should_emit_log(level, minimum_log_level):
        await session.send_log_message(
            level=level,
            data=message,
            logger="synth.swarms",
            related_request_id=related_request_id,
        )


async def wait_run_with_progress(
    client: AsyncSynthClient,
    session: ServerSession,
    run_id: str,
    *,
    timeout_seconds: int,
    progress_token: ProgressToken | None,
    related_request_id: RequestId | None,
    minimum_log_level: LoggingLevel | None = None,
) -> RunWait:
    """Wait on a run via live SSE, emitting MCP progress/log notifications and
    eliciting interactive approval decisions when supported by the client.
    """
    step = 0
    reached_terminal = False
    prompted_approvals: set[tuple[str, int]] = set()
    can_elicit = client_supports_elicitation(session)

    try:
        async with asyncio.timeout(float(timeout_seconds)):
            async with aclosing(client.swarms.follow(run_id)) as stream:
                async for item in stream:
                    step += 1
                    progress, total, message, level = format_event_progress(item, step)
                    await emit_progress_and_log(
                        session,
                        progress_token=progress_token,
                        progress=progress,
                        total=total,
                        message=message,
                        level=level,
                        minimum_log_level=minimum_log_level,
                        related_request_id=related_request_id,
                    )
                    if (
                        isinstance(item.payload, RunApprovalRequestedPayload)
                        and str(item.payload.status) == "pending"
                        and can_elicit
                    ):
                        key = (item.payload.approval_id, item.payload.revision)
                        if key not in prompted_approvals:
                            prompted_approvals.add(key)
                            outcome = await elicit_approval_decision(
                                client,
                                session,
                                run_id,
                                approval_id=item.payload.approval_id,
                                action_kind=item.payload.action_kind,
                                summary=item.payload.summary,
                                revision=item.payload.revision,
                                related_request_id=related_request_id,
                            )
                            step += 1
                            await emit_progress_and_log(
                                session,
                                progress_token=progress_token,
                                progress=float(step),
                                total=None,
                                message=(
                                    f"Approval {outcome.approval_id}: {outcome.status}"
                                    + (
                                        f" ({outcome.decision})"
                                        if outcome.decision is not None
                                        else ""
                                    )
                                    + (f" - {outcome.detail}" if outcome.detail is not None else "")
                                ),
                                level="info" if outcome.status == "decided" else "warning",
                                minimum_log_level=minimum_log_level,
                                related_request_id=related_request_id,
                            )
            reached_terminal = True
    except TimeoutError:
        reached_terminal = False

    snapshot = await client.swarms.get(run_id)
    if str(snapshot.state) == "paused" and can_elicit:
        try:
            pending_page = await client.swarms.approvals(run_id, status="pending", limit=10)
        except SwarmsAPIError:
            pending_page = None
        if pending_page is not None:
            for approval in pending_page.items:
                key = (approval.approval_id, approval.revision)
                if key in prompted_approvals:
                    continue
                prompted_approvals.add(key)
                outcome = await elicit_approval_decision(
                    client,
                    session,
                    run_id,
                    approval_id=approval.approval_id,
                    action_kind=approval.action_kind,
                    summary=approval.summary,
                    revision=approval.revision,
                    related_request_id=related_request_id,
                )
                step += 1
                await emit_progress_and_log(
                    session,
                    progress_token=progress_token,
                    progress=float(step),
                    total=None,
                    message=(
                        f"Approval {outcome.approval_id}: {outcome.status}"
                        + (f" ({outcome.decision})" if outcome.decision is not None else "")
                    ),
                    level="info" if outcome.status == "decided" else "warning",
                    minimum_log_level=minimum_log_level,
                    related_request_id=related_request_id,
                )
            if pending_page.items:
                snapshot = await client.swarms.get(run_id)

    state = str(snapshot.state)
    if reached_terminal or state in TERMINAL_RUN_STATES:
        milestone_level = (
            3.0
            if snapshot.cleanup_complete
            else 2.0
            if snapshot.accounting_final
            else 1.0
            if snapshot.execution_closed
            else 0.0
        )
        if milestone_level > 0.0:
            await emit_progress_and_log(
                session,
                progress_token=progress_token,
                progress=milestone_level,
                total=3.0,
                message=f"Run {run_id} terminal (state={state})",
                level="info",
                minimum_log_level=minimum_log_level,
                related_request_id=related_request_id,
            )
        wait_outcome = RunWaitOutcome.TERMINAL
    elif state == "paused":
        wait_outcome = RunWaitOutcome.PAUSED
    else:
        wait_outcome = RunWaitOutcome.TIMEOUT
    return RunWait(outcome=wait_outcome, run=snapshot)
