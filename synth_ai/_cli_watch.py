"""Watch formatting helpers for unified `synth watch <run_id>` output (SYN-6058)."""

from __future__ import annotations

from ._generated.models import (
    ObjectiveClaimedPayload,
    ObjectiveProgressPayload,
    ResearchArtifactEventPayload,
    ResearchExperimentEventPayload,
    ResearchLogEventPayload,
    ResearchTrialEventPayload,
    RunApprovalDecidedPayload,
    RunApprovalRequestedPayload,
    RunEvent,
    RunLimitStatePayload,
    RunLimitWarningPayload,
    RunMessageEventPayload,
    RunMilestone,
    RunSnapshot,
)

__all__ = ["milestones_from_snapshot", "watch_item"]


def milestones_from_snapshot(value: RunSnapshot) -> list[RunMilestone]:
    """Derive reached closure milestones in order from a RunSnapshot."""
    flags = (
        ("execution_closed", bool(value.execution_closed)),
        ("accounting_final", bool(value.accounting_final)),
        ("cleanup_complete", bool(value.cleanup_complete)),
    )
    reached = 0
    for _, ok in flags:
        if ok:
            reached += 1
        else:
            break
    out: list[RunMilestone] = []
    for idx in range(reached):
        name = flags[idx][0]
        out.append(
            RunMilestone.model_validate(
                {
                    "milestone": name,
                    "run_id": value.run_id,
                    "state": str(value.state),
                    "execution_closed": idx >= 0,
                    "accounting_final": idx >= 1,
                    "cleanup_complete": idx >= 2,
                    "result_available": bool(value.result_available),
                    "failure": value.failure,
                    "final": idx == len(flags) - 1,
                }
            )
        )
    return out


def watch_item(item: RunEvent | RunMilestone) -> str:
    """Format one unified event or milestone line for `synth watch`."""
    if isinstance(item, RunMilestone):
        suffix = " (final)" if item.final else ""
        return (
            f"[milestone] {item.milestone} state={item.state} "
            f"closed={'yes' if item.execution_closed else 'no'} "
            f"accounting={'yes' if item.accounting_final else 'no'} "
            f"cleanup={'yes' if item.cleanup_complete else 'no'}{suffix}"
        )

    ts = item.observed_at or "-"
    prefix = f"{ts} #{item.sequence}"
    payload = item.payload

    if isinstance(payload, ObjectiveProgressPayload):
        return (
            f"{prefix} [progress] {payload.name} {payload.percent}% "
            f"(current={payload.current}, best={payload.best}, target={payload.target})"
        )
    if isinstance(payload, ObjectiveClaimedPayload):
        return (
            f"{prefix} [objective] {payload.objective_id} claimed "
            f"({payload.percent_complete}%): {payload.summary}"
        )
    if isinstance(payload, RunLimitWarningPayload):
        return (
            f"{prefix} [limit:warning] {payload.dimension} ({payload.scope}) "
            f"used={payload.used}/{payload.cap} reserved={payload.reserved}"
        )
    if isinstance(payload, RunLimitStatePayload):
        paused = " paused=yes" if payload.pause is not None else ""
        return f"{prefix} [limit] {item.kind} rev={payload.revision}{paused}"
    if isinstance(payload, RunMessageEventPayload):
        return (
            f"{prefix} [message] {item.kind} id={payload.message_id} "
            f"author={payload.author} state={payload.state.kind}"
        )
    if isinstance(payload, RunApprovalRequestedPayload):
        return (
            f"{prefix} [approval:requested] {payload.approval_id} ({payload.action_kind}) "
            f"rev={payload.revision}: {payload.summary}"
        )
    if isinstance(payload, RunApprovalDecidedPayload):
        reason = (
            f" reason={payload.reason}"
            if payload.reason is not None and payload.reason != ""
            else ""
        )
        return (
            f"{prefix} [approval:decided] {payload.approval_id} "
            f"decision={payload.decision} status={payload.status} rev={payload.revision}{reason}"
        )
    if isinstance(payload, ResearchArtifactEventPayload):
        return (
            f'{prefix} [research:artifact] {payload.artifact_id} "{payload.title}" '
            f"({payload.kind}, rev={payload.revision})"
        )
    if isinstance(payload, ResearchExperimentEventPayload):
        return (
            f'{prefix} [research:experiment] {payload.experiment_id} "{payload.title}" '
            f"(rev={payload.revision}, arms={len(payload.arms)})"
        )
    if isinstance(payload, ResearchTrialEventPayload):
        return (
            f"{prefix} [research:trial] {payload.experiment_id}/{payload.trial_id} "
            f"arm={payload.arm} state={payload.state}"
        )
    if isinstance(payload, ResearchLogEventPayload):
        correction = (
            f" correction_of={payload.correction_of}"
            if payload.correction_of is not None and payload.correction_of != ""
            else ""
        )
        return (
            f"{prefix} [research:{payload.category}] {payload.entry_id} "
            f"by {payload.author_label}{correction}"
        )
    return f"{prefix} [event] {item.kind}"
