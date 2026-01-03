"""Actions for TUI views."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

from .data import PromptLearningDataClient
from .models import ActionResult


class TuiAction(str, Enum):
    CANCEL_JOB = "cancel_job"
    REFRESH = "refresh"
    FETCH_ARTIFACTS = "fetch_artifacts"
    FETCH_SNAPSHOT = "fetch_snapshot"


@dataclass(slots=True)
class TuiActionRequest:
    action: TuiAction
    job_id: str | None = None
    payload: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.payload is None:
            self.payload = {}


async def handle_prompt_learning_action(
    client: PromptLearningDataClient,
    request: TuiActionRequest,
) -> ActionResult:
    if request.action == TuiAction.CANCEL_JOB and request.job_id:
        return await client.cancel_job(request.job_id)
    if request.action == TuiAction.FETCH_ARTIFACTS and request.job_id:
        artifacts = await client.list_artifacts(request.job_id)
        return ActionResult(ok=True, message="Artifacts fetched", payload={"artifacts": artifacts})
    if request.action == TuiAction.FETCH_SNAPSHOT and request.job_id:
        snapshot_id = str(request.payload.get("snapshot_id") or "")
        if not snapshot_id:
            return ActionResult(ok=False, message="snapshot_id is required", payload={})
        snapshot = await client.get_snapshot(request.job_id, snapshot_id)
        return ActionResult(ok=True, message="Snapshot fetched", payload={"snapshot": snapshot})
    if request.action == TuiAction.REFRESH:
        return ActionResult(ok=True, message="Refresh requested", payload={})
    return ActionResult(ok=False, message="Unsupported action", payload={})

