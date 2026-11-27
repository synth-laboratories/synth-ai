"""MCP Tasks (2026-07-28 / io.modelcontextprotocol/tasks) support for stdio MCP (SYN-6057).

Tasks are backed durably by Core `run_id`s with bounded TTL and strict tenant
authority enforced by the underlying `AsyncSynthClient` credential.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from mcp import McpError
from mcp.types import (
    INVALID_PARAMS,
    CallToolResult,
    CancelTaskResult,
    CreateTaskResult,
    ErrorData,
    GetTaskPayloadResult,
    GetTaskResult,
    Task,
    TaskMetadata,
    TaskStatus,
    TextContent,
)

from synth_ai._generated.models import LaunchRequest, RunSnapshot
from synth_ai.errors import (
    NotFoundError,
    ResultUnavailableError,
    SwarmsAPIError,
)

from ._arguments import optional, required

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import JsonValue

    from synth_ai.aclient import AsyncSynthClient

__all__ = [
    "DEFAULT_POLL_INTERVAL_MS",
    "DEFAULT_TASK_TTL_MS",
    "MAX_TASK_TTL_MS",
    "RELATED_TASK_META_KEY",
    "TASK_SUPPORTED_TOOLS",
    "build_task_from_run",
    "call_tool_as_task",
    "cancel_task",
    "get_task",
    "get_task_result",
]

TASK_SUPPORTED_TOOLS = frozenset({"swarms_run_launch", "swarms_run_wait"})
DEFAULT_TASK_TTL_MS = 24 * 60 * 60 * 1000
MAX_TASK_TTL_MS = 7 * 24 * 60 * 60 * 1000
DEFAULT_POLL_INTERVAL_MS = 2000
RELATED_TASK_META_KEY = "io.modelcontextprotocol/related-task"
TASK_NOT_TERMINAL_CODE = -32002
RUN_ID_PATTERN = re.compile(r"^run_[A-Za-z0-9_-]{22}$")
TERMINAL_TASK_STATUSES: frozenset[TaskStatus] = frozenset({"completed", "failed", "cancelled"})


def validate_task_id(task_id: str) -> str:
    cleaned = task_id.strip()
    if not RUN_ID_PATTERN.fullmatch(cleaned):
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message="Invalid params: taskId must be a valid run identifier.",
            )
        )
    return cleaned


def parse_timestamp(raw: str | None, fallback: datetime) -> datetime:
    if raw is None or not raw:
        return fallback
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return fallback
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def bounded_ttl_ms(requested: int | None) -> int:
    if requested is None:
        return DEFAULT_TASK_TTL_MS
    if requested < 0:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message="Invalid params: task.ttl must be a nonnegative integer.",
            )
        )
    return min(requested, MAX_TASK_TTL_MS)


def map_run_state(
    snapshot: RunSnapshot,
    pending_approvals: int,
) -> tuple[TaskStatus, str]:
    state = str(snapshot.state)
    if state == "paused":
        if pending_approvals > 0:
            return (
                "input_required",
                f"Run paused with {pending_approvals} pending approval(s).",
            )
        return ("input_required", "Run is paused and waiting for operator input.")
    if state in {"queued", "running", "pausing", "cancelling"}:
        if pending_approvals > 0:
            return (
                "input_required",
                f"Run has {pending_approvals} pending approval(s) requiring decision.",
            )
        return ("working", f"Run state: {state}")
    if state in {"completed", "partial"}:
        return ("completed", f"Run state: {state}")
    if state == "failed":
        if isinstance(snapshot.failure, dict):
            raw_detail = snapshot.failure.get("detail")
            raw_code = snapshot.failure.get("code")
            detail = raw_detail if isinstance(raw_detail, str) and raw_detail else None
            code = raw_code if isinstance(raw_code, str) and raw_code else None
            summary = detail if detail is not None else code
            if summary is not None:
                return ("failed", f"Run failed: {summary}")
        return ("failed", "Run failed.")
    if state == "cancelled":
        return ("cancelled", "Run cancelled.")
    return ("working", f"Run state: {state}")


async def count_pending_approvals(client: AsyncSynthClient, snapshot: RunSnapshot) -> int:
    if str(snapshot.state) not in {"running", "paused"}:
        return 0
    try:
        page = await client.swarms.approvals(snapshot.run_id, status="pending", limit=20)
    except SwarmsAPIError:
        return 0
    return len(page.items)


async def build_task_from_run(
    client: AsyncSynthClient,
    snapshot: RunSnapshot,
    *,
    requested_ttl_ms: int | None = None,
    now: datetime | None = None,
) -> Task:
    current = now if now is not None else datetime.now(UTC)
    created_at = parse_timestamp(snapshot.created_at, current)
    last_updated = parse_timestamp(snapshot.closed_at or snapshot.started_at, created_at)
    pending = await count_pending_approvals(client, snapshot)
    status, status_message = map_run_state(snapshot, pending)
    ttl_ms = bounded_ttl_ms(requested_ttl_ms)
    if status in TERMINAL_TASK_STATUSES:
        age_ms = int((current - last_updated).total_seconds() * 1000)
        if age_ms > MAX_TASK_TTL_MS:
            raise McpError(
                ErrorData(
                    code=INVALID_PARAMS,
                    message=f"Task expired: {snapshot.run_id}",
                )
            )
        ttl_ms = max(MAX_TASK_TTL_MS - age_ms, 0)
    poll_interval = None if status in TERMINAL_TASK_STATUSES else DEFAULT_POLL_INTERVAL_MS
    return Task(
        taskId=snapshot.run_id,
        status=status,
        statusMessage=status_message,
        createdAt=created_at,
        lastUpdatedAt=last_updated,
        ttl=ttl_ms,
        pollInterval=poll_interval,
    )


async def call_tool_as_task(
    client: AsyncSynthClient,
    name: str,
    arguments: Mapping[str, JsonValue],
    task_meta: TaskMetadata,
) -> CreateTaskResult:
    if name not in TASK_SUPPORTED_TOOLS:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message=f"Tool '{name}' does not support task-augmented execution.",
            )
        )
    ttl_ms = bounded_ttl_ms(task_meta.ttl)
    if name == "swarms_run_launch":
        snapshot = await client.swarms.launch_request(
            required(arguments, "body", LaunchRequest),
            idempotency_key=optional(arguments, "idempotency_key", str),
        )
    else:
        run_id = validate_task_id(required(arguments, "run_id", str))
        snapshot = await client.swarms.get(run_id)
    task = await build_task_from_run(client, snapshot, requested_ttl_ms=ttl_ms)
    return CreateTaskResult(task=task)


async def get_task(client: AsyncSynthClient, task_id: str) -> GetTaskResult:
    run_id = validate_task_id(task_id)
    try:
        snapshot = await client.swarms.get(run_id)
    except NotFoundError as error:
        raise McpError(
            ErrorData(code=INVALID_PARAMS, message=f"Task not found: {run_id}")
        ) from error
    task = await build_task_from_run(client, snapshot)
    return GetTaskResult(
        taskId=task.taskId,
        status=task.status,
        statusMessage=task.statusMessage,
        createdAt=task.createdAt,
        lastUpdatedAt=task.lastUpdatedAt,
        ttl=task.ttl,
        pollInterval=task.pollInterval,
    )


async def get_task_result(
    client: AsyncSynthClient,
    task_id: str,
    *,
    wait_seconds: int = 30,
) -> GetTaskPayloadResult:
    run_id = validate_task_id(task_id)
    try:
        snapshot = await client.swarms.get(run_id)
    except NotFoundError as error:
        raise McpError(
            ErrorData(code=INVALID_PARAMS, message=f"Task not found: {run_id}")
        ) from error
    task = await build_task_from_run(client, snapshot)
    if task.status not in TERMINAL_TASK_STATUSES:
        waited = await client.swarms.wait_update(run_id, timeout_seconds=wait_seconds)
        snapshot = waited.run
        task = await build_task_from_run(client, snapshot)
    if task.status not in TERMINAL_TASK_STATUSES:
        raise McpError(
            ErrorData(
                code=TASK_NOT_TERMINAL_CODE,
                message="Task is not terminal yet; poll tasks/get and retry tasks/result.",
                data={"taskId": run_id, "status": task.status},
            )
        )
    payload_text: str
    is_error = False
    try:
        result = await client.swarms.result(run_id)
        payload_text = result.model_dump_json(exclude_unset=True)
    except ResultUnavailableError:
        payload_text = snapshot.model_dump_json(exclude_unset=True)
        is_error = task.status == "failed"
    tool_result = CallToolResult(
        content=[TextContent(type="text", text=payload_text)],
        structuredContent=cast("dict[str, object]", json.loads(payload_text)),
        isError=is_error,
    )
    dumped = tool_result.model_dump(mode="json", exclude_none=True)
    dumped["_meta"] = {RELATED_TASK_META_KEY: {"taskId": run_id}}
    return GetTaskPayloadResult.model_validate(dumped)


async def cancel_task(client: AsyncSynthClient, task_id: str) -> CancelTaskResult:
    run_id = validate_task_id(task_id)
    try:
        snapshot = await client.swarms.get(run_id)
    except NotFoundError as error:
        raise McpError(
            ErrorData(code=INVALID_PARAMS, message=f"Task not found: {run_id}")
        ) from error
    task = await build_task_from_run(client, snapshot)
    if task.status in TERMINAL_TASK_STATUSES:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message=f"Cannot cancel a task in terminal status: {task.status}",
            )
        )
    updated = await client.swarms.cancel(run_id, idempotency_key=f"mcp-task-cancel-{run_id}")
    cancelled = await build_task_from_run(client, updated)
    return CancelTaskResult(
        taskId=cancelled.taskId,
        status="cancelled",
        statusMessage="Cancellation requested.",
        createdAt=cancelled.createdAt,
        lastUpdatedAt=cancelled.lastUpdatedAt,
        ttl=cancelled.ttl,
    )
