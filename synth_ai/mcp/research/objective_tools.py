"""Typed vocabulary for objective MCP tools."""

from __future__ import annotations

from enum import StrEnum


class ObjectiveToolOperation(StrEnum):
    LIST = "list"
    CREATE = "create"
    GET = "get"
    PATCH = "patch"
    PAUSE = "pause"
    RESUME = "resume"
    WITHDRAW = "withdraw"
    PROGRESS = "progress"
    TASKS = "tasks"
    CLAIMS = "claims"
    CLAIM = "claim"
    REQUEST_REVIEW = "request_review"


def objective_tool_operation_from_wire(value: str) -> ObjectiveToolOperation:
    try:
        return ObjectiveToolOperation(value.strip().lower())
    except ValueError as exc:
        allowed = ", ".join(operation.value for operation in ObjectiveToolOperation)
        raise ValueError(f"'operation' must be one of: {allowed}") from exc


__all__ = [
    "ObjectiveToolOperation",
    "objective_tool_operation_from_wire",
]
