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


class CompatObjectiveToolOperation(StrEnum):
    LIST = "list"
    CREATE = "create"
    GET = "get"
    PATCH = "patch"
    TRANSITION = "transition"


class RunObjectiveScopeToolOperation(StrEnum):
    LIST = "list"
    REGISTER = "register"


def objective_tool_operation_from_wire(value: str) -> ObjectiveToolOperation:
    try:
        return ObjectiveToolOperation(value.strip().lower())
    except ValueError as exc:
        allowed = ", ".join(operation.value for operation in ObjectiveToolOperation)
        raise ValueError(f"'operation' must be one of: {allowed}") from exc


def compat_objective_tool_operation_from_wire(
    value: str,
) -> CompatObjectiveToolOperation:
    try:
        return CompatObjectiveToolOperation(value.strip().lower())
    except ValueError as exc:
        allowed = ", ".join(operation.value for operation in CompatObjectiveToolOperation)
        raise ValueError(f"'operation' must be one of: {allowed}") from exc


def run_objective_scope_tool_operation_from_wire(
    value: str,
) -> RunObjectiveScopeToolOperation:
    try:
        return RunObjectiveScopeToolOperation(value.strip().lower())
    except ValueError as exc:
        allowed = ", ".join(operation.value for operation in RunObjectiveScopeToolOperation)
        raise ValueError(f"'operation' must be one of: {allowed}") from exc


__all__ = [
    "CompatObjectiveToolOperation",
    "ObjectiveToolOperation",
    "RunObjectiveScopeToolOperation",
    "compat_objective_tool_operation_from_wire",
    "objective_tool_operation_from_wire",
    "run_objective_scope_tool_operation_from_wire",
]
