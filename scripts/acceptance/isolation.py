"""Pure refusal comparison for tenant isolation (SYN-5737).

This module performs no I/O or journey execution. Missing routes, untyped
responses and uniformly wrong statuses cannot establish resource isolation.
"""

from __future__ import annotations

# ruff: noqa: UP045 -- Explicit Optional at acceptance contract boundaries.
import json
from collections.abc import Mapping
from typing import Optional, Protocol

__all__ = ["classify"]

RESOURCE_NOT_FOUND = frozenset(
    {
        "resource_not_found",
        "run_not_found",
        "file_not_found",
        "upload_not_found",
        "usage_export_not_found",
        "output_not_found",
        "key_not_found",
        "plan_request_not_found",
        "grant_not_found",
        "message_not_found",
        "run_message_not_found",
    }
)


class Observation(Protocol):
    @property
    def accepted(self) -> bool: ...
    @property
    def status(self) -> Optional[int]: ...
    @property
    def code(self) -> Optional[str]: ...
    @property
    def cause(self) -> str: ...
    @property
    def problem(self) -> Optional[Mapping[str, object]]: ...


def masked(problem: Optional[Mapping[str, object]], identifiers: tuple[str, ...]) -> str:
    text = json.dumps(problem if problem is not None else {}, sort_keys=True)
    for identifier in identifiers:
        text = text.replace(identifier, "<id>")
    return text


def classify(foreign: Observation, absent: Observation, identifiers: tuple[str, ...]) -> str:
    """Return pass/leak/differs or an explicit unproven reason.

    Only equal, typed resource-404 Problems prove isolation. In particular a
    removed route answering the same 404 for both requests remains unproven.
    """
    if foreign.accepted or absent.accepted:
        return "leak" if foreign.accepted else "differs"
    signature = [
        (item.status, item.code, item.cause, masked(item.problem, identifiers))
        for item in (foreign, absent)
    ]
    if signature[0] != signature[1]:
        return "differs"
    if foreign.status is None:
        return "unproven:never_reached_core"
    if foreign.status != 404:
        return f"unproven:uniform_{foreign.status}"
    if foreign.code not in RESOURCE_NOT_FOUND:
        return "unproven:resource_404_not_typed"
    if foreign.problem is None or foreign.problem.get("code") != foreign.code:
        return "unproven:resource_problem_missing"
    return "pass"
