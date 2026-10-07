"""Cursor pagination helpers for hero SDK list methods.

# See: testing/specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

ItemT = TypeVar("ItemT")


@dataclass(frozen=True)
class SyncPage(Generic[ItemT]):
    """One page of list results with optional cursor continuation."""

    items: list[ItemT]
    next_cursor: str | None = None
    has_more: bool = False


def page_from_wire(
    payload: dict[str, object] | list[object],
) -> tuple[list[object], str | None, bool]:
    if isinstance(payload, list):
        return list(payload), None, False
    if not isinstance(payload, dict):
        raise ValueError("page response must be an object or array")
    items = payload.get("items") if "items" in payload else payload.get("data")
    if not isinstance(items, list):
        raise ValueError("page response requires an items or data array")
    # Explicit null is terminal evidence; do not resurrect a stale legacy token.
    next_cursor = payload.get("next_cursor") if "next_cursor" in payload else payload.get("cursor")
    if next_cursor is not None and not isinstance(next_cursor, str):
        raise ValueError("page cursor must be a string or null")
    has_more = payload.get("has_more", bool(next_cursor))
    if not isinstance(has_more, bool):
        raise ValueError("page has_more must be a boolean")
    if has_more and not next_cursor:
        raise ValueError("continuing page requires a nonempty cursor")
    return list(items), next_cursor or None, has_more


__all__ = ["SyncPage", "page_from_wire"]
