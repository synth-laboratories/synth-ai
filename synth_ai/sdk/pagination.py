"""Cursor pagination helpers for hero SDK list methods."""

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
        raise TypeError("page must be an object or legacy list")
    items = payload.get("items") if "items" in payload else payload.get("data")
    if not isinstance(items, list):
        raise ValueError("page items must be an array")
    next_cursor = payload.get("next_cursor") if "next_cursor" in payload else payload.get("cursor")
    if next_cursor is not None and (not isinstance(next_cursor, str) or not next_cursor.strip()):
        raise ValueError("page next_cursor must be a non-empty string or null")
    has_more = payload.get("has_more", next_cursor is not None)
    if not isinstance(has_more, bool):
        raise ValueError("page has_more must be a boolean")
    if has_more != (next_cursor is not None):
        raise ValueError("page continuation contradicts next_cursor")
    return list(items), next_cursor, has_more


__all__ = ["SyncPage", "page_from_wire"]
