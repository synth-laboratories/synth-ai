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


@dataclass(frozen=True)
class AsyncPage(Generic[ItemT]):
    """Async list page (same wire shape as ``SyncPage``)."""

    items: list[ItemT]
    next_cursor: str | None = None
    has_more: bool = False


def page_from_wire(payload: dict[str, object] | list[object]) -> tuple[list[object], str | None, bool]:
    if isinstance(payload, list):
        return list(payload), None, False
    items = payload.get("items")
    if not isinstance(items, list):
        items_obj = payload.get("data")
        items = items_obj if isinstance(items_obj, list) else []
    next_cursor = payload.get("next_cursor") or payload.get("cursor")
    cursor_text = str(next_cursor).strip() if next_cursor is not None else None
    has_more = bool(payload.get("has_more")) if "has_more" in payload else bool(cursor_text)
    return list(items), cursor_text or None, has_more


__all__ = ["AsyncPage", "SyncPage", "page_from_wire"]
