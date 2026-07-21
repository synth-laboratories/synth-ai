"""Cursor pagination contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Mapping, TypeVar

from synth_ai.core.contracts.json_value import JsonValue


ItemT = TypeVar("ItemT")


class PageCursor(str):
    """Opaque cursor returned by the backend."""


@dataclass(frozen=True, slots=True)
class Page(Generic[ItemT]):
    """One immutable page of typed resources."""

    items: tuple[ItemT, ...]
    next_cursor: PageCursor | None = None


def build_query_params(**values: JsonValue) -> dict[str, JsonValue]:
    """Drop only absent or blank-string query values."""

    return {
        key: value
        for key, value in values.items()
        if value is not None and not (isinstance(value, str) and not value.strip())
    }


def extract_next_cursor(payload: Mapping[str, object]) -> PageCursor | None:
    """Decode the canonical next cursor without accepting spelling fallbacks."""

    value = payload.get("next_cursor")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("next_cursor must be a non-empty string when provided")
    return PageCursor(value.strip())


__all__ = ["Page", "PageCursor", "build_query_params", "extract_next_cursor"]
