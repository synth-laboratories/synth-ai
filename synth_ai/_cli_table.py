"""Shared human-readable formatting primitives for CLI views."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._generated.models import Money


def money(value: Money | None) -> str:
    return "-" if value is None else value.amount + " " + value.currency


def facts(rows: Sequence[tuple[str, object]]) -> str:
    width = max(len(label) for label, _ in rows)
    return "\n".join(
        label.ljust(width) + "  " + ("-" if value is None else str(value)) for label, value in rows
    )


def table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    cells = [list(header), *[["-" if c is None else str(c) for c in row] for row in rows]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in cells
    )


def with_next_cursor(rendered: str, *, has_more: bool, next_cursor: str | None) -> str:
    if not has_more or next_cursor is None or next_cursor == "":
        return rendered
    footer = f"next cursor  {next_cursor}"
    return f"{rendered}\n\n{footer}" if rendered else footer
