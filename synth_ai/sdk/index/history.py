"""Private-history request bounds. See backend SDK index_contract.md."""

from __future__ import annotations

from synth_ai.core.parameters import verified

from .types.history import SearchHistoryFilter, SearchHistoryPage


def validate_history_page(
    result: SearchHistoryPage, limit: int, selected: SearchHistoryFilter
) -> SearchHistoryPage:
    return verified(
        result,
        len(result.items) <= limit and result.filter == selected,
        "History page changed selected filters or exceeded its requested limit",
    )
