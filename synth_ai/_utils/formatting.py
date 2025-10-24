from __future__ import annotations

from typing import Any

__all__ = ["format_int", "format_currency", "safe_str"]


def format_int(value: Any) -> str:
    """Return an integer formatted with thousands separators."""

    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def format_currency(value: Any, *, precision: int = 4, symbol: str = "$") -> str:
    """Return a currency string with the requested precision."""

    try:
        amount = float(value or 0.0)
    except Exception:
        amount = 0.0
    return f"{symbol}{amount:.{precision}f}"


def safe_str(value: Any) -> str:
    """Gracefully stringify objects, falling back to '-' for failures."""

    if value is None:
        return "-"
    try:
        return str(value)
    except Exception:
        return "-"
