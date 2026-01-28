"""Shared JSON sanitisation helpers for Task Apps."""

from __future__ import annotations

from typing import Any

import synth_ai_py


def to_jsonable(
    value: Any,
    *,
    _visited: set[int] | None = None,
    _depth: int = 0,
    _max_depth: int = 32,
) -> Any:
    """Convert `value` into structures compatible with JSON serialisation."""

    return synth_ai_py.localapi_to_jsonable(value)
