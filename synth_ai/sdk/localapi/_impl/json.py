"""Shared JSON sanitisation helpers for Task Apps."""

from __future__ import annotations

from typing import Any

try:
    import synth_ai_py
except Exception:  # pragma: no cover - optional dependency
    synth_ai_py = None


def to_jsonable(
    value: Any,
    *,
    _visited: set[int] | None = None,
    _depth: int = 0,
    _max_depth: int = 32,
) -> Any:
    """Convert `value` into structures compatible with JSON serialisation."""

    if synth_ai_py is not None and hasattr(synth_ai_py, "localapi_to_jsonable"):
        return synth_ai_py.localapi_to_jsonable(value)
    return _fallback_to_jsonable(
        value,
        _visited=_visited,
        _depth=_depth,
        _max_depth=_max_depth,
    )


def _fallback_to_jsonable(
    value: Any,
    *,
    _visited: set[int] | None,
    _depth: int,
    _max_depth: int,
) -> Any:
    if _visited is None:
        _visited = set()

    if _depth > _max_depth:
        return str(value)

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")

    value_id = id(value)
    if value_id in _visited:
        return "<recursion>"

    _visited.add(value_id)
    try:
        if isinstance(value, dict):
            converted = {}
            for key, item in value.items():
                key_json = _fallback_to_jsonable(
                    key,
                    _visited=_visited,
                    _depth=_depth + 1,
                    _max_depth=_max_depth,
                )
                if not isinstance(key_json, str):
                    key_json = str(key_json)
                converted[key_json] = _fallback_to_jsonable(
                    item,
                    _visited=_visited,
                    _depth=_depth + 1,
                    _max_depth=_max_depth,
                )
            return converted

        if isinstance(value, (list, tuple, set, frozenset)):
            return [
                _fallback_to_jsonable(
                    item,
                    _visited=_visited,
                    _depth=_depth + 1,
                    _max_depth=_max_depth,
                )
                for item in value
            ]

        if hasattr(value, "model_dump") and callable(value.model_dump):
            return _fallback_to_jsonable(
                value.model_dump(),
                _visited=_visited,
                _depth=_depth + 1,
                _max_depth=_max_depth,
            )

        if hasattr(value, "dict") and callable(value.dict):
            return _fallback_to_jsonable(
                value.dict(),
                _visited=_visited,
                _depth=_depth + 1,
                _max_depth=_max_depth,
            )

        if hasattr(value, "__dict__"):
            return _fallback_to_jsonable(
                value.__dict__,
                _visited=_visited,
                _depth=_depth + 1,
                _max_depth=_max_depth,
            )

        return str(value)
    finally:
        _visited.discard(value_id)
