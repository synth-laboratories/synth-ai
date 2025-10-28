"""Shared JSON sanitisation helpers for Task Apps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

try:  # numpy is optional at runtime; degrade gracefully if absent
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    _np = None  # type: ignore


def _mask_numpy_array(arr: Any) -> str:
    shape = getattr(arr, "shape", None)
    dtype = getattr(arr, "dtype", None)
    return f"<ndarray shape={shape} dtype={dtype}>"


def to_jsonable(
    value: Any,
    *,
    _visited: set[int] | None = None,
    _depth: int = 0,
    _max_depth: int = 32,
) -> Any:
    """Convert `value` into structures compatible with JSON serialisation.

    - numpy scalars are converted to their Python counterparts
    - numpy arrays are represented by a compact descriptor string
    - dataclasses, Enums, and pydantic models are unwrapped recursively
    - sets and tuples are converted to lists
    - non-serialisable objects fall back to `repr`
    """

    if _visited is None:
        _visited = set()

    if _depth > _max_depth:
        return f"<max_depth type={type(value).__name__}>"

    if value is None or isinstance(value, str | bool | int | float):
        return value

    # numpy scalars / arrays
    if _np is not None:
        if isinstance(value, _np.integer):
            return int(value)
        if isinstance(value, _np.floating):
            return float(value)
        if isinstance(value, _np.bool_):
            return bool(value)
        if isinstance(value, _np.ndarray):
            return _mask_numpy_array(value)

    if isinstance(value, Enum):
        return to_jsonable(value.value, _visited=_visited, _depth=_depth + 1, _max_depth=_max_depth)

    if is_dataclass(value):
        return to_jsonable(
            asdict(value), _visited=_visited, _depth=_depth + 1, _max_depth=_max_depth
        )

    # pydantic BaseModel / attrs objects
    for attr in ("model_dump", "dict", "to_dict", "to_json"):
        if hasattr(value, attr) and callable(getattr(value, attr, None)):
            try:
                dumped = getattr(value, attr)()  # type: ignore[misc]
            except TypeError:
                dumped = getattr(value, attr)(exclude_none=False)  # pragma: no cover
            return to_jsonable(
                dumped, _visited=_visited, _depth=_depth + 1, _max_depth=_max_depth
            )

    obj_id = id(value)
    if obj_id in _visited:
        return f"<circular type={type(value).__name__}>"

    if isinstance(value, Mapping):
        _visited.add(obj_id)
        return {
            str(k): to_jsonable(v, _visited=_visited, _depth=_depth + 1, _max_depth=_max_depth)
            for k, v in value.items()
        }

    if isinstance(value, set | tuple):
        _visited.add(obj_id)
        return [
            to_jsonable(v, _visited=_visited, _depth=_depth + 1, _max_depth=_max_depth)
            for v in value
        ]

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        _visited.add(obj_id)
        return [
            to_jsonable(v, _visited=_visited, _depth=_depth + 1, _max_depth=_max_depth)
            for v in value
        ]

    if isinstance(value, bytes | bytearray):
        return f"<bytes len={len(value)}>"

    if hasattr(value, "__dict__"):
        _visited.add(obj_id)
        return to_jsonable(
            vars(value), _visited=_visited, _depth=_depth + 1, _max_depth=_max_depth
        )

    return repr(value)
