"""HTTP-safe serialization helpers for tracing v3.

These utilities normalize tracing structures (including dataclasses) into
JSON-serializable forms and provide a compact JSON encoder suitable for
HTTP transmission to backend services.

Design goals:
- Preserve structure while ensuring standard-compliant JSON (no NaN/Infinity)
- Handle common non-JSON types: datetime, Decimal, bytes, set/tuple, numpy scalars
- Keep output compact (no unnecessary whitespace) while readable if needed
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - numpy optional at runtime
    _np = None  # type: ignore


def normalize_for_json(value: Any) -> Any:
    """Return a JSON-serializable version of ``value``.

    Rules:
    - dataclass → dict (recursively normalized)
    - datetime/date → ISO-8601 string (UTC-aware datetimes preserve tzinfo)
    - Decimal → float (fallback to string if not finite)
    - bytes/bytearray → base64 string (RFC 4648)
    - set/tuple → list
    - Enum → enum.value (normalized)
    - numpy scalar → corresponding Python scalar
    - float NaN/Inf/−Inf → None (to keep JSON standard compliant)
    - dict / list → recursively normalized
    - other primitives (str, int, bool, None, float) passed through
    """

    # Dataclasses
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return normalize_for_json(asdict(value))
        except Exception:
            # Fallback: best-effort conversion via __dict__
            return normalize_for_json(getattr(value, "__dict__", {}))

    # Mapping
    if isinstance(value, dict):
        return {str(k): normalize_for_json(v) for k, v in value.items()}

    # Sequences
    if isinstance(value, list | tuple | set):
        return [normalize_for_json(v) for v in value]

    # Datetime / Date
    if isinstance(value, datetime | date):
        return value.isoformat()

    # Decimal
    if isinstance(value, Decimal):
        try:
            f = float(value)
            if f != f or f in (float("inf"), float("-inf")):
                return str(value)
            return f
        except Exception:
            return str(value)

    # Bytes-like
    if isinstance(value, bytes | bytearray):
        return base64.b64encode(bytes(value)).decode("ascii")

    # Enum
    if isinstance(value, Enum):
        return normalize_for_json(value.value)

    # Numpy scalars / arrays
    if _np is not None:
        if isinstance(value, _np.generic):  # type: ignore[attr-defined]
            return normalize_for_json(value.item())
        if isinstance(value, _np.ndarray):
            return normalize_for_json(value.tolist())

    # Floats: sanitize NaN / Infinity to None
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value

    return value


def dumps_http_json(payload: Any) -> str:
    """Dump ``payload`` into a compact, HTTP-safe JSON string.

    - Recursively normalizes non-JSON types (see ``normalize_for_json``)
    - Disallows NaN/Infinity per RFC 8259 (allow_nan=False)
    - Uses compact separators and preserves Unicode (ensure_ascii=False)
    """

    normalized = normalize_for_json(payload)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def serialize_trace_for_http(trace: Any) -> str:
    """Serialize a tracing v3 session (or dict-like) to HTTP-safe JSON.

    Accepts either a dataclass (e.g., SessionTrace) or a dict/list and
    applies normalization and compact JSON encoding.
    """

    if is_dataclass(trace) and not isinstance(trace, type):
        try:
            return dumps_http_json(asdict(trace))
        except Exception:
            return dumps_http_json(getattr(trace, "__dict__", {}))
    return dumps_http_json(trace)


