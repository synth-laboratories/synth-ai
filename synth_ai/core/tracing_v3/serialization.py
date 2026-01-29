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

from typing import Any

try:
    import synth_ai_py  # type: ignore
except Exception as exc:  # pragma: no cover - rust bindings required
    raise RuntimeError("synth_ai_py is required for tracing serialization.") from exc


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
    return synth_ai_py.normalize_for_json(value)


def dumps_http_json(payload: Any) -> str:
    """Dump ``payload`` into a compact, HTTP-safe JSON string.

    - Recursively normalizes non-JSON types (see ``normalize_for_json``)
    - Disallows NaN/Infinity per RFC 8259 (allow_nan=False)
    - Uses compact separators and preserves Unicode (ensure_ascii=False)
    """
    return synth_ai_py.dumps_http_json(payload)


def serialize_trace_for_http(trace: Any) -> str:
    """Serialize a tracing v3 session (or dict-like) to HTTP-safe JSON.

    Accepts either a dataclass (e.g., SessionTrace) or a dict/list and
    applies normalization and compact JSON encoding.
    """
    return synth_ai_py.dumps_http_json(trace)
