"""Utility functions for tracing v3."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover - rust bindings required
    raise RuntimeError("synth_ai_py is required for tracing_v3.utils.") from exc


def iso_now() -> str:
    """Get current timezone.utc time as ISO format string."""
    return datetime.now(UTC).isoformat()


def json_dumps(obj: Any) -> str:
    """JSON dump with consistent formatting and datetime handling."""
    return json.dumps(obj, default=str, separators=(",", ":"))


def json_loads(s: str) -> Any:
    """Safe JSON load with error handling."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}


def generate_session_id(prefix: str = "session") -> str:
    """Generate a unique session ID."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def generate_experiment_id(name: str) -> str:
    """Generate experiment ID from name."""
    hash_obj = hashlib.sha256(name.encode())
    return f"exp_{hash_obj.hexdigest()[:12]}"


def detect_provider(model_name: str | None) -> str:
    """Detect LLM provider from model name."""
    return synth_ai_py.tracing_detect_provider(model_name)


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float | None:
    """Calculate cost in USD based on model and token counts."""
    return synth_ai_py.tracing_calculate_cost(model_name, input_tokens, output_tokens)


def truncate_content(content: str, max_length: int = 10000) -> str:
    """Truncate content to maximum length."""
    if len(content) <= max_length:
        return content
    return content[: max_length - 3] + "..."


def format_duration(start: datetime, end: datetime) -> str:
    """Format duration between two timestamps."""
    delta = end - start
    total_seconds = delta.total_seconds()

    if total_seconds < 1:
        return f"{int(total_seconds * 1000)}ms"
    elif total_seconds < 60:
        return f"{total_seconds:.1f}s"
    elif total_seconds < 3600:
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
