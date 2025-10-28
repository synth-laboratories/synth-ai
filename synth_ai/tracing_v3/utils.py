"""Utility functions for tracing v3."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any


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
    if not model_name:
        return "unknown"

    model_lower = model_name.lower()

    if any(
        x in model_lower for x in ["gpt-", "text-davinci", "text-curie", "text-babbage", "text-ada"]
    ):
        return "openai"
    elif any(x in model_lower for x in ["claude", "anthropic"]):
        return "anthropic"
    elif any(x in model_lower for x in ["palm", "gemini", "bard"]):
        return "google"
    elif "azure" in model_lower:
        return "azure"
    elif any(x in model_lower for x in ["llama", "mistral", "mixtral", "local"]):
        return "local"
    else:
        return "unknown"


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float | None:
    """Calculate cost in USD based on model and token counts."""
    # This is a simplified version - in production you'd want a proper pricing table
    pricing = {
        "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }

    for model_prefix, prices in pricing.items():
        if model_prefix in model_name.lower():
            input_cost = (input_tokens / 1000) * prices["input"]
            output_cost = (output_tokens / 1000) * prices["output"]
            return input_cost + output_cost

    return None


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
