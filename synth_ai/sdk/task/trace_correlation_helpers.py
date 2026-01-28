"""Backward-compatible trace correlation helpers for task apps."""

from __future__ import annotations

from synth_ai.sdk.localapi._impl.trace_correlation_helpers import (  # noqa: F401
    extract_trace_correlation_id,
)

__all__ = [
    "extract_trace_correlation_id",
]
