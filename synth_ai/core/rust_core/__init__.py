"""Bridge layer for Rust core functionality.

This module provides Python-facing wrappers that mirror the Rust core APIs
and delegate directly to the Rust bindings.
"""

from synth_ai.core.rust_core.http import RustCoreHttpClient, http_request
from synth_ai.core.rust_core.sse import stream_sse_events
from synth_ai.core.rust_core.urls import ensure_api_base, normalize_base_url

__all__ = [
    "RustCoreHttpClient",
    "stream_sse_events",
    "http_request",
    "normalize_base_url",
    "ensure_api_base",
]
