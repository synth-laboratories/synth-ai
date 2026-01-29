"""Shared SDK utilities (Rust core)."""

from synth_ai.core.errors import HTTPError
from synth_ai.core.rust_core.http import RustCoreHttpClient, http_request, sleep

# Backward-compatible alias for callers still importing AsyncHttpClient.
AsyncHttpClient = RustCoreHttpClient

__all__ = [
    "AsyncHttpClient",
    "HTTPError",
    "http_request",
    "sleep",
]
