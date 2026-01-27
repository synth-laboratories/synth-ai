"""Rust-core HTTP exports for SDK compatibility."""

from __future__ import annotations

from synth_ai.core.errors import HTTPError
from synth_ai.core.rust_core.http import RustCoreHttpClient, http_request, sleep

AsyncHttpClient = RustCoreHttpClient

__all__ = ["AsyncHttpClient", "HTTPError", "http_request", "sleep"]
