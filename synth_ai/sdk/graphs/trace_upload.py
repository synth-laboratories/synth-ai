"""Trace upload utilities for large trace files.

This module provides functionality for uploading large traces directly to
blob storage (S3/Wasabi) via presigned URLs, avoiding HTTP timeout issues
that occur when sending large payloads through the API.

Example:
    ```python
    from synth_ai.sdk.graphs import GraphCompletionsClient, TraceUploader

    client = GraphCompletionsClient(base_url, api_key)
    uploader = TraceUploader(base_url, api_key)

    # Upload large trace
    trace_ref = await uploader.upload_trace(large_trace)

    # Use trace_ref in verifier call
    result = await client.verify_with_rubric(
        session_trace=trace_ref,  # Pass ref instead of full trace
        rubric=rubric,
    )
    ```
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping

from synth_ai.core.tracing_v3.serialization import normalize_for_json

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for sdk.graphs.trace_upload.") from exc

# Size threshold for automatic upload (100KB)
AUTO_UPLOAD_THRESHOLD_BYTES = 100 * 1024

# Maximum trace size allowed (50MB)
MAX_TRACE_SIZE_BYTES = 50 * 1024 * 1024


def _require_rust() -> Any:
    if synth_ai_py is None:
        raise RuntimeError("synth_ai_py is required for trace upload. Install rust bindings.")
    return synth_ai_py


@dataclass
class UploadUrlResponse:
    """Response from creating an upload URL."""

    trace_id: str
    """Unique identifier for this trace upload."""

    trace_ref: str
    """The trace reference to use in API requests (e.g., "trace:trace_abc123")."""

    upload_url: str
    """Presigned URL for uploading the trace (HTTP PUT)."""

    expires_in_seconds: int
    """When the presigned URL expires (seconds from now)."""

    storage_key: str
    """The storage key where the trace will be stored."""

    max_size_bytes: int
    """Maximum allowed trace size in bytes."""


class TraceUploadError(Exception):
    """Error during trace upload operations."""

    pass


class TraceUploaderSync:
    """Synchronous client for uploading traces to blob storage.

    Use this when you have large traces (>100KB) that would timeout
    when sent through the API directly.

    Example:
        ```python
        uploader = TraceUploaderSync(base_url, api_key)

        # Check if trace should be uploaded
        if uploader.should_upload(trace):
            trace_ref = uploader.upload_trace(trace)
            # Use trace_ref in API calls
        ```
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 120.0,
        auto_upload_threshold: int = AUTO_UPLOAD_THRESHOLD_BYTES,
    ) -> None:
        """Initialize the trace uploader.

        Args:
            base_url: Graph service base URL
            api_key: API key for authentication
            timeout: Request timeout in seconds (default: 120s for large uploads)
            auto_upload_threshold: Size in bytes above which traces are auto-uploaded
        """
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._timeout = timeout
        self._auto_upload_threshold = auto_upload_threshold
        rust = _require_rust()
        self._rust_client = rust.TraceUploadClient(self._base, self._key, int(timeout))

    def should_upload(self, trace: Mapping[str, Any]) -> bool:
        """Check if a trace should be uploaded via presigned URL.

        Args:
            trace: The trace data

        Returns:
            True if the trace is large enough to warrant upload
        """
        return bool(
            self._rust_client.should_upload(normalize_for_json(trace), self._auto_upload_threshold)
        )

    def get_trace_size(self, trace: Mapping[str, Any]) -> int:
        """Get the serialized size of a trace in bytes.

        Args:
            trace: The trace data

        Returns:
            Size in bytes when serialized to JSON
        """
        return int(self._rust_client.trace_size(normalize_for_json(trace)))

    def create_upload_url(
        self,
        *,
        content_type: str = "application/json",
        expires_in_seconds: int | None = None,
    ) -> UploadUrlResponse:
        """Request a presigned upload URL from the backend.

        Args:
            content_type: MIME type of the content (default: application/json)
            expires_in_seconds: URL expiration time (default: 300s)

        Returns:
            UploadUrlResponse with upload URL and trace reference

        Raises:
            TraceUploadError: If the request fails
        """
        data = self._rust_client.create_upload_url(content_type, expires_in_seconds)
        return UploadUrlResponse(
            trace_id=data["trace_id"],
            trace_ref=data["trace_ref"],
            upload_url=data["upload_url"],
            expires_in_seconds=data["expires_in_seconds"],
            storage_key=data["storage_key"],
            max_size_bytes=data.get("max_size_bytes", MAX_TRACE_SIZE_BYTES),
        )

    def upload_trace(
        self,
        trace: Mapping[str, Any],
        *,
        expires_in_seconds: int | None = None,
    ) -> str:
        """Upload a trace and return its reference.

        This method:
        1. Requests a presigned upload URL
        2. Uploads the trace directly to S3
        3. Returns the trace_ref to use in API calls

        Args:
            trace: The trace data to upload
            expires_in_seconds: URL expiration time (default: 300s)

        Returns:
            trace_ref string (e.g., "trace:trace_abc123")

        Raises:
            TraceUploadError: If upload fails
        """
        return self._rust_client.upload_trace(
            normalize_for_json(trace), "application/json", expires_in_seconds
        )


class TraceUploaderAsync:
    """Asynchronous client for uploading traces to blob storage.

    Use this when you have large traces (>100KB) that would timeout
    when sent through the API directly.

    Example:
        ```python
        uploader = TraceUploaderAsync(base_url, api_key)

        # Check if trace should be uploaded
        if uploader.should_upload(trace):
            trace_ref = await uploader.upload_trace(trace)
            # Use trace_ref in API calls
        ```
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 120.0,
        auto_upload_threshold: int = AUTO_UPLOAD_THRESHOLD_BYTES,
    ) -> None:
        """Initialize the trace uploader.

        Args:
            base_url: Graph service base URL
            api_key: API key for authentication
            timeout: Request timeout in seconds (default: 120s for large uploads)
            auto_upload_threshold: Size in bytes above which traces are auto-uploaded
        """
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._timeout = timeout
        self._auto_upload_threshold = auto_upload_threshold
        rust = _require_rust()
        self._rust_client = rust.TraceUploadClient(self._base, self._key, int(timeout))

    def should_upload(self, trace: Mapping[str, Any]) -> bool:
        """Check if a trace should be uploaded via presigned URL.

        Args:
            trace: The trace data

        Returns:
            True if the trace is large enough to warrant upload
        """
        return bool(
            self._rust_client.should_upload(normalize_for_json(trace), self._auto_upload_threshold)
        )

    def get_trace_size(self, trace: Mapping[str, Any]) -> int:
        """Get the serialized size of a trace in bytes.

        Args:
            trace: The trace data

        Returns:
            Size in bytes when serialized to JSON
        """
        return int(self._rust_client.trace_size(normalize_for_json(trace)))

    async def create_upload_url(
        self,
        *,
        content_type: str = "application/json",
        expires_in_seconds: int | None = None,
    ) -> UploadUrlResponse:
        """Request a presigned upload URL from the backend.

        Args:
            content_type: MIME type of the content (default: application/json)
            expires_in_seconds: URL expiration time (default: 300s)

        Returns:
            UploadUrlResponse with upload URL and trace reference

        Raises:
            TraceUploadError: If the request fails
        """
        data = await asyncio.to_thread(
            self._rust_client.create_upload_url, content_type, expires_in_seconds
        )
        return UploadUrlResponse(
            trace_id=data["trace_id"],
            trace_ref=data["trace_ref"],
            upload_url=data["upload_url"],
            expires_in_seconds=data["expires_in_seconds"],
            storage_key=data["storage_key"],
            max_size_bytes=data.get("max_size_bytes", MAX_TRACE_SIZE_BYTES),
        )

    async def upload_trace(
        self,
        trace: Mapping[str, Any],
        *,
        expires_in_seconds: int | None = None,
    ) -> str:
        """Upload a trace and return its reference.

        This method:
        1. Requests a presigned upload URL
        2. Uploads the trace directly to S3
        3. Returns the trace_ref to use in API calls

        Args:
            trace: The trace data to upload
            expires_in_seconds: URL expiration time (default: 300s)

        Returns:
            trace_ref string (e.g., "trace:trace_abc123")

        Raises:
            TraceUploadError: If upload fails
        """
        return await asyncio.to_thread(
            self._rust_client.upload_trace,
            normalize_for_json(trace),
            "application/json",
            expires_in_seconds,
        )


# Aliases for convenience
TraceUploader = TraceUploaderAsync
"""Alias for TraceUploaderAsync (default)."""
