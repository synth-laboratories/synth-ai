"""Artifact data model for rollout outputs.

Artifacts are the concrete outputs of AI workflows (code, JSON, text, files).
They are stored separately from traces and linked via trace_correlation_id.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

try:
    from . import rust as _rust_data
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for data.artifacts.") from exc

# Re-export context override types for backward compatibility
from synth_ai.data.coding_agent_context import (
    ApplicationErrorType,
    ApplicationStatus,
    ContextOverride,
    ContextOverrideStatus,
    FolderMode,
    OverrideApplicationError,
    OverrideOperation,
    OverrideType,
)


class Artifact(BaseModel):
    """Artifact produced by a rollout.

    Artifacts are the concrete outputs of AI workflows (code, JSON, text, files).
    They are stored separately from traces and linked via trace_correlation_id.

    Fields:
        content: Inline artifact payload (string or JSON-like dict).
        content_type: Optional free-form content type identifier (e.g., "rust_code").
            Can be inferred from content type (dict vs str) and metadata.file_path.
        metadata: Artifact-specific metadata (file_path, line_count, schema_version).
        artifact_id: Backend-assigned identifier after storage.
        trace_correlation_id: Trace correlation ID for linkage.
        size_bytes: Size in bytes (when known).
        sha256: Hash of the content (when known).
        storage: Storage metadata for non-inline artifacts (bucket/key/url).
        created_at: ISO timestamp when stored.
    """

    content: str | Dict[str, Any]
    content_type: Optional[str] = Field(
        default=None,
        description="Optional content type identifier. Can be inferred from content type "
        "(dict = structured data, str = text/code) and metadata.file_path extension.",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    artifact_id: Optional[str] = None
    trace_correlation_id: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    storage: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None

    def validate_size(self, max_size_bytes: int = 10 * 1024 * 1024) -> None:
        """Validate artifact size (client-side constraint).

        Args:
            max_size_bytes: Maximum allowed size for inline artifact content.
        """
        if isinstance(self.content, str):
            size = len(self.content.encode("utf-8"))
        else:
            size = len(json.dumps(self.content, ensure_ascii=True).encode("utf-8"))

        if size > max_size_bytes:
            raise ValueError(
                f"Artifact size {size} bytes exceeds maximum {max_size_bytes} bytes. "
                f"Content type: {self.content_type}"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Artifact:
        if _rust_data is not None:
            with contextlib.suppress(Exception):
                data = _rust_data.normalize_artifact(data)  # noqa: F811
        return cls.model_validate(data)


try:  # Require Rust-backed class
    import synth_ai_py as _rust_models  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for data.artifacts.") from exc

with contextlib.suppress(AttributeError):
    Artifact = _rust_models.Artifact  # noqa: F811


__all__ = [
    "Artifact",
    # Re-exports for backward compatibility
    "OverrideType",
    "OverrideOperation",
    "FolderMode",
    "ApplicationStatus",
    "ApplicationErrorType",
    "ContextOverride",
    "OverrideApplicationError",
    "ContextOverrideStatus",
]
