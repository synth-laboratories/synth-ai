"""Artifact data model for rollout outputs."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Artifact(BaseModel):
    """Artifact produced by a rollout.

    Artifacts are the concrete outputs of AI workflows (code, JSON, text, files).
    They are stored separately from traces and linked via trace_correlation_id.
    """

    content: str | Dict[str, Any]
    content_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    artifact_id: Optional[str] = None
    trace_correlation_id: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    storage: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None

    def validate_size(self, max_size_bytes: int = 10 * 1024 * 1024) -> None:
        """Validate artifact size (client-side constraint)."""
        if isinstance(self.content, str):
            size = len(self.content.encode("utf-8"))
        else:
            size = len(json.dumps(self.content, ensure_ascii=True).encode("utf-8"))

        if size > max_size_bytes:
            raise ValueError(
                f"Artifact size {size} bytes exceeds maximum {max_size_bytes} bytes. "
                f"Content type: {self.content_type}"
            )
