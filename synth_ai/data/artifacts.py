"""Artifact data model for rollout outputs."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Context Override Enums
# =============================================================================


class OverrideType(str, Enum):
    """Type of context override target."""

    FILE = "file"  # Single file (e.g., AGENTS.md, .codex/skills.yaml)
    FOLDER = "folder"  # Directory with multiple files
    PREFLIGHT_SCRIPT = "preflight_script"  # Bash script run before agent
    ENV_VAR = "env_var"  # Environment variable


class OverrideOperation(str, Enum):
    """Operation to apply for an override."""

    REPLACE = "replace"  # Full replacement (default)
    PATCH = "patch"  # Unified diff patch
    DELETE = "delete"  # Remove the target


class FolderMode(str, Enum):
    """How to handle folder overrides."""

    REPLACE = "replace"  # Replace entire folder contents
    MERGE = "merge"  # Merge with existing (default)


class ApplicationStatus(str, Enum):
    """Status of override application by task app."""

    APPLIED = "applied"  # Successfully applied
    PARTIAL = "partial"  # Partially applied (some sub-items failed)
    FAILED = "failed"  # Application failed
    SKIPPED = "skipped"  # Skipped (e.g., target doesn't exist for delete)


class ApplicationErrorType(str, Enum):
    """Classification of override application error."""

    VALIDATION = "validation"  # Invalid override spec
    PATH_TRAVERSAL = "path_traversal"  # Attempt to escape sandbox
    PERMISSION = "permission"  # Permission denied
    SIZE_LIMIT = "size_limit"  # Content too large
    TIMEOUT = "timeout"  # Script timeout
    RUNTIME = "runtime"  # Runtime error during apply
    NOT_FOUND = "not_found"  # Target not found (for patch/delete)


# =============================================================================
# Context Override Data Models
# =============================================================================


class ContextOverride(BaseModel):
    """Context override for unified optimization.

    Represents modifications to the agent's runtime environment that can be
    optimized alongside the system prompt. Includes:
    - file_artifacts: Files to write to agent workspace (e.g., AGENTS.md, skills/*.yaml)
    - preflight_script: Bash script to run before agent execution
    - env_vars: Environment variables to set

    Example:
        override = ContextOverride(
            file_artifacts={"AGENTS.md": "# Guidelines\n...", ".codex/skills.yaml": "style:\n  verbosity: concise"},
            preflight_script="#!/bin/bash\necho 'Setup complete'",
            env_vars={"STRATEGY": "test_first"},
            mutation_type="multi"
        )
    """

    # File artifacts (e.g., AGENTS.md, .codex/skills.yaml, .opencode/skills.yaml)
    # Keys are relative paths, values are file contents
    file_artifacts: Dict[str, str] = Field(default_factory=dict)

    # Bash script to run before agent execution
    # Must start with #!/bin/bash shebang (validated by task app)
    preflight_script: Optional[str] = None

    # Environment variables to set for agent
    env_vars: Dict[str, str] = Field(default_factory=dict)

    # Metadata
    mutation_type: Optional[str] = Field(
        default=None,
        description="Type of mutation: 'prompt', 'file_artifacts', 'preflight_script', 'multi', 'crossover'",
    )
    created_at: Optional[datetime] = None
    override_id: Optional[str] = Field(
        default=None, description="Unique identifier for this override bundle"
    )

    def is_empty(self) -> bool:
        """Check if this override has no actual content."""
        return (
            not self.file_artifacts
            and not self.preflight_script
            and not self.env_vars
        )

    def size_bytes(self) -> int:
        """Calculate total size of override content in bytes."""
        total = 0
        for content in self.file_artifacts.values():
            total += len(content.encode("utf-8"))
        if self.preflight_script:
            total += len(self.preflight_script.encode("utf-8"))
        for key, value in self.env_vars.items():
            total += len(key.encode("utf-8")) + len(value.encode("utf-8"))
        return total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        result: Dict[str, Any] = {}
        if self.file_artifacts:
            result["file_artifacts"] = self.file_artifacts
        if self.preflight_script:
            result["preflight_script"] = self.preflight_script
        if self.env_vars:
            result["env_vars"] = self.env_vars
        if self.mutation_type:
            result["mutation_type"] = self.mutation_type
        if self.created_at:
            result["created_at"] = self.created_at.isoformat()
        if self.override_id:
            result["override_id"] = self.override_id
        return result


class OverrideApplicationError(BaseModel):
    """Error details for a failed override application."""

    error_type: ApplicationErrorType
    message: str
    target: Optional[str] = Field(
        default=None, description="Target path/key that failed"
    )
    details: Dict[str, Any] = Field(default_factory=dict)


class ContextOverrideStatus(BaseModel):
    """Status of context override application by task app.

    Reports per-target application results so GEPA can learn from failures.

    Example:
        status = ContextOverrideStatus(
            override_id="ov_abc123",
            overall_status=ApplicationStatus.PARTIAL,
            file_artifacts={
                "AGENTS.md": {"status": "applied", "bytes_written": 1024},
                ".codex/skills.yaml": {"status": "failed", "error": "Permission denied"},
            },
            preflight_script={"status": "applied", "exit_code": 0, "stdout": "Setup complete"},
            env_vars={"STRATEGY": {"status": "applied"}},
        )
    """

    override_id: Optional[str] = Field(
        default=None, description="ID of the override bundle that was applied"
    )
    overall_status: ApplicationStatus = ApplicationStatus.APPLIED
    errors: list[OverrideApplicationError] = Field(default_factory=list)

    # Per-target results
    file_artifacts: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Status per file artifact: {path: {status, bytes_written, error}}",
    )
    preflight_script: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Preflight script result: {status, exit_code, stdout, stderr, duration_ms}",
    )
    env_vars: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Status per env var: {key: {status, error}}",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        result: Dict[str, Any] = {
            "overall_status": self.overall_status.value,
        }
        if self.override_id:
            result["override_id"] = self.override_id
        if self.errors:
            result["errors"] = [
                {"error_type": e.error_type.value, "message": e.message, "target": e.target}
                for e in self.errors
            ]
        if self.file_artifacts:
            result["file_artifacts"] = self.file_artifacts
        if self.preflight_script:
            result["preflight_script"] = self.preflight_script
        if self.env_vars:
            result["env_vars"] = self.env_vars
        return result


# =============================================================================
# Artifact Model
# =============================================================================


class Artifact(BaseModel):
    """Artifact produced by a rollout.

    Artifacts are the concrete outputs of AI workflows (code, JSON, text, files).
    They are stored separately from traces and linked via trace_correlation_id.

    Fields:
        content: Inline artifact payload (string or JSON-like dict).
        content_type: Free-form content type identifier (e.g., "rust_code").
        metadata: Artifact-specific metadata (file_path, line_count, schema_version).
        artifact_id: Backend-assigned identifier after storage.
        trace_correlation_id: Trace correlation ID for linkage.
        size_bytes: Size in bytes (when known).
        sha256: Hash of the content (when known).
        storage: Storage metadata for non-inline artifacts (bucket/key/url).
        created_at: ISO timestamp when stored.
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
