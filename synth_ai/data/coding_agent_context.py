"""Context override data models for coding agent optimization.

These models define context overrides used by unified optimization (GEPA) to modify
the agent's runtime environment alongside system prompts. Includes file artifacts,
preflight scripts, environment variables, and application status reporting.
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

try:
    from . import rust as _rust_data
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for data.coding_agent_context.") from exc

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

    Note:
        File artifact paths are relative to the task app workspace. Agent-specific
        paths such as `.codex/skills.yaml` or `.opencode/skills.yaml` are resolved
        by the task app based on the agent type.

    Warning:
        Task apps enforce size limits on files, scripts, and env vars. Large
        overrides may be rejected at application time.
    """

    # File artifacts (e.g., AGENTS.md, .codex/skills.yaml, .opencode/skills.yaml)
    # Keys are relative paths, values are file contents
    file_artifacts: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Files to write to the workspace. Keys are relative paths (e.g., "
            "'AGENTS.md', 'skills/test.sh', '.codex/skills.yaml'), values are file contents."
        ),
    )

    # Bash script to run before agent execution
    # Must start with #!/bin/bash shebang (validated by task app)
    preflight_script: Optional[str] = Field(
        default=None,
        description=(
            "Bash script to run before agent execution. Must start with a #!/bin/bash "
            "shebang and will run with task app timeouts."
        ),
    )

    # Environment variables to set for agent
    env_vars: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Environment variables to set for the agent process. Keys must start with "
            "a letter or underscore and values are size-limited."
        ),
    )

    # Metadata
    mutation_type: Optional[str] = Field(
        default=None,
        description="Type of mutation: 'prompt', 'file_artifacts', 'preflight_script', 'multi', 'crossover'",
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the override was created.",
    )
    override_id: Optional[str] = Field(
        default=None, description="Unique identifier for this override bundle"
    )

    def is_empty(self) -> bool:
        """Check if this override has no actual content.

        Returns:
            True if file_artifacts, preflight_script, and env_vars are all empty.

        Example:
            empty = ContextOverride().is_empty()
        """
        return not self.file_artifacts and not self.preflight_script and not self.env_vars

    def size_bytes(self) -> int:
        """Calculate total size of override content in bytes.

        Returns:
            Total UTF-8 encoded size of files, script, and env vars.
        """
        total = 0
        for content in self.file_artifacts.values():
            total += len(content.encode("utf-8"))
        if self.preflight_script:
            total += len(self.preflight_script.encode("utf-8"))
        for key, value in self.env_vars.items():
            total += len(key.encode("utf-8")) + len(value.encode("utf-8"))
        return total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization.

        Returns:
            Dict with only populated fields for compact payloads.
        """
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContextOverride:
        if _rust_data is not None:
            with contextlib.suppress(Exception):
                data = _rust_data.normalize_context_override(data)  # noqa: F811
        return cls.model_validate(data)


class OverrideApplicationError(BaseModel):
    """Error details for a failed override application.

    Example:
        error = OverrideApplicationError(
            error_type=ApplicationErrorType.PERMISSION,
            message="Permission denied writing AGENTS.md",
            target="AGENTS.md",
        )
    """

    error_type: ApplicationErrorType = Field(
        ...,
        description="Classification of the error (validation, permission, size_limit, etc.).",
    )
    message: str = Field(..., description="Human-readable error message.")
    target: Optional[str] = Field(default=None, description="Target path/key that failed")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured details about the error (optional).",
    )


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

    Note:
        overall_status reflects aggregate status across file_artifacts, preflight_script,
        and env_vars. Partial indicates at least one target failed and at least one
        target succeeded.
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
        """Convert to dict for serialization.

        Returns:
            Dict with only populated fields for compact payloads.
        """
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


try:  # Require Rust-backed classes
    import synth_ai_py as _rust_models  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for data.coding_agent_context.") from exc

with contextlib.suppress(AttributeError):
    ContextOverride = _rust_models.ContextOverride  # noqa: F811
    ContextOverrideStatus = _rust_models.ContextOverrideStatus  # noqa: F811


__all__ = [
    # Enums
    "OverrideType",
    "OverrideOperation",
    "FolderMode",
    "ApplicationStatus",
    "ApplicationErrorType",
    # Data models
    "ContextOverride",
    "OverrideApplicationError",
    "ContextOverrideStatus",
]
