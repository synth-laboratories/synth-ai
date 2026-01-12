"""Artifact models for AI workflow outputs.

Artifacts are the concrete outputs of AI workflows (e.g., code, JSON, text, files).
They are used by verifiers for evaluation and can be passed to proposers for examples.
"""

import json
import sys
from typing import Any

from pydantic import BaseModel, Field


class Artifact(BaseModel):
    """Artifact produced by AI workflow.

    Artifacts are the concrete outputs of AI workflows (e.g., code, JSON, text, files).
    They are used by verifiers for evaluation and can be passed to proposers for examples.

    Attributes:
        content: The actual artifact content (string for code/text, dict for structured data)
        content_type: Type identifier (e.g., "rust_code", "json", "markdown", "python_code")
        metadata: Additional context (e.g., file_path, line_count, format, schema_version)

    Example:
        ```python
        artifact = Artifact(
            content="pub struct PokemonCard { ... }",
            content_type="rust_code",
            metadata={"file_path": "pokemon.rs", "line_count": 150},
        )
        ```
    """

    content: str | dict[str, Any]  # The actual artifact content
    content_type: str  # e.g., "rust_code", "json", "markdown", "python_code"
    metadata: dict[str, Any] = Field(default_factory=dict)  # Additional context

    # Backend-assigned fields (optional, populated after storage)
    artifact_id: str | None = None
    trace_correlation_id: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    storage: dict[str, Any] | None = None  # Wasabi S3 info
    created_at: str | None = None  # ISO datetime string

    def validate_size(self, max_size_bytes: int = 10 * 1024 * 1024) -> None:
        """Validate artifact size (client-side hard constraint).

        Args:
            max_size_bytes: Maximum allowed size in bytes (default: 10MB)

        Raises:
            ValueError: If artifact exceeds size limit
        """
        if isinstance(self.content, str):
            size = len(self.content.encode("utf-8"))
        elif isinstance(self.content, dict):
            size = len(json.dumps(self.content).encode("utf-8"))
        else:
            size = sys.getsizeof(self.content)

        if size > max_size_bytes:
            raise ValueError(
                f"Artifact size {size} bytes exceeds maximum {max_size_bytes} bytes. "
                f"Content type: {self.content_type}"
            )


class ContextOverride(BaseModel):
    """Context override to be applied by task app before execution.

    Context overrides allow GEPA to mutate the agent's execution environment
    (e.g., AGENTS.md, skills/, workspace files) in addition to prompt edits.
    """

    override_type: str  # e.g., "agents_md_patch", "skill_add", "file_override"
    content: str | dict[str, Any]  # The override content
    metadata: dict[str, Any] = Field(default_factory=dict)  # Additional context


class OverrideApplication(BaseModel):
    """Result of applying context overrides in task app.

    Reports whether overrides were successfully applied and provides
    structured error information for debugging and optimizer feedback.
    """

    requested: bool = False  # Were overrides requested?
    bundle_id: str | None = None  # Stable ID for the override bundle
    applied: bool = False  # Were all overrides successfully applied?
    applied_components: list[str] = Field(default_factory=list)  # Which overrides succeeded
    errors: list[dict[str, Any]] = Field(default_factory=list)  # Structured errors
    warnings: list[str] = Field(default_factory=list)  # Non-fatal issues
    applied_manifest_ref: str | None = None  # Reference to stored manifest
    duration_ms: int | None = None  # Time taken to apply overrides
