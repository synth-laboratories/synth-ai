"""Validation helpers for task app responses.

Provides client-side validation for artifacts, context snapshots, and other
rollout response components to ensure they meet size and complexity constraints.
"""

import json
from typing import Any

from synth_ai.data.artifacts import Artifact, ContextOverride

# Size limits (client-side hard constraints)
MAX_INLINE_ARTIFACT_BYTES = 64 * 1024  # 64 KB per artifact
MAX_TOTAL_INLINE_ARTIFACTS_BYTES = 256 * 1024  # 256 KB total
MAX_ARTIFACTS_PER_ROLLOUT = 10
MAX_ARTIFACT_METADATA_BYTES = 16 * 1024  # 16 KB
MAX_ARTIFACT_CONTENT_TYPE_LENGTH = 128
MAX_CONTEXT_SNAPSHOT_BYTES = 1 * 1024 * 1024  # 1 MB
MAX_CONTEXT_OVERRIDES_PER_ROLLOUT = 20


def validate_artifact_size(artifact: Artifact, max_bytes: int = MAX_INLINE_ARTIFACT_BYTES) -> None:
    """Validate single artifact size.

    Args:
        artifact: Artifact to validate
        max_bytes: Maximum allowed size in bytes (default: 64 KB)

    Raises:
        ValueError: If artifact exceeds size limit
    """
    artifact.validate_size(max_bytes=max_bytes)


def validate_artifacts_list(artifacts: list[Artifact]) -> None:
    """Validate total artifact payload size and count.

    Args:
        artifacts: List of artifacts to validate

    Raises:
        ValueError: If artifacts exceed size or count limits
    """
    if len(artifacts) > MAX_ARTIFACTS_PER_ROLLOUT:
        raise ValueError(f"Too many artifacts: {len(artifacts)} > {MAX_ARTIFACTS_PER_ROLLOUT}")

    # Validate total size
    total_size = 0
    for artifact in artifacts:
        # Calculate size
        if isinstance(artifact.content, str):
            size = len(artifact.content.encode("utf-8"))
        elif isinstance(artifact.content, dict):
            size = len(json.dumps(artifact.content).encode("utf-8"))
        else:
            import sys

            size = sys.getsizeof(artifact.content)

        total_size += size

        # Validate content_type length
        if len(artifact.content_type) > MAX_ARTIFACT_CONTENT_TYPE_LENGTH:
            raise ValueError(
                f"Artifact content_type too long: {len(artifact.content_type)} > "
                f"{MAX_ARTIFACT_CONTENT_TYPE_LENGTH}"
            )

        # Validate metadata size
        if artifact.metadata:
            metadata_size = len(json.dumps(artifact.metadata).encode("utf-8"))
            if metadata_size > MAX_ARTIFACT_METADATA_BYTES:
                raise ValueError(
                    f"Artifact metadata too large: {metadata_size} > {MAX_ARTIFACT_METADATA_BYTES}"
                )

    if total_size > MAX_TOTAL_INLINE_ARTIFACTS_BYTES:
        raise ValueError(
            f"Total artifacts size {total_size} exceeds {MAX_TOTAL_INLINE_ARTIFACTS_BYTES} bytes"
        )


def validate_context_overrides(overrides: list[ContextOverride]) -> None:
    """Validate context overrides list.

    Validates:
    - Total count of overrides
    - Total size across all override content (file_artifacts, preflight_script, env_vars)
    - Individual file artifact count and sizes

    Args:
        overrides: List of context overrides to validate

    Raises:
        ValueError: If overrides exceed count or size limits
    """
    if len(overrides) > MAX_CONTEXT_OVERRIDES_PER_ROLLOUT:
        raise ValueError(
            f"Too many context overrides: {len(overrides)} > {MAX_CONTEXT_OVERRIDES_PER_ROLLOUT}"
        )

    # Validate total size using the new ContextOverride.size_bytes() method
    total_size = 0
    total_files = 0
    for override in overrides:
        # Use the size_bytes() helper on the new model
        total_size += override.size_bytes()
        total_files += len(override.file_artifacts)

    # Check total size
    if total_size > MAX_CONTEXT_SNAPSHOT_BYTES:
        raise ValueError(
            f"Total context override size {total_size} exceeds {MAX_CONTEXT_SNAPSHOT_BYTES} bytes"
        )

    # Check total file count (reasonable limit)
    max_files = 50
    if total_files > max_files:
        raise ValueError(
            f"Too many file artifacts across overrides: {total_files} > {max_files}"
        )


def validate_context_snapshot(snapshot_data: dict[str, Any]) -> None:
    """Validate context snapshot size.

    Args:
        snapshot_data: Context snapshot data to validate

    Raises:
        ValueError: If snapshot exceeds size limit
    """
    size = len(json.dumps(snapshot_data).encode("utf-8"))
    if size > MAX_CONTEXT_SNAPSHOT_BYTES:
        raise ValueError(f"Context snapshot size {size} exceeds {MAX_CONTEXT_SNAPSHOT_BYTES} bytes")


__all__ = [
    "MAX_INLINE_ARTIFACT_BYTES",
    "MAX_TOTAL_INLINE_ARTIFACTS_BYTES",
    "MAX_ARTIFACTS_PER_ROLLOUT",
    "MAX_ARTIFACT_METADATA_BYTES",
    "MAX_ARTIFACT_CONTENT_TYPE_LENGTH",
    "MAX_CONTEXT_SNAPSHOT_BYTES",
    "MAX_CONTEXT_OVERRIDES_PER_ROLLOUT",
    "validate_artifact_size",
    "validate_artifacts_list",
    "validate_context_overrides",
    "validate_context_snapshot",
]
