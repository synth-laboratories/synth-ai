"""Validation helpers for task app responses.

Provides client-side validation for artifacts, context snapshots, and other
rollout response components to ensure they meet size and complexity constraints.
"""

from __future__ import annotations

from typing import Any

import synth_ai_py

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
    """Validate single artifact size."""

    synth_ai_py.localapi_validate_artifact_size(artifact, max_bytes)


def validate_artifacts_list(artifacts: list[Artifact]) -> None:
    """Validate total artifact payload size and count."""

    synth_ai_py.localapi_validate_artifacts_list(artifacts)


def validate_context_overrides(overrides: list[ContextOverride]) -> None:
    """Validate context overrides list."""

    synth_ai_py.localapi_validate_context_overrides(overrides)


def validate_context_snapshot(snapshot_data: dict[str, Any]) -> None:
    """Validate context snapshot size."""

    synth_ai_py.localapi_validate_context_snapshot(snapshot_data)


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
