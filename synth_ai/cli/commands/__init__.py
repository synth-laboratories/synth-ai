"""
Structured CLI command implementations.

Each subpackage under this namespace provides the core command entrypoints,
validation helpers, and error types for a top-level CLI command (e.g. train,
eval, deploy).
"""

from __future__ import annotations

__all__ = [
    "train",
    "eval",
    "filter",
    "deploy",
    "status",
    "smoke",
]
