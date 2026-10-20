"""Canonical customer environment names. See SDK_0230_MCP.md."""

from __future__ import annotations

import os
import warnings
from threading import Lock


class LegacyEnvironmentWarning(FutureWarning):
    """An accepted compatibility name should migrate to the canonical one."""


_WARNED_NAMES: set[str] = set()
_WARNING_LOCK = Lock()


def environment_value(canonical: str, *legacy: str, default: str | None = None) -> str | None:
    """Prefer the canonical setting; never log its value or guess between aliases."""
    if canonical in os.environ:
        return os.environ[canonical]
    candidates = [(name, os.environ[name]) for name in legacy if name in os.environ]
    if not candidates:
        return default
    if len({value.strip() for _, value in candidates}) > 1:
        raise ValueError(
            f"Conflicting deprecated environment variables; set {canonical} explicitly"
        )
    for name, _ in candidates:
        with _WARNING_LOCK:
            first_use = name not in _WARNED_NAMES
            _WARNED_NAMES.add(name)
        if first_use:
            warnings.warn(
                f"{name} is deprecated and will be removed in 0.25.0; use {canonical}.",
                LegacyEnvironmentWarning,
                stacklevel=3,
            )
    return candidates[0][1]
