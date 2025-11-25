"""Namespace for demo task apps (math, etc.)."""

import contextlib

# Ensure registry entries are loaded for CLI discovery.
with contextlib.suppress(Exception):  # pragma: no cover - optional on downstream installs
    from .math import task_app_entry  # noqa: F401
