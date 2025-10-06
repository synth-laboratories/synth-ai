"""Namespace for demo task apps (math, etc.)."""

# Ensure registry entries are loaded for CLI discovery.
try:  # pragma: no cover - optional on downstream installs
    from .math import task_app_entry  # noqa: F401
except Exception:
    pass
