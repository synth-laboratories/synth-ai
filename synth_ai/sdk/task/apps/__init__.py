"""Registry for Task Apps exposed via the shared FastAPI harness."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from ..server import TaskAppConfig


@dataclass(slots=True)
class ModalDeploymentConfig:
    """Modal deployment defaults for a task app."""

    app_name: str
    python_version: str = "3.11"
    pip_packages: Sequence[str] = field(default_factory=tuple)
    apt_packages: Sequence[str] = field(default_factory=tuple)
    extra_local_dirs: Sequence[tuple[str, str]] = field(default_factory=tuple)
    secret_names: Sequence[str] = field(default_factory=tuple)
    volume_mounts: Sequence[tuple[str, str]] = field(default_factory=tuple)
    env_vars: dict[str, str] = field(default_factory=dict)
    timeout: int = 600
    memory: int = 4096
    cpu: float = 2.0
    min_containers: int = 1
    max_containers: int = 4


@dataclass(slots=True)
class TaskAppEntry:
    """Metadata describing a registered task app."""

    app_id: str
    description: str
    config_factory: Callable[[], TaskAppConfig]
    aliases: Sequence[str] = field(default_factory=tuple)
    env_files: Sequence[str] = field(default_factory=tuple)
    modal: ModalDeploymentConfig | None = None


class TaskAppRegistry:
    """In-memory registry of known task apps."""

    def __init__(self) -> None:
        self._entries: dict[str, TaskAppEntry] = {}
        self._alias_to_id: dict[str, str] = {}

    def register(self, entry: TaskAppEntry) -> None:
        if entry.app_id in self._entries:
            # Allow idempotent registration when modules are imported multiple times.
            return
        self._entries[entry.app_id] = entry
        for alias in entry.aliases:
            existing = self._alias_to_id.get(alias)
            if existing and existing != entry.app_id:
                raise ValueError(f"Alias already registered: {alias}")
            self._alias_to_id[alias] = entry.app_id

    def get(self, app_id: str) -> TaskAppEntry:
        resolved = self._alias_to_id.get(app_id, app_id)
        if resolved not in self._entries:
            raise KeyError(f"Unknown task app id: {app_id}")
        return self._entries[resolved]

    def list(self) -> List[TaskAppEntry]:
        return sorted(self._entries.values(), key=lambda entry: entry.app_id)

    def __iter__(self) -> Iterable[TaskAppEntry]:
        return iter(self.list())

    def clear(self) -> None:
        """Clear all registered task apps."""
        self._entries.clear()
        self._alias_to_id.clear()


registry = TaskAppRegistry()


def register_task_app(*, entry: TaskAppEntry) -> None:
    registry.register(entry)


def discover_task_apps_from_cwd() -> None:
    """Discover and register task apps from the current working directory and subdirectories."""
    cwd = Path.cwd()

    # Look for task app files in common patterns
    patterns = [
        "**/task_app/*.py",
        "**/task_apps/*.py",
        "**/*_task_app.py",
        "**/grpo_crafter.py",
        "**/math_single_step.py",
    ]

    discovered_files = []
    for pattern in patterns:
        discovered_files.extend(cwd.glob(pattern))

    # Add current directory to Python path temporarily
    original_path = sys.path.copy()
    try:
        sys.path.insert(0, str(cwd))

        for file_path in discovered_files:
            if file_path.name.startswith("__"):
                continue

            # Convert file path to module name
            relative_path = file_path.relative_to(cwd)
            module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
            module_name = ".".join(module_parts)

            try:
                # Import the module to trigger registration
                importlib.import_module(module_name)
            except Exception:
                # Silently skip modules that can't be imported
                # This allows for graceful handling of missing dependencies
                continue

    finally:
        sys.path[:] = original_path


# Note: Task apps are now discovered dynamically by the CLI, not auto-registered
# This allows for better separation between SDK and example-specific implementations
