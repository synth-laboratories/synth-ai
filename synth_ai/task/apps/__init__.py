from __future__ import annotations

"""Registry for Task Apps exposed via the shared FastAPI harness."""

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Sequence

from ..server import TaskAppConfig


@dataclass(slots=True)
class ModalDeploymentConfig:
    """Modal deployment defaults for a task app."""

    app_name: str
    python_version: str = "3.11"
    pip_packages: Sequence[str] = field(default_factory=tuple)
    extra_local_dirs: Sequence[tuple[str, str]] = field(default_factory=tuple)
    secret_names: Sequence[str] = field(default_factory=tuple)
    volume_mounts: Sequence[tuple[str, str]] = field(default_factory=tuple)
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
        self._entries: Dict[str, TaskAppEntry] = {}
        self._alias_to_id: Dict[str, str] = {}

    def register(self, entry: TaskAppEntry) -> None:
        if entry.app_id in self._entries:
            raise ValueError(f"Task app already registered: {entry.app_id}")
        self._entries[entry.app_id] = entry
        for alias in entry.aliases:
            if alias in self._alias_to_id:
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


registry = TaskAppRegistry()


def register_task_app(*, entry: TaskAppEntry) -> None:
    registry.register(entry)



# Register built-in task apps
try:
    from . import grpo_crafter  # noqa: F401
except Exception:
    # Defer import errors so CLI can report missing deps gracefully
    pass

try:
    from . import math_single_step  # noqa: F401
except Exception:
    # Defer import errors so CLI can report missing deps gracefully
    pass
