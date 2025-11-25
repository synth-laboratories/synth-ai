"""Dynamic loader for tracing storage components.

This avoids hard dependencies on the tracing_v3 storage package during import
time, which keeps CLI modules usable in constrained environments while still
allowing type checkers to resolve the symbols dynamically.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, cast


def load_storage() -> tuple[Any, Any]:
    """Return (create_storage, StorageConfig) from tracing_v3.storage."""
    storage_module = cast(Any, importlib.import_module("synth_ai.core.tracing_v3.storage"))
    create_storage = cast(Callable[..., Any], storage_module.create_storage)
    storage_config = storage_module.StorageConfig
    return create_storage, storage_config
