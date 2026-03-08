"""Internal implementation modules for optimization.

This package remains importable for compatibility tests and local tooling.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["graph_evolve_streaming", "prompt_learning_streaming"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
