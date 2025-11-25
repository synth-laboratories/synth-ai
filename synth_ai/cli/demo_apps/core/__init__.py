"""
Compatibility layer exposing the legacy demo helpers.

Historically these utilities lived in ``synth_ai.cli.demo_apps.core`` as a module.
Upstream refactors moved the implementation under
``synth_ai.cli.demo_apps.demo_task_apps.core``.  Several call sites (including the new
vision tests) still import the older path, so we re-export everything here.
"""

from __future__ import annotations

from synth_ai.cli.demo_apps.demo_task_apps import core as _demo_core

__all__ = [
    name
    for name in dir(_demo_core)
    if not name.startswith("_")
]

globals().update({name: getattr(_demo_core, name) for name in __all__})


def __getattr__(name: str):
    if name in __all__:
        value = getattr(_demo_core, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
