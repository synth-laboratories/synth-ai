"""
Compatibility shims for legacy `synth_ai._utils.*` imports.

The modern codebase exposes these helpers under ``synth_ai.utils``.  These
modules re-export the public symbols so existing downstream code (and our own
older examples/tests) continue to work without modification.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

_MAPPING = {
    "base_url": "synth_ai.utils.base_url",
    "http": "synth_ai.utils.http",
    "prompts": "synth_ai.utils.prompts",
    "task_app_state": "synth_ai.utils.task_app_state",
    "user_config": "synth_ai.utils.user_config",
}

__all__ = sorted(_MAPPING.keys())


def __getattr__(name: str) -> ModuleType:
    target = _MAPPING.get(name)
    if not target:
        raise AttributeError(f"module 'synth_ai._utils' has no attribute '{name}'")
    module = import_module(target)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def _export(module_name: str) -> None:
    module = import_module(_MAPPING[module_name])
    globals().setdefault(module_name, module)
    if hasattr(module, "__all__"):
        for attr in module.__all__:  # type: ignore[attr-defined]
            globals().setdefault(attr, getattr(module, attr))


for _name in __all__:
    _export(_name)
