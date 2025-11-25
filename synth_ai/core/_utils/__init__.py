"""
Compatibility shims for legacy `synth_ai.core._utils.*` imports.

These modules re-export the public symbols so existing downstream code
continues to work without modification.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Modules that can be eagerly loaded (no circular deps)
_EAGER_MAPPING = {
    "base_url": "synth_ai.core.env",
    "http": "synth_ai.core.http",
    "task_app_state": "synth_ai.core.task_app_state",
    "user_config": "synth_ai.core.user_config",
}

# Modules that must be lazily loaded (have CLI deps)
_LAZY_MAPPING = {
    "prompts": "synth_ai.cli.lib.prompt_args",
}

__all__ = sorted(list(_EAGER_MAPPING.keys()) + list(_LAZY_MAPPING.keys()))


def __getattr__(name: str) -> Any:
    if name in _EAGER_MAPPING:
        module = import_module(_EAGER_MAPPING[name])
        globals()[name] = module
        return module
    if name in _LAZY_MAPPING:
        module = import_module(_LAZY_MAPPING[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'synth_ai.core._utils' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


# Eagerly load non-circular modules
for _name in _EAGER_MAPPING:
    try:
        _module = import_module(_EAGER_MAPPING[_name])
        globals()[_name] = _module
        if hasattr(_module, "__all__"):
            for _attr in _module.__all__:
                globals().setdefault(_attr, getattr(_module, _attr))
    except ImportError:
        pass
