"""Deprecated alias for :mod:`synth_ai.sdk.research`.

# See: unify_sdk_layering.md

Research implementation moved to ``synth_ai/sdk/research`` so that ``core/``
means one thing -- shared plumbing -- instead of two. ``SynthClient().research``
is unchanged and remains the supported entrypoint; only module paths moved.

    synth_ai.core.research.<anything>  ->  synth_ai.sdk.research.<anything>

This is a finder, not 178 stub files. A stub per module would be 178 files to
write, keep in sync, and delete, and every one of them a place for the alias to
drift from what it aliases. The finder cannot drift: it resolves the real module
at import time and registers it under both names, so the two are the *same*
module object, not two copies of it.

    >>> from synth_ai.core.research.contracts.status import SwarmStatus  # warns
    >>> from synth_ai.sdk.research.contracts.status import SwarmStatus
    >>> # same class, same identity -- isinstance across the two paths works

Scheduled for deletion once the sibling repos are clean (phase 5 of the plan),
at which point `core/research` joins the retired-tree absence assertions.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any, Sequence

_OLD = "synth_ai.core.research"
_NEW = "synth_ai.sdk.research"

# Warn once per aliased module rather than once per import statement: a single
# deprecated module imported from thirty call sites is one thing to fix, and
# thirty identical warnings buries the other twenty-nine.
_warned: set[str] = set()


def _warn_once(name: str) -> None:
    if name in _warned:
        return
    _warned.add(name)
    warnings.warn(
        f"{name} is deprecated; import {name.replace(_OLD, _NEW, 1)} instead. "
        "Research moved out of core/ so that core/ means shared plumbing only. "
        "SynthClient().research is unchanged.",
        DeprecationWarning,
        stacklevel=3,
    )


# CPython stamps the spec being loaded onto whatever `create_module` returns,
# and what it returns here is a module that is already fully imported. Left
# alone, the real module ends up carrying the alias's origin-less spec, which
# costs it `importlib.resources`: `files()` yields an orphan path, so reading
# any packaged data file raises FileNotFoundError for a process that touched the
# deprecated path first. Snapshot these before the stamp, restore after.
_IDENTITY_ATTRS = (
    "__spec__",
    "__loader__",
    "__name__",
    "__package__",
    "__path__",
    "__file__",
    "__cached__",
)


class _AliasLoader(Loader):
    """Load the relocated module and publish it under the deprecated name."""

    def __init__(self, new_name: str) -> None:
        self._new_name = new_name
        self._identity: dict[str, Any] = {}

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        # Return the *real* module so both names share one object. Anything else
        # gives two module instances, two sets of class objects, and isinstance
        # checks that fail depending on which path the caller imported.
        module = importlib.import_module(self._new_name)
        self._identity = {
            attr: getattr(module, attr) for attr in _IDENTITY_ATTRS if hasattr(module, attr)
        }
        return module

    def exec_module(self, module: ModuleType) -> None:
        # Already executed under its real name by create_module; all that is
        # left is to undo the identity stamp.
        for attr in _IDENTITY_ATTRS:
            if attr in self._identity:
                setattr(module, attr, self._identity[attr])
            elif hasattr(module, attr):
                delattr(module, attr)


class _ResearchAliasFinder(MetaPathFinder):
    """Map `synth_ai.core.research[.x.y]` onto `synth_ai.sdk.research[.x.y]`."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname != _OLD and not fullname.startswith(f"{_OLD}."):
            return None
        new_name = fullname.replace(_OLD, _NEW, 1)
        if new_name == fullname:
            return None
        _warn_once(fullname)
        spec = ModuleSpec(fullname, _AliasLoader(new_name))
        # Mark it a package when its target is one, so `from ... import submodule`
        # keeps working through the alias.
        target_module = importlib.import_module(new_name)
        if hasattr(target_module, "__path__"):
            spec.submodule_search_locations = list(target_module.__path__)
        return spec


def _install() -> None:
    if any(isinstance(finder, _ResearchAliasFinder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _ResearchAliasFinder())


_install()


def __getattr__(name: str) -> Any:
    """Forward attribute access on the alias package itself."""
    if name.startswith("__"):
        raise AttributeError(name)
    _warn_once(f"{_OLD}.{name}")
    return getattr(importlib.import_module(_NEW), name)


def __dir__() -> list[str]:
    return dir(importlib.import_module(_NEW))
