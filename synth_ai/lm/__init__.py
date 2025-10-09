"""Deprecated shim forwarding to synth_ai.v0.lm."""

import importlib as _importlib
import pkgutil as _pkgutil
import sys as _sys
from pathlib import Path as _Path

_TARGET_PREFIX = "synth_ai.v0.lm"
_ALIAS_PREFIX = __name__

_alias_path = _Path(__file__).resolve().parents[1] / "v0" / "lm"
__path__ = [str(_alias_path)]  # type: ignore[assignment]

_pkg = _importlib.import_module(_TARGET_PREFIX)
_sys.modules[_ALIAS_PREFIX] = _pkg

for _finder, _name, _ispkg in _pkgutil.walk_packages(_pkg.__path__, prefix=_TARGET_PREFIX + "."):  # type: ignore[attr-defined]
    try:
        _module = _importlib.import_module(_name)
    except Exception:  # pragma: no cover - best effort
        continue
    _alias = _ALIAS_PREFIX + _name[len(_TARGET_PREFIX) :]
    _sys.modules[_alias] = _module

del _finder, _name, _ispkg, _module, _alias, _TARGET_PREFIX, _ALIAS_PREFIX, _alias_path
