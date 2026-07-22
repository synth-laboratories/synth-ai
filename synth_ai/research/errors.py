"""Stable Research failures with lazy migration aliases."""

from __future__ import annotations

from synth_ai.core.research import errors as _errors
from synth_ai.core.research.errors import *  # noqa: F403
from synth_ai.core.research.errors import __all__


def __getattr__(name: str) -> object:
    value = getattr(_errors, name)
    globals()[name] = value
    return value
