"""Backward-compatible task contracts re-export."""

from __future__ import annotations

from synth_ai.sdk.localapi.contracts import *  # noqa: F403
from synth_ai.sdk.localapi.contracts import __all__ as _contracts_all

__all__ = list(_contracts_all)
