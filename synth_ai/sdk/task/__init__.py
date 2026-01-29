"""Backward-compatible task SDK re-exports.

This module preserves legacy imports that referenced synth_ai.sdk.task.* by
re-exporting the Local API contracts and helpers.
"""

from __future__ import annotations

from synth_ai.sdk.localapi._impl import run_server_background
from synth_ai.sdk.localapi.contracts import *  # noqa: F403
from synth_ai.sdk.localapi.contracts import __all__ as _contracts_all

__all__ = list(_contracts_all) + ["run_server_background"]
