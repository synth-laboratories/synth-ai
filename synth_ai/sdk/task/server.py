"""Backward-compatible server re-export for synth_ai.sdk.task."""

from __future__ import annotations

from synth_ai.sdk.container._impl.server import RubricBundle, run_server_background

__all__ = ["RubricBundle", "run_server_background"]
