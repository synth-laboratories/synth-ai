"""Backward-compatible server re-export for synth_ai.sdk.task."""

from __future__ import annotations

from synth_ai.sdk.localapi._impl.server import RubricBundle

__all__ = ["RubricBundle"]
