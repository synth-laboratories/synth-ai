"""Integration-oriented SDK namespace."""

from __future__ import annotations

from synth_ai.managed_research.sdk._base import _ClientNamespace


class IntegrationsAPI(_ClientNamespace):
    """Reserved namespace for first-party integration helpers."""


__all__ = ["IntegrationsAPI"]
