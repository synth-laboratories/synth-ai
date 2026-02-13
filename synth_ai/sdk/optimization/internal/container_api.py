"""Container health helpers.

Prefer this module over synth_ai.sdk.optimization.internal.container_app for Container naming.
"""

from __future__ import annotations

from synth_ai.sdk.optimization.internal.container_app import ContainerHealth, check_container_health

__all__ = ["ContainerHealth", "check_container_health"]
