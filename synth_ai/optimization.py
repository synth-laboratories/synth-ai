"""Canonical optimization namespace.

# See: specs/sdk_logic.md
"""

from synth_ai.sdk.optimization import (
    PolicyOptimizationOfflineJob as OfflineJob,
)
from synth_ai.sdk.optimization import (
    PolicyOptimizationOnlineSession as OnlineSession,
)
from synth_ai.sdk.optimization import (
    PolicyOptimizationSystem as System,
)

__all__ = [
    "OfflineJob",
    "OnlineSession",
    "System",
]
