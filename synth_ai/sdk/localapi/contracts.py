"""LocalAPI contract re-exports.

Prefer this module over synth_ai.sdk.task.contracts.* moving forward.
"""

from __future__ import annotations

from synth_ai.sdk.task.contracts import LocalAPIEndpoints, TaskAppEndpoints

__all__ = ["LocalAPIEndpoints", "TaskAppEndpoints"]
