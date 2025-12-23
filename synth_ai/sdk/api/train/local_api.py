"""LocalAPI health helpers.

Prefer this module over synth_ai.sdk.api.train.task_app for LocalAPI naming.
"""

from __future__ import annotations

from synth_ai.sdk.api.train.task_app import LocalAPIHealth, check_local_api_health

__all__ = ["LocalAPIHealth", "check_local_api_health"]
