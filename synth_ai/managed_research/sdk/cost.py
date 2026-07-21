"""Per-run cost summary SDK namespace.

Wraps ``GET /smr/swarms/{run_id}/cost_summary`` which aggregates
``smr_usage_facts`` by ``meter_kind`` (tinker training, inference tokens,
sandbox seconds, ...).
"""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.sdk._base import _ClientNamespace


class RunCostAPI(_ClientNamespace):
    def summary(self, run_id: str) -> dict[str, Any]:
        return self._client._request_json("GET", f"/smr/swarms/{run_id}/cost_summary")


__all__ = ["RunCostAPI"]
