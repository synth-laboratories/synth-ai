"""Per-run cost summary SDK namespace.

Wraps ``GET /smr/runs/{run_id}/cost_summary`` which aggregates
``smr_usage_facts`` by ``meter_kind`` (tinker training, inference tokens,
sandbox seconds, ...).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synth_ai.managed_research.sdk._base import _ClientNamespace


class RunCostAPI(_ClientNamespace):
    def summary(self, run_id: str) -> dict[str, Any]:
        return self._client._request_json("GET", f"/smr/runs/{run_id}/cost_summary")

    def report_tinker_training_usage(
        self,
        run_id: str,
        *,
        actual_cost_usd: float | None = None,
        estimated_cost_usd: float | None = None,
        model: str | None = None,
        task_id: str | None = None,
        idempotency_key: str | None = None,
        provider_result_id: str | None = None,
        request_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = {
            "provider": "tinker",
            "operation_kind": "training_job",
            "meter_kind": "tinker_training_job",
            "quantity": 1,
            "quantity_unit": "job",
            "actual_cost_usd": actual_cost_usd,
            "estimated_cost_usd": estimated_cost_usd,
            "model": model,
            "task_id": task_id,
            "idempotency_key": idempotency_key,
            "provider_result_id": provider_result_id,
            "request_id": request_id,
            "metadata": dict(metadata or {}),
        }
        return self._client._request_json(
            "POST", f"/smr/internal/runs/{run_id}/provider-usage", json_body=body
        )


__all__ = ["RunCostAPI"]
