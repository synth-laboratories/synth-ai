"""Project output namespace for the flatter noun-first SDK surface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, List

from synth_ai.core.research._legacy.models.run_launch import (
    Output,
    OutputKind,
    OutputListRequest,
    RunRef,
)
from synth_ai.core.research._legacy.sdk._base import _ClientNamespace


class OutputsAPI(_ClientNamespace):
    def list(self, project_id: str) -> List[dict[str, Any]]:
        return self._client.list_project_outputs(project_id)

    def list_outputs(self, request: OutputListRequest) -> List[Output]:
        if request.run_ref is not None:
            return self.for_run(
                request.run_ref,
                kinds=request.kinds,
                limit=request.limit,
            )
        assert request.project_id is not None
        payloads = self._client.list_project_outputs(request.project_id)
        outputs = [Output.from_wire(payload) for payload in payloads]
        return _filter_outputs(outputs, kinds=request.kinds, limit=request.limit)

    def for_run(
        self,
        run_ref: RunRef,
        *,
        kinds: Sequence[OutputKind] = (),
        limit: int | None = None,
    ) -> List[Output]:
        payloads = self._client.list_run_work_products(run_ref.project_id, run_ref.run_id)
        outputs = [Output.from_wire(payload, run_ref=run_ref) for payload in payloads]
        return _filter_outputs(outputs, kinds=kinds, limit=limit)


def _filter_outputs(
    outputs: list[Output],
    *,
    kinds: Sequence[OutputKind] = (),
    limit: int | None,
) -> List[Output]:
    filtered = [output for output in outputs if not kinds or output.kind in kinds]
    if limit is None:
        return filtered
    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    return filtered[:limit]


__all__ = ["OutputsAPI"]
