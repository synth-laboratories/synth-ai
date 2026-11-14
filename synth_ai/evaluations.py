"""Evaluation inputs and evaluations over the generated public operations."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import JsonValue

from ._generated.models import (
    CreateEvaluation,
    EvaluationDataset,
    EvaluationEvaluationRequest,
    EvaluationInput,
    EvaluationQuote,
    EvaluationResponse,
    EvaluationRevision,
    EvaluationRevisionList,
    EvaluationRolloutPage,
    EvaluationRolloutSummary,
    EvaluationTaskInstance,
)
from ._transport import Transport

EvaluationInputKind = Literal["task", "pool", "policy", "evaluator", "dataset"]
TERMINAL_EVALUATION_STATUSES = frozenset({"completed", "failed", "cancelled"})


def dataset(
    name: str, instances: Sequence[EvaluationTaskInstance | Mapping[str, JsonValue]]
) -> EvaluationDataset:
    """Validate a seed dataset locally: the contract's 1-1024 instances, id and seed ranges.

    Instance ids must also be unique, because an evaluation selects instances by id.
    """
    rows = [
        item
        if isinstance(item, EvaluationTaskInstance)
        else EvaluationTaskInstance.model_validate(dict(item))
        for item in instances
    ]
    seen: set[str] = set()
    for row in rows:
        if row.id in seen:
            raise ValueError("dataset instance ids must be unique: " + row.id)
        seen.add(row.id)
    return EvaluationDataset(name=name, instances=rows)


class AsyncEvaluations:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def register(
        self,
        kind: EvaluationInputKind,
        payload: EvaluationInput,
        *,
        idempotency_key: str | None = None,
    ) -> EvaluationRevision:
        """Register an immutable input; the result's `revision_id` names it."""
        return await self._transport.request(
            "registerEvaluationInput",
            EvaluationRevision,
            path={"kind": kind},
            body=payload,
            idempotency_key=idempotency_key,
        )

    async def register_dataset(
        self,
        name: str,
        instances: Sequence[EvaluationTaskInstance | Mapping[str, JsonValue]],
        *,
        idempotency_key: str | None = None,
    ) -> EvaluationRevision:
        """Register `{"id", "seed", "inputs"}` instances; invalid datasets send nothing."""
        return await self.register(
            "dataset", dataset(name, instances), idempotency_key=idempotency_key
        )

    async def list_inputs(
        self, kind: EvaluationInputKind, *, name: str | None = None
    ) -> EvaluationRevisionList:
        return await self._transport.request(
            "listEvaluationInputs",
            EvaluationRevisionList,
            path={"kind": kind},
            query={"name": name},
        )

    async def get_input(self, kind: EvaluationInputKind, revision_id: str) -> EvaluationRevision:
        return await self._transport.request(
            "getEvaluationInput",
            EvaluationRevision,
            path={"kind": kind, "revision_id": revision_id},
        )

    async def preflight(
        self,
        request: EvaluationEvaluationRequest,
        *,
        maximum_charge_micros: int,
        idempotency_key: str | None = None,
    ) -> EvaluationQuote:
        """Quote jobs, concurrency and the maximum charge without reserving budget."""
        return await self._transport.request(
            "preflightEvaluation",
            EvaluationQuote,
            body=CreateEvaluation(request=request, maximum_charge_micros=maximum_charge_micros),
            idempotency_key=idempotency_key,
        )

    async def create(
        self,
        request: EvaluationEvaluationRequest,
        *,
        maximum_charge_micros: int,
        idempotency_key: str | None = None,
    ) -> EvaluationResponse:
        """Reserve the budget and start; the same key returns the existing evaluation."""
        return await self._transport.request(
            "createEvaluation",
            EvaluationResponse,
            body=CreateEvaluation(request=request, maximum_charge_micros=maximum_charge_micros),
            idempotency_key=idempotency_key,
        )

    async def get(self, evaluation_id: str) -> EvaluationResponse:
        return await self._transport.request(
            "getEvaluation", EvaluationResponse, path={"evaluation_id": evaluation_id}
        )

    async def rollouts(
        self,
        evaluation_id: str,
        *,
        cursor: str | None = None,
        instance_ids: Sequence[str] | None = None,
    ) -> EvaluationRolloutPage:
        """Read one bounded page of attested rollout reports; follow next_cursor."""
        return await self._transport.request(
            "listEvaluationRollouts",
            EvaluationRolloutPage,
            path={"evaluation_id": evaluation_id},
            query={
                "cursor": cursor,
                "instance_ids": list(instance_ids) if instance_ids is not None else None,
            },
        )

    async def rollout_summary(
        self,
        evaluation_id: str,
        *,
        cursor: str | None = None,
        instance_ids: Sequence[str] | None = None,
    ) -> EvaluationRolloutSummary:
        """Read one compact row per instance with outcome counts; follow next_cursor."""
        return await self._transport.request(
            "listEvaluationRollouts",
            EvaluationRolloutSummary,
            path={"evaluation_id": evaluation_id},
            query={
                "cursor": cursor,
                "view": "summary",
                "instance_ids": list(instance_ids) if instance_ids is not None else None,
            },
        )

    async def cancel(
        self, evaluation_id: str, *, idempotency_key: str | None = None
    ) -> EvaluationResponse:
        return await self._transport.request(
            "cancelEvaluation",
            EvaluationResponse,
            path={"evaluation_id": evaluation_id},
            idempotency_key=idempotency_key,
        )

    async def wait(
        self, evaluation_id: str, *, timeout: float = 600, poll_interval: float = 2
    ) -> EvaluationResponse:
        """Poll until completed, failed or cancelled; raises TimeoutError at the deadline."""
        if (
            not math.isfinite(timeout)
            or not math.isfinite(poll_interval)
            or timeout <= 0
            or poll_interval <= 0
        ):
            raise ValueError("timeout and poll_interval must be finite and positive")
        deadline = time.monotonic() + timeout
        async with asyncio.timeout(timeout):
            while True:
                result = await self.get(evaluation_id)
                if result.status in TERMINAL_EVALUATION_STATUSES:
                    return result
                await asyncio.sleep(min(poll_interval, max(0, deadline - time.monotonic())))
