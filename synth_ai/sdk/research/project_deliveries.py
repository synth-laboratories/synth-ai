"""Accepted-decision-only Project Delivery consumers."""

from __future__ import annotations

import asyncio
import time
from typing import cast

from synth_ai.core.contracts.json_value import JsonObject
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.sdk.research.contracts.common import ProjectId
from synth_ai.sdk.research.contracts.deliveries import (
    DraftDeliveryAuthorityResponse,
    EnsureDraftDeliveryRequest,
)
from synth_ai.sdk.research.operations import research_operation


def _request(
    operation_id: str,
    path: str,
    *,
    body: JsonObject | None = None,
) -> HttpRequest:
    return HttpRequest(research_operation(operation_id), path, body=body)


def _validate_delivery(
    response: DraftDeliveryAuthorityResponse,
    *,
    delivery_id: str | None = None,
    decision_id: str | None = None,
) -> DraftDeliveryAuthorityResponse:
    delivery = response.delivery
    if delivery_id is not None and delivery.delivery_id != delivery_id:
        raise ValueError("Delivery response identity drifted")
    if decision_id is not None and delivery.decision_id != decision_id:
        raise ValueError("Delivery response crossed its accepted decision boundary")
    return response


class ProjectDeliveriesAPI:
    """One accepted organic decision may ensure one open draft Delivery."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def ensure_draft(
        self,
        project_id: ProjectId,
        request: EnsureDraftDeliveryRequest,
    ) -> DraftDeliveryAuthorityResponse:
        """Idempotently ensure an accepted-decision-bound draft Delivery."""
        response = DraftDeliveryAuthorityResponse.from_wire(
            self._transport.execute(
                _request(
                    "ensure_project_draft_delivery",
                    f"/smr/projects/{project_id}/deliveries:ensure-draft",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        return _validate_delivery(response, decision_id=request.decision_id)

    def get(
        self,
        project_id: ProjectId,
        delivery_id: str,
        *,
        decision_id: str,
    ) -> DraftDeliveryAuthorityResponse:
        """Read one draft Delivery while retaining the accepted decision join."""
        response = DraftDeliveryAuthorityResponse.from_wire(
            self._transport.execute(
                _request(
                    "get_project_draft_delivery",
                    f"/smr/projects/{project_id}/deliveries/{delivery_id}",
                )
            )
        )
        return _validate_delivery(
            response,
            delivery_id=delivery_id,
            decision_id=decision_id,
        )

    def wait(
        self,
        project_id: ProjectId,
        delivery_id: str,
        *,
        decision_id: str,
        timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 2.0,
    ) -> DraftDeliveryAuthorityResponse:
        """Boundedly poll until the Delivery reaches a terminal state."""
        if timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("Delivery wait bounds must be positive")
        deadline = time.monotonic() + timeout_seconds
        while True:
            response = self.get(
                project_id,
                delivery_id,
                decision_id=decision_id,
            )
            if response.delivery.state.terminal:
                return response
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Delivery {delivery_id} did not terminalize within {timeout_seconds:g}s"
                )
            time.sleep(min(poll_interval_seconds, remaining))


class AsyncProjectDeliveriesAPI:
    """Native asynchronous peer of :class:`ProjectDeliveriesAPI`."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def ensure_draft(
        self,
        project_id: ProjectId,
        request: EnsureDraftDeliveryRequest,
    ) -> DraftDeliveryAuthorityResponse:
        """Idempotently ensure an accepted-decision-bound draft Delivery."""
        response = DraftDeliveryAuthorityResponse.from_wire(
            await self._transport.execute(
                _request(
                    "ensure_project_draft_delivery",
                    f"/smr/projects/{project_id}/deliveries:ensure-draft",
                    body=cast(JsonObject, request.to_wire()),
                )
            )
        )
        return _validate_delivery(response, decision_id=request.decision_id)

    async def get(
        self,
        project_id: ProjectId,
        delivery_id: str,
        *,
        decision_id: str,
    ) -> DraftDeliveryAuthorityResponse:
        """Read one draft Delivery while retaining the accepted decision join."""
        response = DraftDeliveryAuthorityResponse.from_wire(
            await self._transport.execute(
                _request(
                    "get_project_draft_delivery",
                    f"/smr/projects/{project_id}/deliveries/{delivery_id}",
                )
            )
        )
        return _validate_delivery(
            response,
            delivery_id=delivery_id,
            decision_id=decision_id,
        )

    async def wait(
        self,
        project_id: ProjectId,
        delivery_id: str,
        *,
        decision_id: str,
        timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 2.0,
    ) -> DraftDeliveryAuthorityResponse:
        """Boundedly poll until the Delivery reaches a terminal state."""
        if timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("Delivery wait bounds must be positive")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while True:
            response = await self.get(
                project_id,
                delivery_id,
                decision_id=decision_id,
            )
            if response.delivery.state.terminal:
                return response
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Delivery {delivery_id} did not terminalize within {timeout_seconds:g}s"
                )
            await asyncio.sleep(min(poll_interval_seconds, remaining))


__all__ = ["AsyncProjectDeliveriesAPI", "ProjectDeliveriesAPI"]
