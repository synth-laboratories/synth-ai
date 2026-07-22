"""Synchronous and asynchronous Research limits and economics operations."""

from __future__ import annotations

from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.research.contracts.economics import (
    ResearchBillingCatalog,
    ResearchBillingDrawdown,
    ResearchBillingEntitlements,
    ResearchBillingPlan,
    ResearchOrgLimits,
    ResearchProjectEconomics,
)
from synth_ai.core.research.operations import research_operation


def _request(operation_id: str, path: str) -> HttpRequest:
    return HttpRequest(research_operation(operation_id), path)


class LimitsAPI:
    """Read the authenticated organization's typed Research limits."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def retrieve(self) -> ResearchOrgLimits:
        return ResearchOrgLimits.from_wire(
            self._transport.execute(_request("retrieve_research_limits", "/smr/limits"))
        )


class AsyncLimitsAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def retrieve(self) -> ResearchOrgLimits:
        value = await self._transport.execute(_request("retrieve_research_limits", "/smr/limits"))
        return ResearchOrgLimits.from_wire(value)


class EconomicsAPI:
    """Read backend-owned billing and budget projections without client policy."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def entitlements(self) -> ResearchBillingEntitlements:
        value = self._transport.execute(
            _request("retrieve_billing_entitlements", "/api/v1/billing/entitlements")
        )
        return ResearchBillingEntitlements.from_wire(value)

    def project(self, project_id: str) -> ResearchProjectEconomics:
        value = self._transport.execute(
            _request(
                "retrieve_project_economics",
                f"/smr/projects/{project_id}/economics",
            )
        )
        return ResearchProjectEconomics.from_wire(value)

    def catalog(self) -> ResearchBillingCatalog:
        value = self._transport.execute(
            _request("retrieve_billing_catalog", "/smr/billing/catalog")
        )
        return ResearchBillingCatalog.from_wire(value)

    def plan(self) -> ResearchBillingPlan:
        value = self._transport.execute(_request("retrieve_billing_plan", "/smr/billing/plan"))
        return ResearchBillingPlan.from_wire(value)

    def swarm_drawdown(self, swarm_id: str) -> ResearchBillingDrawdown:
        value = self._transport.execute(
            _request(
                "retrieve_swarm_billing_drawdown",
                f"/smr/billing/runs/{swarm_id}/drawdown",
            )
        )
        return ResearchBillingDrawdown.from_wire(value)

    def effort_drawdown(self, effort_id: str) -> ResearchBillingDrawdown:
        value = self._transport.execute(
            _request(
                "retrieve_effort_billing_drawdown",
                f"/smr/billing/factory-efforts/{effort_id}/drawdown",
            )
        )
        return ResearchBillingDrawdown.from_wire(value)

    def run_drawdown(self, run_id: str) -> ResearchBillingDrawdown:
        """Compatibility alias for :meth:`swarm_drawdown`."""
        return self.swarm_drawdown(run_id)

    def factory_effort_drawdown(self, factory_effort_id: str) -> ResearchBillingDrawdown:
        """Compatibility alias for :meth:`effort_drawdown`."""
        return self.effort_drawdown(factory_effort_id)


class AsyncEconomicsAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def entitlements(self) -> ResearchBillingEntitlements:
        value = await self._transport.execute(
            _request("retrieve_billing_entitlements", "/api/v1/billing/entitlements")
        )
        return ResearchBillingEntitlements.from_wire(value)

    async def project(self, project_id: str) -> ResearchProjectEconomics:
        value = await self._transport.execute(
            _request(
                "retrieve_project_economics",
                f"/smr/projects/{project_id}/economics",
            )
        )
        return ResearchProjectEconomics.from_wire(value)

    async def catalog(self) -> ResearchBillingCatalog:
        value = await self._transport.execute(
            _request("retrieve_billing_catalog", "/smr/billing/catalog")
        )
        return ResearchBillingCatalog.from_wire(value)

    async def plan(self) -> ResearchBillingPlan:
        value = await self._transport.execute(
            _request("retrieve_billing_plan", "/smr/billing/plan")
        )
        return ResearchBillingPlan.from_wire(value)

    async def swarm_drawdown(self, swarm_id: str) -> ResearchBillingDrawdown:
        value = await self._transport.execute(
            _request(
                "retrieve_swarm_billing_drawdown",
                f"/smr/billing/runs/{swarm_id}/drawdown",
            )
        )
        return ResearchBillingDrawdown.from_wire(value)

    async def effort_drawdown(self, effort_id: str) -> ResearchBillingDrawdown:
        value = await self._transport.execute(
            _request(
                "retrieve_effort_billing_drawdown",
                f"/smr/billing/factory-efforts/{effort_id}/drawdown",
            )
        )
        return ResearchBillingDrawdown.from_wire(value)


ResearchLimitsAPI = LimitsAPI
AsyncResearchLimitsAPI = AsyncLimitsAPI
ResearchEconomicsAPI = EconomicsAPI
AsyncResearchEconomicsAPI = AsyncEconomicsAPI


__all__ = [
    "AsyncEconomicsAPI",
    "AsyncLimitsAPI",
    "EconomicsAPI",
    "LimitsAPI",
    "AsyncResearchEconomicsAPI",
    "AsyncResearchLimitsAPI",
    "ResearchEconomicsAPI",
    "ResearchLimitsAPI",
]
