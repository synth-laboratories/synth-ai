"""``client.research.economics`` — typed entitlement and project economics reads."""

from __future__ import annotations

from synth_ai.managed_research.sdk.client import ManagedResearchClient
from synth_ai.research.models import (
    ResearchBillingCatalog,
    ResearchBillingDrawdown,
    ResearchBillingEntitlements,
    ResearchBillingPlan,
    ResearchProjectEconomics,
)


class ResearchEconomicsAPI:
    """Read backend-owned billing entitlements and project economics.

    This namespace exposes authoritative read models only. It does not calculate
    discounts, allowances, or budget decisions in the client.
    """

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def entitlements(self) -> ResearchBillingEntitlements:
        """Return the authenticated organization's billing entitlement snapshot."""
        return self._session.usage.get_billing_entitlements()

    def project(self, project_id: str) -> ResearchProjectEconomics:
        """Return typed usage, entitlement, and budget state for one project."""
        return self._session.get_project_economics(project_id)

    def catalog(self) -> ResearchBillingCatalog:
        """Return the canonical billing catalog."""
        return self._session.billing.catalog()

    def plan(self) -> ResearchBillingPlan:
        """Return the authenticated organization's canonical billing plan snapshot."""
        return self._session.billing.plan()

    def run_drawdown(self, run_id: str) -> ResearchBillingDrawdown:
        """Return canonical allowance and wallet drawdown for one run."""
        return self._session.billing.run_drawdown(run_id)

    def factory_effort_drawdown(self, factory_effort_id: str) -> ResearchBillingDrawdown:
        """Return canonical drawdown for one Factory effort."""
        return self._session.billing.factory_effort_drawdown(factory_effort_id)


__all__ = ["ResearchEconomicsAPI"]
