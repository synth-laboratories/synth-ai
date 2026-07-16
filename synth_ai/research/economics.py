"""``client.research.economics`` — typed entitlement and project economics reads."""

from __future__ import annotations

from synth_ai.managed_research.sdk.client import ManagedResearchClient
from synth_ai.research.models import ResearchBillingEntitlements, ResearchProjectEconomics


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


__all__ = ["ResearchEconomicsAPI"]
