"""``client.research.limits`` — org limits readout."""

from __future__ import annotations

from synth_ai.managed_research.sdk.client import ManagedResearchClient
from synth_ai.research.models import ResearchOrgLimits


class ResearchLimitsAPI:
    """Read organization-level Managed Research limits and allowance windows.

    Use ``client.research.limits.get()`` before launching runs to confirm the
    org has headroom under plan caps and flex-credit rules.
    """

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def get(self) -> ResearchOrgLimits:
        """Return typed current org limits and usage windows.

        Returns:
            ``ResearchOrgLimits`` parsed from the backend-owned ``GET /smr/limits``
            response.

        Example:
            >>> limits = client.research.limits.get()
            >>> limits.plan
        """
        return self._session.usage.get_limits()


__all__ = ["ResearchLimitsAPI"]
