"""``client.research.limits`` — org limits readout."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.sdk.client import ManagedResearchClient
from synth_ai.research.models import ResearchOrgLimits


class ResearchLimitsAPI:
    """Read organization-level Managed Research limits and allowance windows.

    Use ``client.research.limits.get_typed()`` before launching runs to confirm the
    org has headroom under plan caps and flex-credit rules.
    """

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def get(self) -> dict[str, Any]:
        """Return the backward-compatible raw organization limits payload.

        This shape is retained for existing callers and JSON-producing CLI
        commands. New integrations should use :meth:`get_typed`.
        """
        return self._session.get_limits()

    def get_typed(self) -> ResearchOrgLimits:
        """Return typed current organization limits and usage windows.

        Returns:
            ``ResearchOrgLimits`` parsed from the backend-owned ``GET /smr/limits``
            response.

        Example:
            >>> limits = client.research.limits.get_typed()
            >>> limits.plan
        """
        return self._session.usage.get_limits()


__all__ = ["ResearchLimitsAPI"]
