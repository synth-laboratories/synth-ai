"""``client.research.limits`` — org limits readout."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.sdk.client import ManagedResearchClient


class ResearchLimitsAPI:
    """Read organization-level Managed Research limits and allowance windows.

    Use ``client.research.limits.get()`` before launching runs to confirm the
    org has headroom under plan caps and flex-credit rules.
    """

    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def get(self) -> dict[str, Any]:
        """Return current org limits, allowance windows, and drawdown hints.

        Returns:
            Wire JSON from ``GET /smr/limits`` (allowance, plan tier, reset
            timestamps, flex-credit balance when applicable).

        Example:
            >>> limits = client.research.limits.get()
            >>> limits["plan_tier"]
        """
        return self._session.get_limits()


__all__ = ["ResearchLimitsAPI"]
