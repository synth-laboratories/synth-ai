"""``client.research.limits`` — org limits readout."""

from __future__ import annotations

from typing import Any

from synth_ai.managed_research.sdk.client import ManagedResearchClient


class ResearchLimitsAPI:
    def __init__(self, session: ManagedResearchClient) -> None:
        self._session = session

    def get(self) -> dict[str, Any]:
        return self._session.get_limits()


__all__ = ["ResearchLimitsAPI"]
