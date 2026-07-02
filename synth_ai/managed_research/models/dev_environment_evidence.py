"""Client-side DevEnvironment evidence snapshot models."""

from __future__ import annotations

from dataclasses import dataclass, field

from synth_ai.managed_research.models.billing import (
    SmrBillingDrawdown,
    SmrBillingPreflight,
)
from synth_ai.managed_research.models.types import (
    DevEnvironment,
    DevEnvironmentAttach,
    DevEnvironmentCollection,
    DevEnvironmentPreflight,
    DevEnvironmentUsage,
)


@dataclass(frozen=True)
class DevEnvironmentEvidence:
    """Composed read snapshot over existing DevEnvironment owner routes."""

    dev_environment_id: str
    environment: DevEnvironment
    services: DevEnvironmentCollection
    attach: DevEnvironmentAttach
    runs: DevEnvironmentCollection
    usage: DevEnvironmentUsage
    receipts: DevEnvironmentCollection
    preflight: DevEnvironmentPreflight | None = None
    logs: DevEnvironmentCollection | None = None
    billing_preflight: SmrBillingPreflight | None = None
    billing_drawdown: SmrBillingDrawdown | None = None
    summary: dict[str, object] = field(default_factory=dict)


__all__ = ["DevEnvironmentEvidence"]
