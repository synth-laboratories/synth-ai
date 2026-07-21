"""Typed Research economics contracts.

The names in this module are the stable core vocabulary.  Their current class
objects are shared with the compatibility decoder so old imports preserve
identity while consumers migrate.
"""

from __future__ import annotations

from synth_ai.core.research._legacy.models.billing import (
    SmrBillingCatalog as ResearchBillingCatalog,
    SmrBillingDrawdown as ResearchBillingDrawdown,
    SmrBillingPlanSnapshot as ResearchBillingPlan,
)
from synth_ai.core.research._legacy.models.canonical_usage import (
    BillingEntitlementSnapshot as ResearchBillingEntitlements,
    OrgLimits as ResearchOrgLimits,
    SmrProjectEconomics as ResearchProjectEconomics,
)

__all__ = [
    "ResearchBillingCatalog",
    "ResearchBillingDrawdown",
    "ResearchBillingEntitlements",
    "ResearchBillingPlan",
    "ResearchOrgLimits",
    "ResearchProjectEconomics",
]
