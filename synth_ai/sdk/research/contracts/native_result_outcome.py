# Generated from backend native evidence producer; do not edit.
# Source SHA256: 6e01d203cb61a70634bcef1b7ee4b69c4d32a6558b02befcdf7e53cc7f7f2f62
"""Native scientific outcomes remain records even without a measured number.

See platform/forge_scientific_delivery.md and Forge decision 0002.
"""

import math
from enum import StrEnum


class ResearchResultOutcome(StrEnum):
    MEASURED = "measured"
    FAILED = "failed"
    NEGATIVE = "negative"
    NULL = "null"
    INCONCLUSIVE = "inconclusive"
    INTERRUPTED = "interrupted"
    REJECTED = "rejected"


def require_result_measurement(
    value: object, outcome: object
) -> tuple[float | None, ResearchResultOutcome]:
    """Validate before persistence; never infer or fabricate a zero score."""
    selected = ResearchResultOutcome(outcome)
    if value is None:
        if selected is ResearchResultOutcome.MEASURED:
            raise ValueError("measured scientific result requires a numeric value")
        return None, selected
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("scientific result value must be a finite number or null")
    return float(value), selected
