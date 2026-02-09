"""Default adapter placeholder for Synth GEPA compatibility."""
# See: specifications/tanha/master_specification.md

from __future__ import annotations

from typing import Any


class DefaultAdapter:
    """Placeholder adapter for compatibility."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ValueError(
            "DefaultAdapter is not supported in Synth GEPA compatibility mode. "
            "Use gepa.optimize with task_lm instead."
        )
