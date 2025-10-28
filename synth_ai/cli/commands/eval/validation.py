from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

__all__ = ["validate_eval_options"]


def validate_eval_options(options: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Lightweight hook for validating eval CLI options."""
    return options
