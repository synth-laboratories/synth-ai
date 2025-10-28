from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

__all__ = ["validate_filter_options"]


def validate_filter_options(options: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Validate parameters passed to the filter CLI command."""
    return options
