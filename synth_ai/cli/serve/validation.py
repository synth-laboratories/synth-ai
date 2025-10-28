from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

__all__ = ["validate_serve_options"]


def validate_serve_options(options: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Validate parameters passed to the serve CLI command."""
    return options
