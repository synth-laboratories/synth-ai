from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

__all__ = ["validate_deploy_options"]


def validate_deploy_options(options: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Validate parameters passed to the deploy CLI command."""
    return options
