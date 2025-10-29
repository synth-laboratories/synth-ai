from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

__all__ = ["validate_modal_serve_options"]


def validate_modal_serve_options(options: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Validate parameters passed to the modal-serve CLI command."""
    return options
