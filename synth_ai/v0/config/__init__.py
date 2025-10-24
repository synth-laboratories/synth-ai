from __future__ import annotations

# Compatibility package to mirror historical import paths.
# Re-export constants from the modern location under synth_ai.config.

from synth_ai._utils.base_url import PROD_BASE_URL_DEFAULT as _PROD

__all__ = [
    "_PROD",
]

