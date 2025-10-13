from __future__ import annotations

# Compatibility package to mirror historical import paths.
# Re-export constants from the modern location under synth_ai.config.

try:
    from synth_ai.config.base_url import PROD_BASE_URL_DEFAULT as _PROD
except Exception:  # pragma: no cover
    _PROD = None

__all__ = [
    "_PROD",
]


