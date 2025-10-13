from __future__ import annotations

try:
    # Prefer the modern constant
    from synth_ai.config.base_url import PROD_BASE_URL_DEFAULT as PROD_BASE_URL_DEFAULT  # type: ignore
except Exception:  # pragma: no cover
    # Fallback if the modern module moves; provide a safe default
    PROD_BASE_URL_DEFAULT = "https://agent-learning.onrender.com"

__all__ = ["PROD_BASE_URL_DEFAULT"]


