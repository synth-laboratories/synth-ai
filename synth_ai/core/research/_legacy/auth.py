"""Compatibility helpers re-exported from the new internal layout."""

from synth_ai.core.research._legacy._internal.crypto import encrypt_for_backend
from synth_ai.core.research._legacy._internal.env import get_api_key
from synth_ai.core.research._legacy._internal.urls import BACKEND_URL_BASE, normalize_backend_base

__all__ = [
    "BACKEND_URL_BASE",
    "encrypt_for_backend",
    "get_api_key",
    "normalize_backend_base",
]
