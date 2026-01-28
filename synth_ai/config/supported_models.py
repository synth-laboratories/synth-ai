"""Supported models configuration (single source of truth)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover - rust bindings required
    raise RuntimeError("synth_ai_py is required for config.supported_models.") from exc


@lru_cache(maxsize=1)
def get_supported_models() -> Dict[str, Any]:
    """Load supported model metadata from Rust core assets."""
    if synth_ai_py is None or not hasattr(synth_ai_py, "supported_models"):
        raise RuntimeError("Rust core supported_models required; synth_ai_py is unavailable.")
    return synth_ai_py.supported_models()
