"""Supported models configuration (single source of truth)."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any, Dict


@lru_cache(maxsize=1)
def get_supported_models() -> Dict[str, Any]:
    """Load supported model metadata from packaged JSON."""
    data = (
        resources.files("synth_ai.config")
        .joinpath("supported_models.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(data)
