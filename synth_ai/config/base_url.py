"""
Base URL resolution for learning-v2 and related backend APIs.

Default to production, allow overrides via environment variables:
- LEARNING_V2_BASE_URL (highest precedence)
- SYNTH_BASE_URL (legacy)
- SYNTH_LOCAL_BASE_URL
- SYNTH_DEV_BASE_URL
- SYNTH_PROD_BASE_URL (fallback if none provided)

Normalization: ensure the returned URL ends with "/api".
"""

import os
from typing import Literal


PROD_BASE_URL_DEFAULT = "https://agent-learning.onrender.com"


def _normalize_base(url: str) -> str:
    url = url.strip()
    if url.endswith("/v1"):
        url = url[:-3]
    url = url.rstrip("/")
    if not url.endswith("/api"):
        url = f"{url}/api"
    return url


def get_learning_v2_base_url(mode: Literal["dev","prod"] = "prod") -> str:
    if mode == "prod":
        prod = os.getenv("SYNTH_PROD_BASE_URL") or PROD_BASE_URL_DEFAULT
        return _normalize_base(prod)
    # Priority order
    env_url = os.getenv("LEARNING_V2_BASE_URL")
    if env_url:
        return _normalize_base(env_url)

    legacy = os.getenv("SYNTH_BASE_URL")
    if legacy:
        return _normalize_base(legacy)

    local = os.getenv("SYNTH_LOCAL_BASE_URL")
    if local:
        return _normalize_base(local)

    dev = os.getenv("SYNTH_DEV_BASE_URL")
    if dev:
        return _normalize_base(dev)

    raise Exception()

