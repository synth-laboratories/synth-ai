from __future__ import annotations

import os


DEFAULT_PROD_BACKEND = "https://agent-learning.onrender.com/api"


def resolve_backend_url() -> str:
    """Resolve backend base URL with sensible defaults.

    Precedence:
    1) BACKEND_OVERRIDE env (already including /api or not)
    2) PROD_BACKEND_URL env (already including /api or not)
    3) DEFAULT_PROD_BACKEND constant in this file
    Always returns a URL ending with /api
    """
    override = os.getenv("BACKEND_OVERRIDE", "").strip()
    if override:
        base = override
    else:
        raw = os.getenv("PROD_BACKEND_URL", "").strip()
        base = raw or DEFAULT_PROD_BACKEND
    base = base.rstrip("/")
    return base if base.endswith("/api") else f"{base}/api"


if __name__ == "__main__":
    print(resolve_backend_url())


