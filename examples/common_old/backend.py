from __future__ import annotations

from synth_ai.config.base_url import get_backend_from_env, PROD_BASE_URL_DEFAULT


DEFAULT_PROD_BACKEND = f"{PROD_BASE_URL_DEFAULT.rstrip('/')}/api"


def resolve_backend_url() -> str:
    """Resolve backend base URL honoring BACKEND_OVERRIDE and env overrides.

    Always returns a URL ending with /api.
    """
    base, _ = get_backend_from_env()
    base = base.rstrip("/")
    return base if base.endswith("/api") else f"{base}/api"


if __name__ == "__main__":
    print(resolve_backend_url())
