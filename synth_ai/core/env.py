"""Environment resolution utilities.

This module provides non-interactive environment variable resolution
for use by SDK and CLI. URL configuration is handled by urls.py.
"""

import os

import httpx

from .errors import AuthenticationError
from .urls import BACKEND_URL_BASE


def get_api_key(env_key: str = "SYNTH_API_KEY", required: bool = True) -> str | None:
    """Get API key from environment.

    Args:
        env_key: Environment variable name to check
        required: If True, raises AuthenticationError when not found

    Returns:
        API key string or None if not required and not found

    Raises:
        AuthenticationError: If required and not found
    """
    value = os.environ.get(env_key)
    if not value and required:
        raise AuthenticationError(
            f"Missing required API key: {env_key}\n"
            f"Set it via: export {env_key}=<your-key>\n"
            f"Or run synth-ai setup to store it in ~/.synth-ai"
        )
    return value


def mask_value(value: str, visible_chars: int = 4) -> str:
    """Mask a sensitive value for display.

    Args:
        value: The value to mask
        visible_chars: Number of characters to show at start and end

    Returns:
        Masked string like "abc...xyz"
    """
    if len(value) <= visible_chars * 2:
        return "***"
    return f"{value[:visible_chars]}...{value[-visible_chars:]}"


def mint_demo_api_key(
    backend_url: str | None = None,
    ttl_hours: int = 4,
    timeout: float = 30.0,
) -> str:
    """Mint a demo Synth API key from the backend.

    Args:
        backend_url: Backend URL (defaults to BACKEND_URL_BASE)
        ttl_hours: Time-to-live in hours (default: 4)
        timeout: Request timeout in seconds (default: 30.0)

    Returns:
        Demo API key string

    Raises:
        RuntimeError: If the request fails or returns invalid response
    """
    if backend_url is None:
        backend_url = BACKEND_URL_BASE

    url = f"{backend_url}/api/demo/keys"

    try:
        resp = httpx.post(
            url,
            json={"ttl_hours": ttl_hours},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        api_key = data.get("api_key")
        if not api_key or not isinstance(api_key, str):
            raise RuntimeError(f"Invalid response from demo key endpoint: {data}")

        return api_key
    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to mint demo API key: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error minting demo API key: {e}") from e


__all__ = [
    "get_api_key",
    "mint_demo_api_key",
    "mask_value",
]
