"""Environment resolution utilities.

This module provides non-interactive environment variable resolution
for use by SDK and CLI. It consolidates the various env resolution
patterns into a clean API.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from .errors import AuthenticationError, ConfigError

# Default production URL
PROD_BASE_URL = "https://www.api.usesynth.ai"
PROD_BASE_URL_DEFAULT = PROD_BASE_URL  # Alias for backward compatibility


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
            f"Or add to .env file: {env_key}=<your-key>"
        )
    return value


def get_backend_url(
    mode: Literal["prod", "dev", "local"] | None = None,
) -> str:
    """Resolve backend URL.

    Priority order:
    1. SYNTH_BACKEND_URL env var (if set)
    2. Mode-specific URL based on SYNTH_BACKEND_MODE or explicit mode
    3. Default to production

    Args:
        mode: Force a specific mode (prod/dev/local), or detect from env

    Returns:
        Backend URL (without trailing /api)
    """
    # Direct override takes precedence
    direct = os.environ.get("SYNTH_BACKEND_URL")
    if direct:
        return _normalize_url(direct)

    # Determine mode
    if mode is None:
        mode_env = os.environ.get("SYNTH_BACKEND_MODE", "").lower()
        mode = mode_env if mode_env in ("prod", "dev", "local") else "prod"  # type: ignore

    if mode == "local":
        url = os.environ.get("SYNTH_LOCAL_URL", "http://localhost:8000")
    elif mode == "dev":
        url = os.environ.get("SYNTH_DEV_URL", "http://localhost:8000")
    else:
        url = os.environ.get("SYNTH_PROD_URL", PROD_BASE_URL)

    return _normalize_url(url)


def _normalize_url(url: str) -> str:
    """Normalize URL: strip trailing slashes and /api suffix."""
    url = url.strip().rstrip("/")
    if url.endswith("/api"):
        url = url[:-4]
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def resolve_env_file(
    explicit_path: str | Path | None = None,
    search_cwd: bool = True,
) -> Path | None:
    """Find and return path to .env file.

    Args:
        explicit_path: If provided, use this path directly
        search_cwd: If True, search current directory for .env

    Returns:
        Path to .env file, or None if not found
    """
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if path.exists():
            return path
        raise ConfigError(f"Env file not found: {path}")

    if search_cwd:
        cwd_env = Path.cwd() / ".env"
        if cwd_env.exists():
            return cwd_env.resolve()

    return None


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file into a dictionary.

    Args:
        path: Path to .env file

    Returns:
        Dict mapping env var names to values
    """
    result: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Strip quotes
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                result[key] = value
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"Failed to read env file {path}: {e}") from e
    return result


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


def get_backend_from_env() -> tuple[str, str]:
    """Resolve (base_url, api_key) using LOCAL/DEV/PROD override scheme.

    Env vars consulted:
    - BACKEND_OVERRIDE = full URL (with or without /api)
    - SYNTH_BACKEND_URL_OVERRIDE = local|dev|prod (case-insensitive)
    - LOCAL_BACKEND_URL, TESTING_LOCAL_SYNTH_API_KEY
    - DEV_BACKEND_URL, DEV_SYNTH_API_KEY
    - PROD_BACKEND_URL, TESTING_PROD_SYNTH_API_KEY (fallback to SYNTH_API_KEY)

    Base URL is normalized (no trailing /api).
    Defaults: prod base URL → https://api.usesynth.ai

    Returns:
        Tuple of (base_url, api_key)
    """
    direct_override = (os.environ.get("BACKEND_OVERRIDE") or "").strip()
    if direct_override:
        base = _normalize_url(direct_override)
        api_key = os.environ.get("SYNTH_API_KEY", "").strip()
        return base, api_key

    # Determine mode from env
    mode_override = (os.environ.get("SYNTH_BACKEND_URL_OVERRIDE", "") or "").strip().lower()
    mode = mode_override if mode_override in ("local", "dev", "prod") else "prod"

    if mode == "local":
        base = os.environ.get("LOCAL_BACKEND_URL", "http://localhost:8000")
        key = os.environ.get("TESTING_LOCAL_SYNTH_API_KEY", "")
        return _normalize_url(base), key

    if mode == "dev":
        base = os.environ.get("DEV_BACKEND_URL", "") or "http://localhost:8000"
        key = os.environ.get("DEV_SYNTH_API_KEY", "")
        return _normalize_url(base), key

    # prod
    base = os.environ.get("PROD_BACKEND_URL", PROD_BASE_URL)
    key = (
        os.environ.get("PROD_SYNTH_API_KEY", "")
        or os.environ.get("TESTING_PROD_SYNTH_API_KEY", "")
        or os.environ.get("SYNTH_API_KEY", "")
    )
    return _normalize_url(base), key


__all__ = [
    "get_api_key",
    "get_backend_url",
    "get_backend_from_env",
    "resolve_env_file",
    "load_env_file",
    "mask_value",
    "PROD_BASE_URL",
    "PROD_BASE_URL_DEFAULT",
]

