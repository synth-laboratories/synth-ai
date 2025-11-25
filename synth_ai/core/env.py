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
PROD_BASE_URL = "https://agent-learning.onrender.com"


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
        if mode_env in ("prod", "dev", "local"):
            mode = mode_env  # type: ignore
        else:
            mode = "prod"

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


__all__ = [
    "get_api_key",
    "get_backend_url",
    "resolve_env_file",
    "load_env_file",
    "mask_value",
    "PROD_BASE_URL",
]

