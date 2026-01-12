"""Environment resolution utilities.

This module provides non-interactive environment variable resolution
for use by SDK and CLI. URL configuration is handled by urls.py.
"""

import json
import os
from pathlib import Path

import httpx

from synth_ai.core.paths import SYNTH_HOME_DIR
from synth_ai.core.secure_files import write_private_json
from synth_ai.core.user_config import load_user_env

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
            f"Or run synth-ai setup to store it in {SYNTH_HOME_DIR}"
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


def get_synth_and_env_keys() -> tuple[str, str]:
    load_user_env(override=False)
    synth_api_key = os.environ.get("SYNTH_API_KEY")
    env_api_key = os.environ.get("ENVIRONMENT_API_KEY")
    if not synth_api_key:
        raise RuntimeError(
            f"SYNTH_API_KEY not in process environment or {SYNTH_HOME_DIR} config. "
            "Either run synth-ai setup to load automatically or manually set it in your shell."
        )
    if not env_api_key:
        raise RuntimeError(
            f"ENVIRONMENT_API_KEY not in process environment or {SYNTH_HOME_DIR} config. "
            "Either run synth-ai setup to load automatically or manually set it in your shell."
        )
    return synth_api_key, env_api_key


def mask_str(input: str, position: int = 3) -> str:
    return input[:position] + "..." + input[-position:] if len(input) > position * 2 else "***"


def ensure_env_var(key: str, expected_value: str) -> None:
    actual_value = os.getenv(key)
    if expected_value != actual_value:
        raise ValueError(f"Expected: {key}={expected_value}\nActual: {key}={actual_value}")


def resolve_env_var(key: str, override_process_env: bool = False) -> str:
    """Resolve an environment variable from available sources.

    Non-interactive: uses first available option or raises error.
    Never prompts - fails hard if value cannot be found.
    """
    import click

    env_value = os.getenv(key)
    if env_value is not None and not override_process_env:
        click.echo(f"Using {key}={mask_str(env_value)} from process environment")
        return env_value

    applied = load_user_env(override=override_process_env)
    config_value = applied.get(key)

    if override_process_env and config_value is not None:
        value = config_value
        source = "synth config"
    elif env_value is not None:
        value = env_value
        source = "process environment"
    elif config_value is not None:
        value = config_value
        source = "synth config"
    else:
        raise click.ClickException(
            f"❌ Missing required environment variable: {key}\n\n"
            f"  Options:\n"
            f"  1. Set environment variable: export {key}=<value>\n"
            f"  2. Run `synth-ai setup` to store credentials in {SYNTH_HOME_DIR}\n\n"
            f"  Searched for {key} in:\n"
            f"    - Process environment\n"
            f"    - {SYNTH_HOME_DIR}/*.json config files"
        )

    os.environ[key] = value
    ensure_env_var(key, value)
    click.echo(f"Loaded {key}={mask_str(value)} from {source}")
    return value


def write_env_var_to_json(
    key: str,
    value: str,
    output_file_path: str | Path,
) -> None:
    path = Path(output_file_path).expanduser()
    if path.exists() and not path.is_file():
        raise RuntimeError(f"{path} exists and is not a file")

    data: dict[str, str] = {}

    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc
        except OSError as exc:
            raise RuntimeError(f"Failed to read {path}: {exc}") from exc

        if not isinstance(existing, dict):
            raise RuntimeError(f"Expected JSON object in {path}")

        for existing_key, existing_value in existing.items():
            if existing_key == key:
                continue
            data[str(existing_key)] = (
                existing_value if isinstance(existing_value, str) else str(existing_value)
            )

    data[key] = value

    try:
        write_private_json(path, data, indent=2, sort_keys=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to write {path}: {exc}") from exc

    print(f"Wrote {key}={mask_str(value)} to {path}")


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
    "get_synth_and_env_keys",
    "mint_demo_api_key",
    "mask_value",
    "mask_str",
    "ensure_env_var",
    "resolve_env_var",
    "write_env_var_to_json",
]
