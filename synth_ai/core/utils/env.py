"""Non-interactive credential resolution. URL configuration lives in urls.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from synth_ai.core.errors import AuthenticationError, ConfigError
from synth_ai.core.utils.paths import SYNTH_HOME_DIR


def get_api_key(env_key: str = "SYNTH_API_KEY", required: bool = True) -> str | None:
    """Read an API key from the process environment, then from ``~/.synth_ai``.

    Args:
        env_key: Environment variable name to check
        required: If True, raises AuthenticationError when not found

    Returns:
        API key string or None if not required and not found

    Raises:
        AuthenticationError: If required and not found
        ConfigurationError: If a config file exists but cannot be read
    """
    value = os.getenv(env_key) or _load_user_env().get(env_key)
    if not value and required:
        raise AuthenticationError(
            f"Missing required API key: {env_key}\n"
            f"Set it via: export {env_key}=<your-key>\n"
            f"Or run synth-ai setup to store it in {SYNTH_HOME_DIR}"
        )
    return value


def _load_user_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in _candidate_config_paths():
        if not path.is_file():
            continue
        # A config file that exists but will not parse is a broken machine, not
        # a machine without credentials. Reporting it as "no API key" sends the
        # reader to look for a key they already set.
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload: Any = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Cannot read Synth config at {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ConfigError(f"Synth config at {path} must contain a JSON object")
        for key, value in payload.items():
            if isinstance(value, str):
                values[str(key)] = value
            elif value is not None:
                values[str(key)] = str(value)
    return values


def _candidate_config_paths() -> list[Path]:
    paths = [SYNTH_HOME_DIR / "config.json"]
    if SYNTH_HOME_DIR.exists():
        paths.extend(sorted(SYNTH_HOME_DIR.glob("*.json")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


__all__ = ["get_api_key"]
