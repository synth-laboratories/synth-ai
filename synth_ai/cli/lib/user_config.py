from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


CONFIG_DIR = Path(os.path.expanduser("~/.synth-ai"))
USER_CONFIG_PATH = CONFIG_DIR / "user_config.json"


def _ensure_config_dir() -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def load_user_config() -> dict[str, Any]:
    """Return the persisted user config as a dict (empty if missing or invalid)."""
    try:
        if USER_CONFIG_PATH.is_file():
            with USER_CONFIG_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def save_user_config(config: Mapping[str, Any]) -> None:
    """Persist a new user config dictionary (overwrites previous contents)."""
    _ensure_config_dir()
    try:
        with USER_CONFIG_PATH.open("w", encoding="utf-8") as fh:
            json.dump(dict(config), fh, indent=2, sort_keys=True)
    except Exception:
        pass


def update_user_config(updates: Mapping[str, Any]) -> dict[str, Any]:
    """Merge `updates` into the existing user config and persist the result."""
    current = load_user_config()
    current.update(updates)
    save_user_config(current)
    return current
