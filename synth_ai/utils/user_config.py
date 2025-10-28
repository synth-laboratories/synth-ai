import contextlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(os.path.expanduser("~/.synth-ai"))
USER_CONFIG_PATH = CONFIG_DIR / "user_config.json"


def _ensure_config_dir() -> None:
    with contextlib.suppress(Exception):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)


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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _load_task_app_entries() -> dict[str, Any]:
    data = _load_json(Path(os.path.expanduser("~/.synth-ai/task_app_config.json")))
    if "apps" in data and isinstance(data["apps"], dict):
        return data["apps"]
    return {}


def _select_task_app_entry(entries: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    if not entries:
        return None, {}

    demo_dir = os.environ.get("SYNTH_DEMO_DIR")
    if demo_dir and demo_dir in entries:
        return demo_dir, entries[demo_dir]

    try:
        cwd = str(Path.cwd().resolve())
        if cwd in entries:
            return cwd, entries[cwd]
    except Exception:
        pass

    best_key = None
    best_entry: dict[str, Any] = {}
    best_ts = ""
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        ts = str(entry.get("last_used") or "")
        if ts > best_ts:
            best_key = key
            best_entry = entry
            best_ts = ts
    return best_key, best_entry


def load_user_env(*, override: bool = True) -> dict[str, str]:
    """Hydrate ``os.environ`` from persisted Synth SDK state."""

    applied: dict[str, str] = {}

    def _apply(mapping: Mapping[str, Any]) -> None:
        for key, value in mapping.items():
            if value is None:
                continue
            str_value = value if isinstance(value, str) else str(value)
            if override or key not in os.environ:
                os.environ[key] = str_value
            applied[key] = str_value

    config = load_user_config()
    _apply(config)

    entry_key, entry = _select_task_app_entry(_load_task_app_entries())
    if entry:
        modal_block = entry.get("modal") if isinstance(entry.get("modal"), dict) else {}
        if modal_block:
            _apply(
                {
                    "TASK_APP_BASE_URL": modal_block.get("base_url"),
                    "TASK_APP_NAME": modal_block.get("app_name"),
                    "TASK_APP_SECRET_NAME": modal_block.get("secret_name"),
                }
            )
        if entry_key:
            _apply({"SYNTH_DEMO_DIR": entry_key})

    return applied


__all__ = [
    "USER_CONFIG_PATH",
    "load_user_config",
    "save_user_config",
    "update_user_config",
    "load_user_env",
]
