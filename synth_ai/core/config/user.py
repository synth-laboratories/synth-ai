from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for config.user.") from exc


def load_user_config() -> dict[str, Any]:
    """Return the persisted user config as a dict (empty if missing or invalid)."""
    result = synth_ai_py.auth_load_user_config()
    return result if isinstance(result, dict) else {}


def save_user_config(config: Mapping[str, Any]) -> None:
    """Persist a new user config dictionary (overwrites previous contents)."""
    synth_ai_py.auth_save_user_config(dict(config))


def update_user_config(updates: Mapping[str, Any]) -> dict[str, Any]:
    """Merge `updates` into the existing user config and persist the result."""
    result = synth_ai_py.auth_update_user_config(dict(updates))
    return result if isinstance(result, dict) else {}


def load_user_env(*, override: bool = True) -> dict[str, str]:
    """Hydrate ``os.environ`` from persisted Synth SDK state."""
    result = synth_ai_py.auth_load_user_env(override)
    return result if isinstance(result, dict) else {}


__all__ = [
    "load_user_config",
    "save_user_config",
    "update_user_config",
    "load_user_env",
]
