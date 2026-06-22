"""Environment and local-config helpers.

Missing or invalid local config files are treated as empty (not fatal) so CLI and
library entrypoints can still rely on environment variables alone; failures are
logged at DEBUG for support.

# See: Synth Style — explicit degradation; one configuration authority layer.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

_logger = logging.getLogger(__name__)


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _logger.debug("managed-research: skipped config file %s", path, exc_info=exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def config_search_paths() -> tuple[Path, ...]:
    home = Path.home()
    return (
        home / ".synth_ai" / "config.json",
        home / ".config" / "synth" / "config.json",
    )


def get_api_key(env_key: str = "SYNTH_API_KEY", required: bool = True) -> str | None:
    """Resolve the API key from environment or known local Synth config files."""

    value = str(os.getenv(env_key) or "").strip()
    if value:
        return value
    for path in config_search_paths():
        payload = _read_json_object(path)
        candidate = payload.get(env_key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    if required:
        raise ValueError(f"{env_key} is required (set {env_key} or add it to local Synth config)")
    return None


__all__ = ["config_search_paths", "get_api_key"]
