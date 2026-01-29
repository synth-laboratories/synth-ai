"""Credential storage utilities."""

import os

from synth_ai.core.utils.env import mask_str
from synth_ai.core.utils.paths import SYNTH_USER_CONFIG_PATH

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover - rust bindings required
    raise RuntimeError("synth_ai_py is required for credential storage.") from exc


def store_credentials(credentials, config_path=None):
    """Store credentials to config file and environment."""
    required = {"SYNTH_API_KEY"}
    missing = [k for k in required if not credentials.get(k)]
    if missing:
        raise ValueError(f"Missing credential values: {', '.join(missing)}")

    if not credentials.get("ENVIRONMENT_API_KEY"):
        print(
            "ENVIRONMENT_API_KEY not provided by device auth. "
            "Set it manually if you plan to deploy or run task apps."
        )

    resolved_path = config_path or str(SYNTH_USER_CONFIG_PATH)
    if hasattr(synth_ai_py, "auth_store_credentials_atomic"):
        synth_ai_py.auth_store_credentials_atomic(credentials, resolved_path)
    else:
        synth_ai_py.auth_store_credentials(credentials, resolved_path)
    for k, v in credentials.items():
        os.environ[k] = v
        print(f"Loaded {k}={mask_str(v)} to process environment")
