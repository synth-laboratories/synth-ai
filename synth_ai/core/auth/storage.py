"""Credential storage utilities."""

import os

from synth_ai.core.utils.env import mask_str, write_env_var_to_json
from synth_ai.core.utils.paths import SYNTH_USER_CONFIG_PATH


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
    for k, v in credentials.items():
        write_env_var_to_json(k, v, resolved_path)
        os.environ[k] = v
        print(f"Loaded {k}={mask_str(v)} to process environment")
