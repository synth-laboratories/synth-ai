"""Setup orchestration for CLI."""

from __future__ import annotations

from synth_ai.core.auth.device import fetch_credentials_from_web_browser
from synth_ai.core.auth.storage import store_credentials
from synth_ai.core.config.user import load_user_env
from synth_ai.core.utils.env import resolve_env_var


def run_setup(source="web", skip_confirm=False, confirm_callback=None):
    """Run credential setup.

    Args:
        source: "web" for browser auth, "local" for env vars
        skip_confirm: Skip confirmation prompt for web auth
        confirm_callback: Optional callable for confirmation, returns bool
    """
    credentials = {}
    if source == "local":
        credentials["SYNTH_API_KEY"] = resolve_env_var("SYNTH_API_KEY")
    elif source == "web":
        if (
            not skip_confirm
            and confirm_callback
            and not confirm_callback(
                "This will open your web browser for authentication. Continue?"
            )
        ):
            return
        credentials = fetch_credentials_from_web_browser()
        load_user_env(override=False)

    store_credentials(credentials)
