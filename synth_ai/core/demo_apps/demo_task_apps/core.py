import os
import urllib.request
from dataclasses import dataclass

from synth_ai.core import localapi_state
from synth_ai.core.urls import BACKEND_URL_API
from synth_ai.core.user_config import load_user_env

DEFAULT_TASK_APP_SECRET_NAME = "hendrycks-math-task-app-secret"


@dataclass
class DemoEnv:
    dev_backend_url: str = ""
    synth_api_key: str = ""
    env_api_key: str = ""
    task_app_base_url: str = ""
    task_app_name: str = ""
    task_app_secret_name: str = DEFAULT_TASK_APP_SECRET_NAME


def _mask(value: str, keep: int = 4) -> str:
    if not value:
        return ""
    return value[:keep] + "…" if len(value) > keep else value


def modal_auth_status() -> tuple[bool, str]:
    """Return (ok, message) describing Modal CLI credential status."""

    env_token_id = (os.environ.get("MODAL_TOKEN_ID") or "").strip()
    env_token_secret = (os.environ.get("MODAL_TOKEN_SECRET") or "").strip()

    try:
        from modal.config import config as modal_config
        from modal.config import user_config_path
    except Exception as exc:  # pragma: no cover - modal optional in some envs
        return False, f"Modal client unavailable ({exc})"

    token_id = env_token_id or str(modal_config.get("token_id") or "")
    token_secret = env_token_secret or str(modal_config.get("token_secret") or "")
    profile = os.environ.get("MODAL_PROFILE") or "default"

    if token_id and token_secret:
        source = "environment variables" if env_token_id else f"profile {profile}"
        return True, f"{source} ({_mask(token_id, keep=6)})"

    missing: list[str] = []
    if not token_id:
        missing.append("token_id")
    if not token_secret:
        missing.append("token_secret")

    # If MODAL_TOKEN_ID is set but secret missing, highlight that specifically.
    if env_token_id and not env_token_secret:
        return False, (
            "MODAL_TOKEN_ID is set but MODAL_TOKEN_SECRET is missing. Set both env vars "
            "or regenerate credentials via `modal token new`."
        )

    try:
        config_path = user_config_path
    except Exception:  # pragma: no cover - defensive
        config_path = os.path.expanduser("~/.modal.toml")

    hint = "Run `modal setup` or `modal token new` to authenticate."
    if config_path and os.path.exists(config_path):
        hint += f" (config: {config_path})"

    missing_str = ", ".join(missing) or "credentials"
    return False, f"Missing Modal {missing_str}. {hint}"


def load_env() -> DemoEnv:
    """Resolve environment with sane defaults.

    Backend URL: Uses centralized BACKEND_URL_API from urls.py
    API keys: SYNTH_API_KEY from environment
    TASK_APP_BASE_URL: From environment or state
    """
    env = DemoEnv()

    load_user_env(override=False)

    synth_api_key = os.getenv("SYNTH_API_KEY") or ""
    env_api_key = os.getenv("ENVIRONMENT_API_KEY") or ""

    task_url = os.getenv("TASK_APP_BASE_URL") or ""
    task_app_name = str(os.getenv("TASK_APP_NAME") or "")
    task_app_secret_name = str(os.getenv("TASK_APP_SECRET_NAME") or DEFAULT_TASK_APP_SECRET_NAME)

    env.dev_backend_url = BACKEND_URL_API
    env.synth_api_key = synth_api_key
    env.env_api_key = env_api_key
    env.task_app_base_url = task_url.rstrip("/") if task_url else ""
    env.task_app_name = task_app_name
    env.task_app_secret_name = task_app_secret_name

    if os.getenv("SYNTH_CLI_VERBOSE", "0") == "1":
        print("ENV:")
        print(f"  BACKEND_URL={env.dev_backend_url}")
        print(f"  SYNTH_API_KEY={_mask(env.synth_api_key)}")
        print(f"  ENVIRONMENT_API_KEY={_mask(env.env_api_key)}")
        print(f"  TASK_APP_BASE_URL={env.task_app_base_url}")
        if task_app_name:
            print(f"  TASK_APP_NAME={task_app_name}")
        if task_app_secret_name:
            print(f"  TASK_APP_SECRET_NAME={task_app_secret_name}")
    return env


def assert_http_ok(
    url: str, method: str = "GET", allow_redirects: bool = True, timeout: float = 10.0
) -> bool:
    try:
        import ssl

        req = urllib.request.Request(url, method=method)
        # Default: disable SSL verification for local/dev convenience.
        # Set SYNTH_SSL_VERIFY=1 to enable verification.
        ctx = ssl._create_unverified_context()  # nosec: disabled by default for dev
        if os.getenv("SYNTH_SSL_VERIFY", "0") == "1":
            ctx = None
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:  # nosec - controlled URL
            code = getattr(resp, "status", 200)
            return 200 <= int(code) < 400
    except Exception:
        return False


def persist_localapi_url(url: str, *, name: str | None = None) -> None:
    localapi_state.persist_localapi_url(url, name=name)


def run_job(
    env: DemoEnv,
    config_toml_path: str,
    *,
    batch_size: int | None = None,
    group_size: int | None = None,
    model: str | None = None,
) -> None:
    """Create and stream a short RL job using the backend API (placeholder: prints cURL to execute)."""
    print("\nTo create an RL job, run:")
    print(
        'curl -s -X POST "' + env.dev_backend_url + '/rl/jobs" '
        "-H 'Content-Type: application/json' "
        f"-H 'Authorization: Bearer {env.synth_api_key}' "
        "-d '{"  # intentionally not fully formed here for brevity in this scaffold
    )
    print(
        "  NOTE: CLI implementation will build the full JSON body with inline TOML config and stream events."
    )
