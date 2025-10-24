from __future__ import annotations

import os
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from synth_ai._utils.base_url import PROD_BASE_URL_DEFAULT
from synth_ai._utils.task_app_state import (
    DEFAULT_TASK_APP_SECRET_NAME,
)
from synth_ai._utils.task_app_state import (
    load_demo_dir as _load_demo_dir,
)
from synth_ai._utils.task_app_state import (
    load_template_id as _load_template_id,
)
from synth_ai._utils.task_app_state import (
    persist_api_key as _persist_api_key,
)
from synth_ai._utils.task_app_state import (
    persist_demo_dir as _persist_demo_dir,
)
from synth_ai._utils.task_app_state import (
    persist_env_api_key as _persist_env_api_key,
)
from synth_ai._utils.task_app_state import (
    persist_task_url as _persist_task_url,
)
from synth_ai._utils.task_app_state import (
    persist_template_id as _persist_template_id,
)
from synth_ai._utils.task_app_state import (
    resolve_task_app_entry as _resolve_task_app_entry,
)
from synth_ai._utils.task_app_state import (
    task_app_id_from_path as _task_app_id_from_path,
)
from synth_ai._utils.task_app_state import (
    update_task_app_entry as _update_task_app_entry,
)
from synth_ai._utils.user_config import load_user_config


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


persist_env_api_key = _persist_env_api_key
persist_task_url = _persist_task_url
persist_api_key = _persist_api_key


def persist_dotenv_values(values: dict[str, str], *, cwd: str | None = None) -> str:
    """Persist key/value pairs to a local .env file for compatibility."""

    base_dir = Path(cwd or os.getcwd())
    base_dir.mkdir(parents=True, exist_ok=True)
    env_path = base_dir / ".env"

    existing: dict[str, str] = {}
    if env_path.is_file():
        for raw in env_path.read_text().splitlines():
            if not raw or raw.lstrip().startswith("#") or "=" not in raw:
                continue
            k, v = raw.split("=", 1)
            existing[k.strip()] = v.strip()

    existing.update({k: str(v) for k, v in values.items() if v is not None})

    with env_path.open("w", encoding="utf-8") as handle:
        for key, value in existing.items():
            handle.write(f"{key}={value}\n")

    for key, value in existing.items():
        os.environ.setdefault(key, value)

    return str(env_path)


def persist_demo_dir(demo_dir: str) -> None:
    """Store the demo directory path for subsequent commands."""

    _persist_demo_dir(demo_dir)


def load_demo_dir() -> str | None:
    """Load the stored demo directory path, if any."""

    return _load_demo_dir()


def persist_template_id(template_id: str | None) -> None:
    """Record the last materialized demo template id."""

    _persist_template_id(template_id)


def load_template_id() -> str | None:
    """Return the stored demo template id, if any."""

    return _load_template_id()


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
    """Resolve environment values from ``os.environ``, user configuration, and task-app metadata."""

    env = DemoEnv()

    os_env: dict[str, str] = dict(os.environ)
    user_config_map: dict[str, str] = load_user_config()

    # Export persisted keys into the process environment if not already set.
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        if key not in os_env:
            value = user_config_map.get(key, "").strip()
            if value:
                os.environ.setdefault(key, value)
                os_env[key] = value

    preferred_path = _task_app_id_from_path(load_demo_dir())
    if not preferred_path:
        preferred_path = _task_app_id_from_path(Path.cwd())
    current_task_path, entry = _resolve_task_app_entry(preferred_path)
    modal_entry = entry.get("modal", {}) if isinstance(entry, dict) else {}

    default_root = PROD_BASE_URL_DEFAULT.rstrip("/")
    prod_default = f"{default_root}/api"

    def _pick(*candidates: str | None) -> str:
        for candidate in candidates:
            if candidate is None:
                continue
            candidate = str(candidate).strip()
            if candidate:
                return candidate
        return ""

    # Backend URL resolution
    backend_override = _pick(os_env.get("BACKEND_OVERRIDE"), user_config_map.get("BACKEND_OVERRIDE"))
    dev_env = _pick(os_env.get("DEV_BACKEND_URL"), user_config_map.get("DEV_BACKEND_URL"))

    if backend_override:
        dev_url = backend_override
    elif dev_env:
        lower = dev_env.lower()
        if "localhost" in lower or "127.0.0.1" in lower or lower.endswith(":8000"):
            dev_url = dev_env
        else:
            dev_url = prod_default
    else:
        dev_url = prod_default
    if not dev_url.endswith("/api"):
        dev_url = dev_url.rstrip("/") + "/api"

    synth_api_key = _pick(os_env.get("SYNTH_API_KEY"), user_config_map.get("SYNTH_API_KEY"))
    if not synth_api_key:
        mode = (
            "prod"
            if default_root in dev_url
            else ("local" if ("localhost" in dev_url or "127.0.0.1" in dev_url) else "dev")
        )
        if mode == "prod":
            synth_api_key = _pick(
                os_env.get("PROD_SYNTH_API_KEY"),
                user_config_map.get("PROD_SYNTH_API_KEY"),
            )
        else:
            synth_api_key = _pick(
                os_env.get("DEV_SYNTH_API_KEY"),
                user_config_map.get("DEV_SYNTH_API_KEY"),
                os_env.get("TESTING_LOCAL_SYNTH_API_KEY"),
                user_config_map.get("TESTING_LOCAL_SYNTH_API_KEY"),
            )
        if synth_api_key:
            persist_api_key(synth_api_key)

    env_api_key = _pick(
        os_env.get("ENVIRONMENT_API_KEY"),
        user_config_map.get("ENVIRONMENT_API_KEY"),
        os_env.get("DEV_ENVIRONMENT_API_KEY"),
        user_config_map.get("DEV_ENVIRONMENT_API_KEY"),
    )

    # Task app URL
    task_url = _pick(
        os_env.get("TASK_APP_BASE_URL"),
        user_config_map.get("TASK_APP_BASE_URL"),
        modal_entry.get("base_url") if isinstance(modal_entry, dict) else None,
    )

    task_app_name = _pick(
        os_env.get("TASK_APP_NAME"),
        user_config_map.get("TASK_APP_NAME"),
        modal_entry.get("app_name") if isinstance(modal_entry, dict) else None,
    )
    task_app_secret_name = _pick(
        os_env.get("TASK_APP_SECRET_NAME"),
        user_config_map.get("TASK_APP_SECRET_NAME"),
        modal_entry.get("secret_name") if isinstance(modal_entry, dict) else None,
        DEFAULT_TASK_APP_SECRET_NAME,
    )

    env.dev_backend_url = dev_url.rstrip("/")
    env.synth_api_key = synth_api_key
    env.env_api_key = env_api_key
    env.task_app_base_url = task_url.rstrip("/") if task_url else ""
    env.task_app_name = task_app_name
    env.task_app_secret_name = task_app_secret_name

    if current_task_path:
        _update_task_app_entry(current_task_path)

    if os.getenv("SYNTH_CLI_VERBOSE", "0") == "1":
        print("ENV:")
        print(f"  DEV_BACKEND_URL={env.dev_backend_url}")
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
        ctx = ssl._create_unverified_context()  # nosec: disabled by default for dev
        if os.getenv("SYNTH_SSL_VERIFY", "0") == "1":
            ctx = None
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:  # nosec - controlled URL
            code = getattr(resp, "status", 200)
            return 200 <= int(code) < 400
    except Exception:
        return False


def deploy_modal_math(env: DemoEnv) -> str:
    """Deploy Math Task App to Modal using in-repo deploy script; return public URL."""

    this_dir = os.path.dirname(__file__)
    demo_script = os.path.join(this_dir, "math", "deploy_task_app.sh")
    repo_root = os.path.abspath(os.path.join(this_dir, "../.."))
    fallback_script = os.path.join(repo_root, "examples", "rl", "deploy_task_app.sh")
    script = demo_script if os.path.isfile(demo_script) else fallback_script
    if not os.path.isfile(script):
        raise RuntimeError(f"deploy_task_app.sh not found at {demo_script} or {fallback_script}")

    envp = os.environ.copy()
    if env.env_api_key:
        envp["ENVIRONMENT_API_KEY"] = env.env_api_key
    print(f"Deploying Math Task App to Modal using: {script}")
    subprocess.check_call(["bash", script], cwd=os.path.dirname(script), env=envp)

    for candidate in (".last_deploy.log", ".last_deploy.dev.log", ".last_deploy.manual.log"):
        p = os.path.join(os.path.dirname(script), candidate)
        try:
            with open(p) as fh:
                for line in fh:
                    if "modal.run" in line:
                        return line.strip().split()[-1].rstrip("/")
        except Exception:
            continue
    raise RuntimeError("Failed to extract Modal Task App URL from deploy logs")


def run_job(
    env: DemoEnv,
    config_toml_path: str,
    *,
    batch_size: int | None = None,
    group_size: int | None = None,
    model: str | None = None,
) -> None:
    """Create and stream a short RL job using the backend API (placeholder: prints cURL to execute)."""

    backend = env.dev_backend_url.rstrip("/")
    api_base = backend if backend.endswith("/api") else backend + "/api"
    print("\nTo create an RL job, run:")
    print(
        'curl -s -X POST "' + api_base + '/rl/jobs" '
        "-H 'Content-Type: application/json' "
        f"-H 'Authorization: Bearer {env.synth_api_key}' "
        "-d '{"  # intentionally not fully formed here for brevity in this scaffold
    )
    print(
        "  NOTE: CLI implementation will build the full JSON body with inline TOML config and stream events."
    )
