from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from dataclasses import dataclass
from typing import Any

from synth_ai.core.env import PROD_BASE_URL_DEFAULT

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


def _state_path() -> str:
    return os.path.expanduser("~/.synth-ai/demo.json")


def _read_state() -> dict[str, Any]:
    try:
        path = _state_path()
        if os.path.isfile(path):
            with open(path) as fh:
                data = json.load(fh) or {}
                return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _write_state(data: dict[str, Any]) -> None:
    try:
        path = _state_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh)
    except Exception:
        pass


def load_dotenv_file(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        with open(path) as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return out


def _persist_dotenv_values(path: str, values: dict[str, str]) -> None:
    """Ensure ``values`` are present in ``path`` (.env style)."""

    try:
        existing_lines: list[str] = []
        if os.path.isfile(path):
            with open(path) as fh:
                existing_lines = fh.read().splitlines()
        else:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        mapping: dict[str, str] = {}
        order: list[str] = []
        for line in existing_lines:
            if not line or line.startswith("#") or "=" not in line:
                order.append(line)
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            mapping[key] = val
            order.append(key)
        for key, value in values.items():
            if key not in mapping:
                order.append(key)
            mapping[key] = value
        with open(path, "w") as fh:
            for item in order:
                if item in mapping:
                    fh.write(f"{item}={mapping[item]}\n")
                else:
                    fh.write(item + "\n")
            for key, value in values.items():
                if key not in order:
                    fh.write(f"{key}={value}\n")
    except Exception:
        # Best-effort; failure to persist shouldn't crash CLI usage.
        pass


def persist_dotenv_values(values: dict[str, str], *, cwd: str | None = None) -> str:
    path = os.path.join(cwd or os.getcwd(), ".env")
    _persist_dotenv_values(path, values)
    return path


def persist_env_api_key(key: str) -> None:
    data = _read_state()
    data["ENVIRONMENT_API_KEY"] = key
    _write_state(data)


def persist_demo_dir(demo_dir: str) -> None:
    """Store the demo directory path for subsequent commands."""
    data = _read_state()
    data["DEMO_DIR"] = demo_dir
    _write_state(data)


def load_demo_dir() -> str | None:
    """Load the stored demo directory path, if any."""
    data = _read_state()
    return data.get("DEMO_DIR")


def persist_env_file_path(env_path: str) -> None:
    """Store the .env file path for subsequent commands."""
    data = _read_state()
    data["ENV_FILE_PATH"] = env_path
    _write_state(data)


def load_env_file_path() -> str | None:
    """Load the stored .env file path, if any."""
    data = _read_state()
    return data.get("ENV_FILE_PATH")


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
    """Resolve environment with sane defaults and auto-detection.

    Backend URL:
      - Use BACKEND_OVERRIDE (any) from CWD .env if set
      - Else use DEV_BACKEND_URL from CWD .env ONLY if it's localhost/127.0.0.1 or :8000
      - Else default to production backend (PROD_BASE_URL_DEFAULT)

    API keys:
      - SYNTH_API_KEY from OS -> CWD .env -> repo .env -> pkg demo .env -> state
      - If still missing, auto-pick DEV/PROD key based on backend and persist

    TASK_APP_BASE_URL:
      - OS -> CWD .env -> repo .env -> pkg demo .env -> state
    """
    env = DemoEnv()

    os_env: dict[str, str] = dict(os.environ)

    # CWD .env
    cwd_env_path = os.path.join(os.getcwd(), ".env")
    cwd_env = load_dotenv_file(cwd_env_path)

    # Repo/package .envs (fallbacks)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    repo_env = load_dotenv_file(os.path.join(repo_root, ".env"))
    pkg_env = load_dotenv_file(
        os.path.join(repo_root, "synth_ai", "demos", "demo_task_apps", "math", ".env")
    )
    examples_env = load_dotenv_file(os.path.join(repo_root, "examples", "rl", ".env"))

    state = _read_state()

    default_root = PROD_BASE_URL_DEFAULT.rstrip("/")
    prod_default = f"{default_root}/api"

    # Backend URL resolution
    backend_override = (
        os_env.get("BACKEND_OVERRIDE")
        or cwd_env.get("BACKEND_OVERRIDE")
        or repo_env.get("BACKEND_OVERRIDE")
        or pkg_env.get("BACKEND_OVERRIDE")
        or examples_env.get("BACKEND_OVERRIDE")
        or ""
    ).strip()
    dev_env = (
        os_env.get("DEV_BACKEND_URL")
        or cwd_env.get("DEV_BACKEND_URL")
        or repo_env.get("DEV_BACKEND_URL")
        or pkg_env.get("DEV_BACKEND_URL")
        or ""
    ).strip()
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

    # API key selection
    synth_api_key = (
        os_env.get("SYNTH_API_KEY")
        or cwd_env.get("SYNTH_API_KEY")
        or repo_env.get("SYNTH_API_KEY")
        or pkg_env.get("SYNTH_API_KEY")
        or str(state.get("SYNTH_API_KEY") or "")
    )
    if not synth_api_key:
        mode = (
            "prod"
            if default_root in dev_url
            else ("local" if ("localhost" in dev_url or "127.0.0.1" in dev_url) else "dev")
        )
        if mode == "prod":
            synth_api_key = (
                os_env.get("PROD_SYNTH_API_KEY")
                or cwd_env.get("PROD_SYNTH_API_KEY")
                or repo_env.get("PROD_SYNTH_API_KEY")
                or pkg_env.get("PROD_SYNTH_API_KEY")
                or ""
            )
        else:
            synth_api_key = (
                os_env.get("DEV_SYNTH_API_KEY")
                or cwd_env.get("DEV_SYNTH_API_KEY")
                or repo_env.get("DEV_SYNTH_API_KEY")
                or pkg_env.get("DEV_SYNTH_API_KEY")
                or os_env.get("TESTING_LOCAL_SYNTH_API_KEY")
                or cwd_env.get("TESTING_LOCAL_SYNTH_API_KEY")
                or repo_env.get("TESTING_LOCAL_SYNTH_API_KEY")
                or pkg_env.get("TESTING_LOCAL_SYNTH_API_KEY")
                or ""
            )
        if synth_api_key:
            st = dict(state)
            st["SYNTH_API_KEY"] = synth_api_key
            _write_state(st)

    env_api_key = (
        os_env.get("ENVIRONMENT_API_KEY")
        or cwd_env.get("ENVIRONMENT_API_KEY")
        or repo_env.get("ENVIRONMENT_API_KEY")
        or pkg_env.get("ENVIRONMENT_API_KEY")
        or examples_env.get("ENVIRONMENT_API_KEY")
        or str(state.get("ENVIRONMENT_API_KEY") or "")
    )

    # Task app URL
    task_url = (
        os_env.get("TASK_APP_BASE_URL")
        or cwd_env.get("TASK_APP_BASE_URL")
        or repo_env.get("TASK_APP_BASE_URL")
        or pkg_env.get("TASK_APP_BASE_URL")
        or str(state.get("TASK_APP_BASE_URL") or "")
    )

    task_app_name = str(state.get("TASK_APP_NAME") or "")
    task_app_secret_name = str(state.get("TASK_APP_SECRET_NAME") or DEFAULT_TASK_APP_SECRET_NAME)

    env.dev_backend_url = dev_url.rstrip("/")
    env.synth_api_key = synth_api_key
    env.env_api_key = env_api_key
    env.task_app_base_url = task_url.rstrip("/")
    env.task_app_name = task_app_name
    env.task_app_secret_name = task_app_secret_name

    # Suppress environment echo by default for cleaner CLI output.
    # If needed for debugging, set SYNTH_CLI_VERBOSE=1 to print resolved values.
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


def deploy_modal_math(env: DemoEnv) -> str:
    """Deploy Math Task App to Modal using in-repo deploy script; return public URL."""
    # Prefer the script colocated under demo_task_apps/math relative to this file
    this_dir = os.path.dirname(__file__)
    demo_script = os.path.join(this_dir, "math", "deploy_task_app.sh")
    # Fallback to top-level examples path if needed (repo root heuristic)
    repo_root = os.path.abspath(os.path.join(this_dir, "../../.."))
    fallback_script = os.path.join(repo_root, "examples", "rl", "deploy_task_app.sh")
    script = demo_script if os.path.isfile(demo_script) else fallback_script
    if not os.path.isfile(script):
        raise RuntimeError(f"deploy_task_app.sh not found at {demo_script} or {fallback_script}")

    envp = os.environ.copy()
    if env.env_api_key:
        envp["ENVIRONMENT_API_KEY"] = env.env_api_key
    print(f"Deploying Math Task App to Modal using: {script}")
    subprocess.check_call(["bash", script], cwd=os.path.dirname(script), env=envp)

    # Read last deploy log for URL
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


def persist_task_url(url: str, *, name: str | None = None) -> None:
    data = _read_state()
    changed: list[str] = []
    if data.get("TASK_APP_BASE_URL") != url:
        data["TASK_APP_BASE_URL"] = url
        changed.append("TASK_APP_BASE_URL")
    if name:
        if data.get("TASK_APP_NAME") != name:
            data["TASK_APP_NAME"] = name
            changed.append("TASK_APP_NAME")
        if data.get("TASK_APP_SECRET_NAME") != DEFAULT_TASK_APP_SECRET_NAME:
            data["TASK_APP_SECRET_NAME"] = DEFAULT_TASK_APP_SECRET_NAME
            changed.append("TASK_APP_SECRET_NAME")
    elif data.get("TASK_APP_SECRET_NAME") != DEFAULT_TASK_APP_SECRET_NAME:
        data["TASK_APP_SECRET_NAME"] = DEFAULT_TASK_APP_SECRET_NAME
        changed.append("TASK_APP_SECRET_NAME")
    _write_state(data)
    if changed:
        print(f"Saved {', '.join(changed)} to {_state_path()}")
        if "TASK_APP_SECRET_NAME" in changed:
            print(f"TASK_APP_SECRET_NAME={DEFAULT_TASK_APP_SECRET_NAME}")


def persist_api_key(key: str) -> None:
    data = _read_state()
    data["SYNTH_API_KEY"] = key
    _write_state(data)


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
