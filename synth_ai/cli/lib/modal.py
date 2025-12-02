
import contextlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, urlunparse

from synth_ai.cli.lib.env import mask_str
from synth_ai.core.http import http_request
from synth_ai.core.process import popen_capture
from synth_ai.core.user_config import load_user_config

if TYPE_CHECKING:
    from synth_ai.cli.demo_apps.core import DemoEnv


def _get_demo_core():
    """Lazy import to avoid circular dependency."""
    from synth_ai.cli.demo_apps import core as demo_core
    return demo_core


def _get_default_task_app_secret_name():
    """Lazy import to avoid circular dependency."""
    from synth_ai.cli.demo_apps.core import DEFAULT_TASK_APP_SECRET_NAME
    return DEFAULT_TASK_APP_SECRET_NAME


def is_modal_public_url(url: str | None) -> bool:
    try:
        candidate = (url or "").strip().lower()
        if not candidate or not (candidate.startswith("http://") or candidate.startswith("https://")):
            return False
        return (".modal.run" in candidate) and ("modal.local" not in candidate) and ("pypi-mirror" not in candidate)
    except Exception:
        return False


def is_local_demo_url(url: str | None) -> bool:
    try:
        candidate = (url or "").strip().lower()
        if not candidate:
            return False
        return candidate.startswith("http://127.0.0.1") or candidate.startswith("http://localhost")
    except Exception:
        return False


def normalize_endpoint_url(url: str) -> str:
    """Convert loopback URLs to forms accepted by the backend."""
    if not url:
        return url
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if host in {"127.0.0.1", "::1"}:
            new_host = "localhost"
            netloc = new_host
            if parsed.port:
                netloc = f"{new_host}:{parsed.port}"
            if parsed.username:
                creds = parsed.username
                if parsed.password:
                    creds += f":{parsed.password}"
                netloc = f"{creds}@{netloc}"
            parsed = parsed._replace(netloc=netloc)
            return urlunparse(parsed)
    except Exception:
        pass
    return url


def find_asgi_apps(root: Path) -> list[Path]:
    """Recursively search for Python files that declare a Modal ASGI app."""
    results: list[Path] = []
    skip_dirs = {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "dist",
        "build",
        "__pycache__",
        ".ruff_cache",
        ".mypy_cache",
        "venv",
        ".venv",
    }
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            if not str(name).endswith(".py"):
                continue
            path = Path(dirpath) / name
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as fh:
                    txt = fh.read()
                if ("@asgi_app()" in txt) or ("@modal.asgi_app()" in txt):
                    results.append(path)
            except Exception:
                continue

    def _priority(path: Path) -> tuple[int, str]:
        rel = str(path.resolve())
        in_demo = "/synth_demo/" in rel or rel.endswith("/synth_demo/task_app.py")
        return (0 if in_demo else 1, rel)

    results.sort(key=_priority)
    return results


def ensure_task_app_ready(env: "DemoEnv", synth_key: str, *, label: str) -> "DemoEnv":
    demo_core = _get_demo_core()
    default_task_app_secret_name = _get_default_task_app_secret_name()
    persist_path = demo_core.load_demo_dir() or os.getcwd()
    user_config_map = load_user_config()

    env_key = (env.env_api_key or "").strip()
    if not env_key:
        raise RuntimeError(
            f"[{label}] ENVIRONMENT_API_KEY missing. Run `uvx synth-ai demo deploy` first."
        )

    template_id = demo_core.load_template_id()
    allow_local = template_id == "crafter-local"

    task_url = env.task_app_base_url
    url_ok = is_modal_public_url(task_url) or (allow_local and is_local_demo_url(task_url or ""))
    if not task_url or not url_ok:
        resolved = task_url or ""
        dynamic_lookup_allowed = env.task_app_name and not (
            allow_local and is_local_demo_url(task_url or "")
        )
        if dynamic_lookup_allowed and not is_modal_public_url(resolved):
            code, out = popen_capture(
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "modal",
                    "app",
                    "url",
                    env.task_app_name,
                ]
            )
            if code == 0 and out:
                for token in out.split():
                    if is_modal_public_url(token):
                        resolved = token.strip().rstrip("/")
                        break
        if dynamic_lookup_allowed and not is_modal_public_url(resolved):
            try:
                choice = (
                    input(
                        f"Resolve URL from Modal for app '{env.task_app_name}'? [Y/n]: "
                    ).strip().lower()
                    or "y"
                )
            except Exception:
                choice = "y"
            if choice.startswith("y"):
                code, out = popen_capture(
                    [
                        "uv",
                        "run",
                        "python",
                        "-m",
                        "modal",
                        "app",
                        "url",
                        env.task_app_name,
                    ]
                )
                if code == 0 and out:
                    for token in out.split():
                        if is_modal_public_url(token):
                            resolved = token.strip().rstrip("/")
                            break
        if not is_modal_public_url(resolved):
            hint = "Examples: https://<app-name>-fastapi-app.modal.run"
            if allow_local:
                hint += " or http://127.0.0.1:8001"
            print(f"[{label}] Task app URL not configured or not a valid target.")
            print(hint)
            entered = input(
                "Enter Task App base URL (must contain '.modal.run'), or press Enter to abort: "
            ).strip()
            if not entered:
                raise RuntimeError(f"[{label}] Task App URL is required.")
            entered_clean = entered.rstrip("/")
            if not (
                is_modal_public_url(entered_clean)
                or (allow_local and is_local_demo_url(entered_clean))
            ):
                raise RuntimeError(f"[{label}] Valid Task App URL is required.")
            task_url = entered_clean
        else:
            task_url = resolved
        demo_core.persist_task_url(task_url, name=(env.task_app_name or None), path=persist_path)

    app_name = (env.task_app_name or "").strip()
    requires_modal_name = is_modal_public_url(task_url)
    if requires_modal_name and not app_name:
        fallback = input("Enter Modal app name for the task app (required): ").strip()
        if not fallback:
            raise RuntimeError(f"[{label}] Task app name is required.")
        app_name = fallback
        demo_core.persist_task_url(task_url, name=app_name, path=persist_path)

    demo_core.persist_task_url(task_url, name=app_name if requires_modal_name else None, path=persist_path)
    if synth_key:
        os.environ["SYNTH_API_KEY"] = synth_key

    openai_key = (
        os.environ.get("OPENAI_API_KEY")
        or str(user_config_map.get("OPENAI_API_KEY") or "")
    ).strip()
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    print(f"[{label}] Verifying rollout health:")
    try:
        preview = mask_str(env_key)
        print(f"[{label}] {preview}")
    except Exception:
        pass
    health_base = task_url.rstrip("/")
    health_urls = [f"{health_base}/health/rollout", f"{health_base}/health"]
    rc = 0
    body: Any = ""
    for h in health_urls:
        print(f"[{label}] GET", h)
        rc, body = http_request("GET", h, headers={"X-API-Key": env_key})
        if rc == 200:
            break
    print(f"[{label}] status: {rc}")
    try:
        preview_body = json.dumps(body)[:800] if isinstance(body, dict) else str(body)[:800]
    except Exception:
        preview_body = str(body)[:800]
    print(f"[{label}] body:", preview_body)
    if rc != 200:
        print(f"[{label}] Warning: rollout health check failed ({rc}). Response: {body}")
        with contextlib.suppress(Exception):
            print(f"[{label}] Sent header X-API-Key → {mask_str(env_key)}")
    else:
        print(f"[{label}] Task app rollout health check OK.")

    os.environ["TASK_APP_BASE_URL"] = task_url
    os.environ["ENVIRONMENT_API_KEY"] = env_key
    os.environ["TASK_APP_SECRET_NAME"] = default_task_app_secret_name
    updated_env = demo_core.load_env()
    updated_env.env_api_key = env_key
    updated_env.task_app_base_url = task_url
    updated_env.task_app_name = app_name if requires_modal_name else ""
    updated_env.task_app_secret_name = default_task_app_secret_name
    return updated_env


def ensure_modal_installed() -> None:
    """Install the modal package if it is not already available and check authentication."""
    modal_installed = False
    try:
        import importlib.util as import_util

        if import_util.find_spec("modal") is not None:
            modal_installed = True
    except Exception:
        pass

    if not modal_installed:
        print("modal not found; installing…")
        try:
            if shutil.which("uv"):
                code, out = popen_capture(["uv", "pip", "install", "modal>=1.1.4"])
            else:
                code, out = popen_capture([sys.executable, "-m", "pip", "install", "modal>=1.1.4"])
            if code != 0:
                print(out)
                print("Failed to install modal; continuing may fail.")
                return
            print("✓ modal installed successfully")
            modal_installed = True
        except Exception as exc:
            print(f"modal install error: {exc}")
            return

    if modal_installed:
        try:
            import importlib.util as import_util

            if import_util.find_spec("modal") is None:
                print("Warning: modal is still not importable after install attempt.")
                return
        except Exception:
            print("Warning: unable to verify modal installation.")
            return

    demo_core = _get_demo_core()
    auth_ok, auth_msg = demo_core.modal_auth_status()
    if auth_ok:
        print(f"✓ Modal authenticated: {auth_msg}")
    else:
        print("\n⚠️  Modal authentication required")
        print(f"   Status: {auth_msg}")
        print("\n   To authenticate Modal, run:")
        print("     modal setup")
        print("\n   Or set environment variables:")
        print("     export MODAL_TOKEN_ID=your-token-id")
        print("     export MODAL_TOKEN_SECRET=your-token-secret")
        print("\n   You can deploy later after authenticating.\n")
