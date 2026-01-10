import os
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from synth_ai.core.process import popen_capture

# Note: Demo apps have been removed. DemoEnv type and related helpers are no longer available.


def is_modal_public_url(url: str | None) -> bool:
    try:
        candidate = (url or "").strip().lower()
        if not candidate or not (
            candidate.startswith("http://") or candidate.startswith("https://")
        ):
            return False
        return (
            (".modal.run" in candidate)
            and ("modal.local" not in candidate)
            and ("pypi-mirror" not in candidate)
        )
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


def ensure_task_app_ready(env: Any, synth_key: str, *, label: str) -> Any:
    """Ensure task app is ready for use.

    Note: Demo apps have been removed. This function is no longer available.
    """
    raise NotImplementedError(
        "ensure_task_app_ready is no longer available. Demo apps have been removed."
    )


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

    # Check Modal authentication status
    auth_ok = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))
    if auth_ok:
        print("✓ Modal authentication configured via environment variables")
    else:
        print("\n⚠️  Modal authentication may be required")
        print("\n   To authenticate Modal, run:")
        print("     modal setup")
        print("\n   Or set environment variables:")
        print("     export MODAL_TOKEN_ID=your-token-id")
        print("     export MODAL_TOKEN_SECRET=your-token-secret")
        print("\n   You can deploy later after authenticating.\n")
