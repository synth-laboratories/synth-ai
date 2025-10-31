import ast
import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import click
from modal.config import config
from synth_ai.demos import core as demo_core
from synth_ai.demos.core import DEFAULT_TASK_APP_SECRET_NAME, DemoEnv
from synth_ai.task_app_cfgs import ModalTaskAppConfig

from .env import mask_str, resolve_env_var, write_env_var_to_dotenv
from .http import http_request
from .process import popen_capture
from .user_config import load_user_config

__all__ = [
    "ensure_modal_installed",
    "ensure_task_app_ready",
    "find_asgi_apps",
    "is_local_demo_url",
    "is_modal_public_url",
    "normalize_endpoint_url",
]


REPO_ROOT = Path(__file__).resolve().parents[2]

START_DIV = f"{'-' * 31} Modal start {'-' * 31}"
END_DIV = f"{'-' * 32} Modal end {'-' * 32}"
MODAL_URL_REGEX = re.compile(r"https?://[^\s]+modal\.run[^\s]*")


def get_default_modal_bin_path() -> Path | None:
    resolved = shutil.which("modal")
    return Path(resolved) if resolved else None


def ensure_py_file_defines_modal_app(file_path: Path) -> None:
    if file_path.suffix != ".py":
        raise TypeError()
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    except OSError as exc:
        raise OSError() from exc

    app_aliases: set[str] = set()
    modal_aliases: set[str] = set()

    def literal_name(call: ast.Call) -> str | None:
        for kw in call.keywords:
            if (
                kw.arg in {"name", "app_name"}
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                return kw.value.value
        if call.args:
            first = call.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                return first.value
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "modal":
            for alias in node.names:
                if alias.name == "App":
                    app_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "modal":
                    modal_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in app_aliases:
                if literal_name(node):
                    return None
            elif (
                isinstance(func, ast.Attribute)
                and func.attr == "App"
                and isinstance(func.value, ast.Name)
                and func.value.id in modal_aliases
                and literal_name(node)
            ):
                return None
    raise ValueError()


def run_modal_setup(modal_bin_path: Path) -> None:
    
    print("\n🌐 Connecting to your Modal account via https://modal.com")
    print(START_DIV)
    cmd = [str(modal_bin_path), "setup"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(END_DIV)
        raise RuntimeError(
            f"`{' '.join(cmd)}` exited with status {exc.returncode}"
            f"Run `{' '.join(cmd)} manually to inspect output"
        ) from exc
    print(END_DIV)
    print("✅ Connected to your Modal account")


def ensure_modal_config() -> None:
    token_id = os.environ.get("MODAL_TOKEN_ID") \
        or config.get("token_id") \
        or ''
    token_secret = os.environ.get("MODAL_TOKEN_SECRET") \
        or config.get("token_secret") \
        or ''
    if token_id and token_secret:
        print(f"Found Modal token_id={mask_str(token_id)}")
        print(f"Found Modal token_secret={mask_str(token_secret)}")
        return
    
    modal_bin_path = get_default_modal_bin_path()
    if not modal_bin_path:
        raise RuntimeError("Modal CLI not found on PATH")
    run_modal_setup(modal_bin_path)


def deploy_modal_app(cfg: ModalTaskAppConfig) -> None:
    ensure_py_file_defines_modal_app(cfg.modal_app_path)
    ensure_modal_config()
    
    py_paths: list[str] = []

    source_dir = cfg.modal_app_path.parent.resolve()
    py_paths.append(str(source_dir))
    if (source_dir / "__init__.py").exists():  # if the modal app lives in a package, ensure the parent package is importable
        py_paths.append(str(source_dir.parent.resolve()))

    py_paths.append(str(REPO_ROOT))
    
    env_api_key = resolve_env_var("ENVIRONMENT_API_KEY")
    if not os.environ["ENVIRONMENT_API_KEY"]:
        raise RuntimeError()
    
    env_copy = os.environ.copy()
    existing_python_path = env_copy.get("PYTHONPATH")
    if existing_python_path:
        py_paths.append(existing_python_path)
    unique_python_paths = list(dict.fromkeys(py_paths))
    env_copy["PYTHONPATH"] = os.pathsep.join(unique_python_paths)
    if "PYTHONPATH" in env_copy: # ensure wrapper has access to synth source for intra-repo imports
        env_copy["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT)] + env_copy["PYTHONPATH"].split(os.pathsep)
        )
    else:
        env_copy["PYTHONPATH"] = str(REPO_ROOT)

    modal_app_dir = cfg.modal_app_path.parent.resolve()
    tmp_root = Path(tempfile.mkdtemp(prefix="synth_modal_app"))
    wrapper_src = textwrap.dedent(f"""
        from importlib import util as _util
        from pathlib import Path as _Path
        import sys as _sys

        _source_dir = _Path({str(modal_app_dir)!r}).resolve()
        _module_path = _source_dir / {cfg.modal_app_path.name!r}
        _package_name = _source_dir.name
        _repo_root = _Path({str(REPO_ROOT)!r}).resolve()
        _synth_dir = _repo_root / "synth_ai"

        for _path in (str(_source_dir), str(_source_dir.parent), str(_repo_root)):
            if _path not in _sys.path:
                _sys.path.insert(0, _path)

        _spec = _util.spec_from_file_location("_synth_modal_target", str(_module_path))
        if _spec is None or _spec.loader is None:
            raise SystemExit("Unable to load modal task app from {cfg.modal_app_path}")
        _module = _util.module_from_spec(_spec)
        _sys.modules.setdefault("_synth_modal_target", _module)
        _spec.loader.exec_module(_module)

        try:
            from modal import App as _ModalApp
            from modal import Image as _ModalImage
        except Exception:
            _ModalApp = None  # type: ignore[assignment]
            _ModalImage = None  # type: ignore[assignment]

        def _apply_local_mounts(image):
            if _ModalImage is None or not isinstance(image, _ModalImage):
                return image
            mounts = [
                (str(_source_dir), f"/root/{{_package_name}}"),
                (str(_synth_dir), "/root/synth_ai"),
            ]
            for local_path, remote_path in mounts:
                try:
                    image = image.add_local_dir(local_path, remote_path=remote_path)
                except Exception:
                    pass
            return image

        if hasattr(_module, "image"):
            _module.image = _apply_local_mounts(getattr(_module, "image"))

        _candidate = getattr(_module, "app", None)
        if _ModalApp is None or not isinstance(_candidate, _ModalApp):
            candidate_modal_app = getattr(_module, "modal_app", None)
            if _ModalApp is not None and isinstance(candidate_modal_app, _ModalApp):
                _candidate = candidate_modal_app
                setattr(_module, "app", _candidate)

        if _ModalApp is not None and not isinstance(_candidate, _ModalApp):
            raise SystemExit(
                "Modal task app must expose an 'app = modal.App(...)' (or modal_app) attribute."
            )

        try:
            from modal import Secret as _Secret
        except Exception:
            _Secret = None

        for remote_path in ("/root/synth_ai", f"/root/{{_package_name}}"):
            if remote_path not in _sys.path:
                _sys.path.insert(0, remote_path)

        globals().update({{k: v for k, v in vars(_module).items() if not k.startswith("__")}})
        app = getattr(_module, "app")
        _ENVIRONMENT_API_KEY = {env_api_key!r}
        if _Secret is not None and _ENVIRONMENT_API_KEY:
            try:
                _inline_secret = _Secret.from_dict({{"ENVIRONMENT_API_KEY": _ENVIRONMENT_API_KEY}})
            except Exception:
                _inline_secret = None
            if _inline_secret is not None:
                try:
                    _decorators = list(getattr(app, "_function_decorators", []))
                except Exception:
                    _decorators = []
                for _decorator in _decorators:
                    _existing = getattr(_decorator, "secrets", None)
                    if not _existing:
                        continue
                    try:
                        if _inline_secret not in _existing:
                            _decorator.secrets = list(_existing) + [_inline_secret]
                    except Exception:
                        pass
    """).strip()
    wrapper_path = tmp_root / "__modal_wrapper__.py"
    wrapper_path.write_text(wrapper_src + '\n', encoding="utf-8")
    wrapper_info = (wrapper_path, tmp_root)

    cmd = [str(cfg.modal_bin_path), cfg.cmd_arg, str(wrapper_path)]
    if cfg.task_app_name and cfg.cmd_arg == "deploy":
        cmd.extend(["--name", cfg.task_app_name])
    
    msg = " ".join(shlex.quote(c) for c in cmd)
    if cfg.dry_run:
        print("Dry run:\n", msg)
        return
    print(f"Running:\n{msg}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env_copy
        )
        task_app_url = None
        assert process.stdout is not None
        print(START_DIV)
        for line in process.stdout:
            click.echo(line, nl=False)
            if task_app_url is None:
                match = MODAL_URL_REGEX.search(line)
                if match:
                    task_app_url = match.group(0).rstrip(".,")
                    if task_app_url:
                        write_env_var_to_dotenv(
                            "TASK_APP_URL",
                            task_app_url,
                            print_msg=True,
                            mask_msg=False,
                        )
        print(END_DIV)
        rc = process.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"modal {cfg.cmd_arg} failed with exit code: {exc.returncode}"
        ) from exc
    finally:
        if wrapper_info is not None:
            wrapper_path, tmp_root = wrapper_info
            wrapper_path.unlink(missing_ok=True)
        shutil.rmtree(tmp_root, ignore_errors=True)


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
            if not name.endswith(".py"):
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


def ensure_task_app_ready(env: DemoEnv, synth_key: str, *, label: str) -> DemoEnv:
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
    os.environ["TASK_APP_SECRET_NAME"] = DEFAULT_TASK_APP_SECRET_NAME
    updated_env = demo_core.load_env()
    updated_env.env_api_key = env_key
    updated_env.task_app_base_url = task_url
    updated_env.task_app_name = app_name if requires_modal_name else ""
    updated_env.task_app_secret_name = DEFAULT_TASK_APP_SECRET_NAME
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
