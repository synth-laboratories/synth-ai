import os
import re
import shlex
import subprocess
import tempfile
import textwrap
from pathlib import Path

import click
from modal.config import config

from synth_ai.core.cfgs import ModalDeployCfg
from synth_ai.core.paths import REPO_ROOT, cleanup_paths, configure_import_paths
from synth_ai.core.telemetry import log_error, log_info


def __validate_modal_app(*args, **kwargs):
    """Lazy import to avoid circular dependency."""
    from synth_ai.cli.lib.apps.modal_app import validate_modal_app
    return validate_modal_app(*args, **kwargs)


def __write_env_var_to_dotenv(key: str, value: str, **kwargs) -> None:
    """Lazy import to avoid circular dependency."""
    from synth_ai.cli.lib.env import write_env_var_to_dotenv
    write_env_var_to_dotenv(key, value, **kwargs)

MODAL_URL_REGEX = re.compile(r"https?://[^\s]+modal\.run[^\s]*")


def is_modal_setup_needed() -> bool:
    token_id = os.environ.get("MODAL_TOKEN_ID") \
        or config.get("token_id") \
        or ''
    token_secret = os.environ.get("MODAL_TOKEN_SECRET") \
        or config.get("token_secret") \
        or ''
    return not token_id or not token_secret


def run_modal_setup(modal_bin_path: Path) -> None:
    ctx: dict[str, object] = {"modal_bin_path": str(modal_bin_path)}
    log_info("running modal setup", ctx=ctx)
    cmd = [str(modal_bin_path), "setup"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as err:
        ctx["exit_code"] = err.returncode
        log_error("modal setup failed", ctx=ctx)
        raise RuntimeError(f"`{' '.join(cmd)}` exited with status {err.returncode}") from err


def create_modal_wrapper(cfg: ModalDeployCfg) -> tuple[Path, Path]:
    ctx: dict[str, object] = {
        "task_app_path": str(cfg.task_app_path),
        "modal_app_path": str(cfg.modal_app_path),
        "modal_bin_path": str(cfg.modal_bin_path),
    }
    log_info("creating modal wrapper", ctx=ctx)
    src = textwrap.dedent(f"""
        from importlib import util as _util
        from pathlib import Path as _Path
        import sys as _sys

        _source_dir = _Path({str(cfg.modal_app_path.parent.resolve())!r}).resolve()
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
        _ENVIRONMENT_API_KEY = {cfg.env_api_key!r}
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
    dir = Path(tempfile.mkdtemp(prefix="synth_modal_app"))
    file = dir / "__modal_wrapper__.py"
    file.write_text(src + '\n', encoding="utf-8")
    ctx["wrapper_dir"] = str(dir)
    log_info("modal wrapper created", ctx=ctx)
    return (dir, file)


def stream_modal_cmd_output(
    process: subprocess.Popen,
    print_stdout: bool = True
) -> tuple[int, str | None]:
    stdout = process.stdout
    if stdout is None:
        return process.wait(), None
    url: str | None = None
    for line in stdout:
        if print_stdout:
            print(line, end="")
        if url is None:
            match = MODAL_URL_REGEX.search(line)
            if not match:
                continue
            url = match.group(0).rstrip(".,")
            if not url:
                continue
            __write_env_var_to_dotenv("TASK_APP_URL", url, print_msg=False)
            log_info("modal deploy URL detected", ctx={"task_app_url": url})
    return process.wait(), url


def run_modal_cmd(cmd: list[str], env: dict[str, str]) -> subprocess.Popen:
    ctx: dict[str, object] = {"cmd": cmd}
    log_info("starting modal cli", ctx=ctx)
    try:
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
    except FileNotFoundError as exc:
        ctx["error"] = str(exc)
        log_error("modal cli not found", ctx=ctx)
        raise click.ClickException(f"Modal CLI not found at {cmd[0]}: {exc}") from exc
    except Exception as exc:
        ctx["error"] = str(exc)
        log_error("modal cli launch failed", ctx=ctx)
        raise click.ClickException(f"Failed to start Modal CLI ({cmd[0]}): {exc}") from exc


def deploy_app_modal(cfg: ModalDeployCfg, wait: bool = False) -> str | None:
    is_mcp = os.getenv("CTX") == "mcp"
    os.environ["ENVIRONMENT_API_KEY"] = cfg.env_api_key
    ctx: dict[str, object] = {
        "task_app_path": str(cfg.task_app_path),
        "modal_app_path": str(cfg.modal_app_path),
        "modal_bin_path": str(cfg.modal_bin_path),
        "cmd_arg": cfg.cmd_arg,
        "dry_run": cfg.dry_run,
        "modal_app_name": cfg.modal_app_name,
        "is_mcp": is_mcp,
        "wait": wait,
    }
    log_info("deploy_app_modal invoked", ctx=ctx)

    __validate_modal_app(cfg.modal_app_path)

    if is_modal_setup_needed():
        log_info("modal setup required", ctx=ctx)
        run_modal_setup(cfg.modal_bin_path)

    configure_import_paths(cfg.modal_app_path, REPO_ROOT)

    wrapper_dir, wrapper_file = create_modal_wrapper(cfg)

    cmd = [str(cfg.modal_bin_path), cfg.cmd_arg, str(wrapper_file)]
    if cfg.modal_app_name and cfg.cmd_arg == "deploy":
        cmd.extend(["--name", cfg.modal_app_name])
    ctx["cmd"] = cmd

    task_app_url: str | None = None
    try:
        if cfg.dry_run:
            cleanup_paths(file=wrapper_file, dir=wrapper_dir)
            print(f"deploy --runtime modal --dry-run → {shlex.join(cmd)}")
            log_info("modal dry-run completed", ctx=ctx)
            return None
        process = run_modal_cmd(cmd, os.environ.copy())
        ctx["pid"] = process.pid

        if wait:
            # Blocking mode: wait for process to complete
            if is_mcp:
                rc, task_app_url = stream_modal_cmd_output(process, print_stdout=False)
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, cmd)
                summary = f"[deploy_modal] modal {cfg.cmd_arg} completed"
                if task_app_url:
                    summary = f"{summary} → {task_app_url}"
                return summary
            print(f"{'-' * 31} Modal start {'-' * 31}")
            rc, task_app_url = stream_modal_cmd_output(process)
            if rc != 0:
                raise subprocess.CalledProcessError(rc, cmd)
            print(f"{'-' * 32} Modal end {'-' * 32}")
            if task_app_url:
                print(f"Your task app is live on Modal at: {task_app_url}")
            ctx["task_app_url"] = task_app_url
            log_info("modal deploy completed", ctx=ctx)
        else:
            # Non-blocking mode: start process and return immediately
            # Process will continue in background
            print(f"[deploy_modal] Starting modal {cfg.cmd_arg} in background (PID: {process.pid})")
            print("[deploy_modal] Process will continue running. Check logs for URL when ready.")
            # Don't wait for process - let it run in background
            # The URL will be written to .env when detected by the background process
            log_info("modal deploy started in background", ctx=ctx)
            return f"[deploy_modal] modal {cfg.cmd_arg} started in background (PID: {process.pid})"
    except subprocess.CalledProcessError as err:
        ctx["exit_code"] = err.returncode
        ctx["task_app_url"] = task_app_url
        log_error("modal deploy failed", ctx=ctx)
        raise RuntimeError(f"modal {cfg.cmd_arg} failed with exit code: {err.returncode}") from err
    finally:
        cleanup_paths(file=wrapper_file, dir=wrapper_dir)
