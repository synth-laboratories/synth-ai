import os
import re
import shlex
import subprocess
from pathlib import Path

import click
from modal.config import config

from synth_ai.core.cfgs import ModalDeployCfg
from synth_ai.core.paths import REPO_ROOT, temporary_import_paths
from synth_ai.core.telemetry import log_error, log_info


def __validate_modal_app(*args, **kwargs):
    """Lazy import to avoid circular dependency."""
    from synth_ai.core.apps.modal_app import validate_modal_app

    return validate_modal_app(*args, **kwargs)


MODAL_URL_REGEX = re.compile(r"https?://[^\s]+modal\.run[^\s]*")


def is_modal_setup_needed() -> bool:
    token_id = os.environ.get("MODAL_TOKEN_ID") or config.get("token_id") or ""
    token_secret = os.environ.get("MODAL_TOKEN_SECRET") or config.get("token_secret") or ""
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


def stream_modal_cmd_output(
    process: subprocess.Popen, print_stdout: bool = True
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
            os.environ["SYNTH_LOCALAPI_URL"] = url
            try:
                from synth_ai.core.user_config import update_user_config

                update_user_config({"SYNTH_LOCALAPI_URL": url})
            except Exception:
                pass
            log_info("modal deploy URL detected", ctx={"localapi_url": url})
    return process.wait(), url


def run_modal_cmd(cmd: list[str], env: dict[str, str]) -> subprocess.Popen:
    ctx: dict[str, object] = {"cmd": cmd}
    log_info("starting modal cli", ctx=ctx)
    try:
        return subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env
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
    os.environ["ENVIRONMENT_API_KEY"] = cfg.localapi_key
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

    with temporary_import_paths(cfg.modal_app_path, REPO_ROOT):
        __validate_modal_app(cfg.modal_app_path)

    if is_modal_setup_needed():
        log_info("modal setup required", ctx=ctx)
        run_modal_setup(cfg.modal_bin_path)

    cmd = [str(cfg.modal_bin_path), cfg.cmd_arg, str(cfg.modal_app_path)]
    if cfg.modal_app_name and cfg.cmd_arg == "deploy":
        cmd.extend(["--name", cfg.modal_app_name])
    ctx["cmd"] = cmd

    localapi_url: str | None = None
    try:
        if cfg.dry_run:
            print(f"deploy --runtime modal --dry-run → {shlex.join(cmd)}")
            log_info("modal dry-run completed", ctx=ctx)
            return None
        process = run_modal_cmd(cmd, os.environ.copy())
        ctx["pid"] = process.pid

        if wait:
            # Blocking mode: wait for process to complete
            if is_mcp:
                rc, localapi_url = stream_modal_cmd_output(process, print_stdout=False)
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, cmd)
                summary = f"[deploy_modal] modal {cfg.cmd_arg} completed"
                if localapi_url:
                    summary = f"{summary} → {localapi_url}"
                return summary
            print(f"{'-' * 31} Modal start {'-' * 31}")
            rc, localapi_url = stream_modal_cmd_output(process)
            if rc != 0:
                raise subprocess.CalledProcessError(rc, cmd)
            print(f"{'-' * 32} Modal end {'-' * 32}")
            if localapi_url:
                print(f"Your LocalAPI is live on Modal at: {localapi_url}")
            ctx["localapi_url"] = localapi_url
            log_info("modal deploy completed", ctx=ctx)
        else:
            # Non-blocking mode: start process and return immediately
            # Process will continue in background
            print(f"[deploy_modal] Starting modal {cfg.cmd_arg} in background (PID: {process.pid})")
            print("[deploy_modal] Process will continue running. Check logs for URL when ready.")
            # Don't wait for process - let it run in background
            # The URL will be persisted to ~/.synth-ai when detected by the background process
            log_info("modal deploy started in background", ctx=ctx)
            return f"[deploy_modal] modal {cfg.cmd_arg} started in background (PID: {process.pid})"
    except subprocess.CalledProcessError as err:
        ctx["exit_code"] = err.returncode
        ctx["localapi_url"] = localapi_url
        log_error("modal deploy failed", ctx=ctx)
        raise RuntimeError(f"modal {cfg.cmd_arg} failed with exit code: {err.returncode}") from err
