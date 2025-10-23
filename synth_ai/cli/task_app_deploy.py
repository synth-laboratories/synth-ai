"""Task app deploy command."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import click
from synth_ai.cli.lib.process import ensure_local_port_available
from synth_ai.cli.lib.task_app_discovery import select_app_choice
from synth_ai.cli.lib.task_app_env import ensure_env_credentials
from synth_ai.cli.lib.print_next_step_message import print_next_step_message
from synth_ai.cli.lib.user_config import load_user_env
from click.core import ParameterSource

from .task_apps import _maybe_use_demo_dir, _run_modal_script, _run_modal_with_entry, task_app_group


def _load_demo_context() -> tuple[Any, Any, str | None] | None:
    try:
        from synth_ai.demos import core as demo_core
    except Exception:
        return None

    try:
        env = demo_core.load_env()
        template_id = demo_core.load_template_id()
        return demo_core, env, template_id
    except Exception:
        return demo_core, None, None


def _derive_modal_name_from_url(url: str) -> str | None:
    try:
        from urllib.parse import urlparse

        host = urlparse(url).hostname or ""
        if "--" not in host:
            return None
        suffix = host.split("--", 1)[1]
        core = suffix.split('.modal', 1)[0]
        if core.endswith('-fastapi-app'):
            core = core[: -len('-fastapi-app')]
        return core.strip() or None
    except Exception:
        return None


def _persist_task_app_metadata(
    demo_core: Any | None,
    demo_env: Any | None,
    modal_name: str | None,
    task_app_url: str | None,
) -> None:
    if not task_app_url:
        return

    effective_name = modal_name or _derive_modal_name_from_url(task_app_url)

    persist_path: str | None = None
    if demo_core is not None:
        with contextlib.suppress(Exception):
            persist_path = demo_core.load_demo_dir()
    if not persist_path:
        with contextlib.suppress(Exception):
            persist_path = str(Path.cwd().resolve())

    if demo_core is None:
        with contextlib.suppress(Exception):
            from synth_ai.demos import core as demo_core  # type: ignore
    if demo_core is None:
        return

    demo_core.persist_task_url(task_app_url, name=effective_name, path=persist_path)
    if demo_env is not None:
        demo_env.task_app_base_url = task_app_url
        if effective_name:
            demo_env.task_app_name = effective_name
        demo_env.task_app_secret_name = demo_core.DEFAULT_TASK_APP_SECRET_NAME


def _deploy_with_legacy_script(
    script_path: Path,
    modal_name: str | None,
    demo_core: Any | None,
    demo_env: Any | None,
) -> None:
    if demo_core is None or demo_env is None:
        raise click.ClickException("--script requires a demo environment. Run `synth-ai demo init` first.")

    from synth_ai.demos.math.deploy_modal import deploy as modal_deploy

    env_api_key = (demo_env.env_api_key or "").strip()
    if not env_api_key:
        raise click.ClickException("ENVIRONMENT_API_KEY missing; run `synth-ai setup` first.")

    click.echo(f"Running legacy deploy script: {script_path}")
    url = modal_deploy(script_path=str(script_path), env_api_key=env_api_key)
    click.echo(f"✓ Task app URL: {url}")
    _persist_task_app_metadata(demo_core, demo_env, modal_name, url, [])


def _deploy_local_task_app(
    demo_core: Any,
    demo_env: Any,
    template_id: str | None,
) -> None:
    is_local_template = template_id == "crafter-local"

    os.environ["TASK_APP_SECRET_NAME"] = demo_core.DEFAULT_TASK_APP_SECRET_NAME

    click.echo("Starting local Task App…")
    cwd = os.getcwd()
    run_env = os.environ.copy()
    if is_local_template:
        run_env.setdefault("TASKAPP_TRACING_ENABLED", "1")
        traces_dir = os.path.join(cwd, "traces", "v3")
        run_env.setdefault("TASKAPP_SFT_OUTPUT_DIR", traces_dir)

    target = "http://127.0.0.1:8080"
    if is_local_template:
        task_app_path = os.path.join(cwd, "task_app.py")
        if not os.path.isfile(task_app_path):
            raise click.ClickException("Expected task_app.py in demo directory for Crafter template")
        target = "http://127.0.0.1:8001"
        if not ensure_local_port_available("127.0.0.1", 8001):
            return
        local_cmd = [
            sys.executable,
            task_app_path,
            "--host",
            "0.0.0.0",
            "--port",
            "8001",
        ]
    else:
        if not ensure_local_port_available("127.0.0.1", 8080):
            return
        local_cmd = [
            sys.executable,
            "-c",
            "from synth_ai.demos.math.app import run; run()",
        ]

    local_proc: subprocess.Popen[str] | None = None
    try:
        local_proc = subprocess.Popen(
            local_cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=cwd,
            env=run_env,
        )
        click.echo(
            "\nLocal server is running in this terminal. Leave this window open and run the next "
            "step from a new terminal.\nPress Ctrl+C here when you're ready to stop the server.\n"
        )
        url = ""
        for _ in range(60):
            if local_proc.poll() is not None:
                break
            if demo_core.assert_http_ok(
                target + "/health", method="GET"
            ) or demo_core.assert_http_ok(
                target, method="GET"
            ):
                url = target
                break
            time.sleep(1)
        if not url:
            raise click.ClickException("Failed to verify local task app health. See logs above.")

        demo_core.persist_task_url(url, name=None, path=cwd)
        if demo_env is not None:
            demo_env.task_app_base_url = url
            demo_env.task_app_name = ""
            demo_env.task_app_secret_name = demo_core.DEFAULT_TASK_APP_SECRET_NAME
        click.echo(f"TASK_APP_BASE_URL={url}")
        print_next_step_message(
            "export traces to SFT JSONL",
            [
                f"cd {cwd}",
                "uvx python export_trace_sft.py --db traces/v3/synth_ai.db --output demo_sft.jsonl",
            ],
        )

        click.echo("\nPress Ctrl+C here to stop the local server and exit this command.\n")
        try:
            local_proc.wait()
        except KeyboardInterrupt:
            click.echo("Stopping local server…")
            with contextlib.suppress(Exception):
                local_proc.send_signal(signal.SIGINT)
            try:
                local_proc.wait(timeout=10)
            except Exception:
                with contextlib.suppress(Exception):
                    local_proc.kill()
            click.echo("Local server stopped.")
    except Exception:
        if local_proc and local_proc.poll() is None:
            with contextlib.suppress(Exception):
                local_proc.kill()
        raise


@task_app_group.command("deploy")
@click.argument("app_id", type=str, required=False)
@click.option("--local", is_flag=True, help="Run local FastAPI instead of Modal deploy")
@click.option("--app", type=click.Path(), default=None, help="Path to Modal app.py for manual deploy")
@click.option("--script", type=click.Path(), default=None, help="Path to legacy deploy_task_app.sh script")
@click.option("--name", "modal_name", default=None, help="Override Modal app name")
@click.option("--dry-run", is_flag=True, help="Print Modal command without executing")
@click.option("--modal-cli", default="modal", help="Path to Modal CLI executable")
def deploy_command(
    app_id: str | None,
    local: bool,
    app: str | None,
    script: str | None,
    modal_name: str | None,
    dry_run: bool,
    modal_cli: str,
) -> None:
    """Deploy a task app locally or to Modal."""

    _maybe_use_demo_dir()
    load_user_env()
    ensure_env_credentials(require_synth=False)

    demo_context = _load_demo_context()
    demo_core = demo_env = template_id = None
    if demo_context:
        demo_core, demo_env, template_id = demo_context

    ctx = click.get_current_context()
    local_source = ctx.get_parameter_source("local")

    mode: str
    if app or script:
        mode = "modal"
    elif local_source == ParameterSource.COMMANDLINE:
        mode = "local" if local else "modal"
    else:
        default_mode = "local" if template_id == "crafter-local" else "modal"
        prompt = click.prompt(
            "Deploy target",
            type=click.Choice(["modal", "local"]),
            default=default_mode,
            show_choices=True,
        )
        mode = prompt
        click.echo(f"Deploying to {mode.upper()} target.\n")
    local = mode == "local"

    if local:
        if demo_core is None or demo_env is None:
            raise click.ClickException("--local deployment requires a demo environment. Run `synth-ai demo init` first.")
        _deploy_local_task_app(demo_core, demo_env, template_id)
        return

    if app and script:
        raise click.ClickException("Use either --app or --script, not both.")

    if script:
        script_path = Path(script).expanduser().resolve()
        if not script_path.exists():
            raise click.ClickException(f"Legacy deploy script not found: {script_path}")
        _deploy_with_legacy_script(script_path, modal_name, demo_core, demo_env)
        return

    if app:
        script_path = Path(app).expanduser().resolve()
        if not script_path.is_file():
            raise click.ClickException(f"App file not found: {script_path}")
        task_app_url = _run_modal_script(
            script_path,
            modal_cli,
            "deploy",
            modal_name=modal_name,
            dry_run=dry_run,
        )
        persist_name = modal_name or (getattr(demo_env, "task_app_name", None) if demo_env else None)
        _persist_task_app_metadata(demo_core, demo_env, persist_name, task_app_url)
        return

    choice = select_app_choice(app_id, purpose="deploy")

    if choice.modal_script:
        task_app_url = _run_modal_script(
            choice.modal_script,
            modal_cli,
            "deploy",
            modal_name=modal_name,
            dry_run=dry_run,
        )
        deploy_name = modal_name or (getattr(demo_env, "task_app_name", None) if demo_env else None)
    else:
        entry = choice.ensure_entry()
        if entry.modal is None:
            raise click.ClickException(
                f"Task app '{entry.app_id}' does not define Modal deployment settings"
            )
        deploy_name = modal_name or entry.modal.app_name
        task_app_url = _run_modal_with_entry(
            entry,
            entry.modal,
            modal_cli,
            deploy_name,
            "deploy",
            dry_run=dry_run,
            original_path=choice.path,
        )

    _persist_task_app_metadata(demo_core, demo_env, deploy_name, task_app_url)


__all__ = ["deploy_command"]
