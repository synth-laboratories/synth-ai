import os
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import click
from synth_ai.demos import core as demo_core
from synth_ai.utils.modal import is_modal_public_url
from synth_ai.utils.process import popen_capture

from .errors import (
    DeployCliError,
    EnvFileDiscoveryError,
    EnvironmentKeyLoadError,
    EnvKeyPreflightError,
    MissingEnvironmentApiKeyError,
    ModalCliResolutionError,
    ModalExecutionError,
    TaskAppNotFoundError,
)

try:
    from synth_ai.cli.commands.help import DEPLOY_HELP
except ImportError:
    DEPLOY_HELP = """
Deploy a Synth AI task app locally or to Modal.

OVERVIEW
--------
The deploy command supports two runtimes:
  • modal: Deploy to Modal's cloud platform (default)
  • uvicorn: Run locally with FastAPI/Uvicorn

BASIC USAGE
-----------
  # Deploy to Modal (production)
  uvx synth-ai deploy

  # Deploy specific task app
  uvx synth-ai deploy my-math-app

  # Run locally for development
  uvx synth-ai deploy --runtime=uvicorn --port 8001

MODAL DEPLOYMENT
----------------
Modal deployment requires:
  1. Modal authentication (run: modal token new)
  2. ENVIRONMENT_API_KEY (run: uvx synth-ai setup)

Options:
  --modal-mode [deploy|serve]  Use 'deploy' for production (default),
                                'serve' for ephemeral development
  --name TEXT                   Override Modal app name
  --dry-run                     Preview the deploy command without executing
  --env-file PATH               Env file(s) to load (can be repeated)

Examples:
  # Standard production deployment
  uvx synth-ai deploy --runtime=modal

  # Deploy with custom name
  uvx synth-ai deploy --runtime=modal --name my-task-app-v2

  # Preview deployment command
  uvx synth-ai deploy --dry-run

  # Deploy with custom env file
  uvx synth-ai deploy --env-file .env.production

LOCAL DEVELOPMENT
-----------------
Run locally with auto-reload and tracing:

  uvx synth-ai deploy --runtime=uvicorn --port 8001 --reload

Options:
  --host TEXT                   Bind address (default: 0.0.0.0)
  --port INTEGER                Port number (prompted if not provided)
  --reload/--no-reload          Enable auto-reload on code changes
  --force/--no-force            Kill existing process on port
  --trace PATH                  Enable tracing to directory (default: traces/v3)
  --trace-db PATH               SQLite DB for traces

Examples:
  # Basic local server
  uvx synth-ai deploy --runtime=uvicorn

  # Development with auto-reload
  uvx synth-ai deploy --runtime=uvicorn --reload --port 8001

  # With custom trace directory
  uvx synth-ai deploy --runtime=uvicorn --trace ./my-traces

TROUBLESHOOTING
---------------
Common issues:

1. "ENVIRONMENT_API_KEY is required"
   → Run: uvx synth-ai setup

2. "Modal CLI not found"
   → Install: pip install modal
   → Authenticate: modal token new

3. "Task app not found"
   → Check app_id matches your task_app.py configuration
   → Run: uvx synth-ai task-app list (if available)

4. "Port already in use" (uvicorn)
   → Use --force to kill existing process
   → Or specify different --port

5. "No env file discovered"
   → Create .env file with required keys
   → Or pass --env-file explicitly

ENVIRONMENT VARIABLES
---------------------
  SYNTH_API_KEY              Your Synth platform API key
  ENVIRONMENT_API_KEY        Task environment authentication
  TASK_APP_BASE_URL          Base URL for deployed task app
  DEMO_DIR                   Demo directory path
  SYNTH_DEMO_DIR             Alternative demo directory

For more information: https://docs.usesynth.ai/deploy
"""  # type: ignore[assignment]

try:  # Click >= 8.1
    from click.core import ParameterSource
except ImportError:  # pragma: no cover - fallback for older versions
    ParameterSource = None  # type: ignore[assignment]

__all__ = [
    "command",
    "get_command",
    "modal_serve_command",
    "register_task_app_commands",
    "run_modal_runtime",
    "run_uvicorn_runtime",
]


def _translate_click_exception(err: click.ClickException) -> DeployCliError | None:
    message = getattr(err, "message", str(err)).strip()
    lower = message.lower()

    def _missing_env_hint() -> str:
        return (
            "Run `uvx synth-ai setup` to mint credentials or pass --env-file pointing to a file "
            "with ENVIRONMENT_API_KEY."
        )

    if "environment_api_key missing" in lower:
        return MissingEnvironmentApiKeyError(hint=_missing_env_hint())
    if "environment api key is required" in lower:
        return MissingEnvironmentApiKeyError(hint=_missing_env_hint())
    if "failed to load environment_api_key from generated .env" in lower:
        return EnvironmentKeyLoadError()

    if message.startswith("Env file not found:"):
        path = message.split(":", 1)[1].strip()
        return EnvFileDiscoveryError(attempted=(path,), hint=_missing_env_hint())
    if "env file required (--env-file) for this task app" in lower:
        return EnvFileDiscoveryError(hint=_missing_env_hint())
    if message.startswith("No .env file discovered automatically"):
        return EnvFileDiscoveryError(hint=_missing_env_hint())
    if message == "No environment values found":
        return EnvFileDiscoveryError(hint=_missing_env_hint())

    if message.startswith("Task app '") and " not found. Available:" in message:
        try:
            before, after = message.split(" not found. Available:", 1)
            app_id = before.split("Task app '", 1)[1].rstrip("'")
            available = tuple(item.strip() for item in after.split(",") if item.strip())
        except Exception:
            app_id = None
            available = ()
        return TaskAppNotFoundError(app_id=app_id, available=available)
    if message == "No task apps discovered for this command.":
        return TaskAppNotFoundError()

    if "modal cli not found" in lower:
        return ModalCliResolutionError(detail=message)
    if "--modal-cli path does not exist" in lower or "--modal-cli is not executable" in lower:
        return ModalCliResolutionError(detail=message)
    if "modal cli resolution found the synth-ai shim" in lower:
        return ModalCliResolutionError(detail=message)

    if message.startswith("modal ") and "failed with exit code" in message:
        parts = message.split(" failed with exit code ")
        command = parts[0].replace("modal ", "").strip() if len(parts) > 1 else "deploy"
        exit_code = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
        return ModalExecutionError(command=command, exit_code=exit_code)

    if message.startswith("[CRITICAL] ") or "[CRITICAL]" in message:
        return EnvKeyPreflightError(detail=message.removeprefix("[CRITICAL] ").strip())

    return None


def _format_deploy_error(err: DeployCliError) -> str:
    if isinstance(err, MissingEnvironmentApiKeyError):
        hint = err.hint or "Provide ENVIRONMENT_API_KEY via --env-file or run `uvx synth-ai setup`."
        return f"ENVIRONMENT_API_KEY is required. {hint}"
    if isinstance(err, EnvironmentKeyLoadError):
        base = "Failed to persist or reload ENVIRONMENT_API_KEY"
        if err.path:
            base += f" from {err.path}"
        return f"{base}. Regenerate the env file with `uvx synth-ai setup` or edit it manually."
    if isinstance(err, EnvFileDiscoveryError):
        attempted = ", ".join(err.attempted) if err.attempted else "No env files located"
        hint = err.hint or "Pass --env-file explicitly or run `uvx synth-ai setup`."
        return f"Unable to locate a usable env file ({attempted}). {hint}"
    if isinstance(err, TaskAppNotFoundError):
        available = ", ".join(err.available) if err.available else "no registered apps"
        app_id = err.app_id or "requested app"
        return f"Could not find task app '{app_id}'. Available choices: {available}."
    if isinstance(err, ModalCliResolutionError):
        detail = err.detail or "Modal CLI could not be resolved."
        return (
            f"{detail} Install the `modal` package or pass --modal-cli with the path to the Modal binary."
        )
    if isinstance(err, ModalExecutionError):
        return (
            f"Modal {err.command} exited with status {err.exit_code}. "
            "Review the Modal output above or rerun with --dry-run."
        )
    if isinstance(err, EnvKeyPreflightError):
        detail = err.detail or "Failed to upload ENVIRONMENT_API_KEY to the backend."
        return f"{detail} Ensure SYNTH_API_KEY is set and retry `uvx synth-ai setup`."
    return str(err)


@lru_cache(maxsize=1)
def _task_apps_module():
    from synth_ai.cli import task_apps as module  # local import to avoid circular deps

    return module


def _maybe_fix_task_url(modal_name: str | None = None, demo_dir: str | None = None) -> None:
    """Look up the Modal public URL and persist it to the task app config if needed."""
    env = demo_core.load_env()
    task_app_name = modal_name or env.task_app_name
    if not task_app_name:
        return
    current = env.task_app_base_url
    needs_lookup = not current or not is_modal_public_url(current)
    if not needs_lookup:
        return
    code, out = popen_capture(
        [
            "uv",
            "run",
            "python",
            "-m",
            "modal",
            "app",
            "url",
            task_app_name,
        ]
    )
    if code != 0 or not out:
        return
    new_url = ""
    for token in out.split():
        if is_modal_public_url(token):
            new_url = token.strip().rstrip("/")
            break
    if new_url and new_url != current:
        click.echo(f"Updating TASK_APP_BASE_URL from Modal CLI → {new_url}")
        persist_path = demo_dir or os.getcwd()
        demo_core.persist_task_url(new_url, name=task_app_name, path=persist_path)
        os.environ["TASK_APP_BASE_URL"] = new_url


def run_uvicorn_runtime(
    app_id: str | None,
    host: str,
    port: int | None,
    env_file: Sequence[str],
    reload_flag: bool,
    force: bool,
    trace_dir: str | None,
    trace_db: str | None,
) -> None:
    module = _task_apps_module()

    if not host:
        host = "0.0.0.0"

    try:
        demo_dir_path = module._load_demo_directory()
        if demo_dir_path:
            if not demo_dir_path.is_dir():
                raise click.ClickException(
                    f"Demo directory not found: {demo_dir_path}\nRun 'synth-ai setup' to create a demo."
                )
            os.chdir(demo_dir_path)
            click.echo(f"Using demo directory: {demo_dir_path}\n")
            os.environ["SYNTH_DEMO_DIR"] = str(demo_dir_path.resolve())

        if port is None:
            port = click.prompt("Port to serve on", type=int, default=8001)

        auto_trace = os.getenv("SYNTH_AUTO_TRACE", "1")
        auto_trace_enabled = auto_trace not in {"0", "false", "False", ""}

        if trace_dir is None:
            if auto_trace_enabled:
                demo_base = Path(os.environ.get("SYNTH_DEMO_DIR") or Path.cwd())
                default_trace_dir = (demo_base / "traces" / "v3").resolve()
                try:
                    default_trace_dir.mkdir(parents=True, exist_ok=True)
                except Exception as exc:
                    raise click.ClickException(
                        f"Failed to prepare trace directory {default_trace_dir}: {exc}"
                    ) from exc
                trace_dir = str(default_trace_dir)
                click.echo(f"[trace] Using trace directory: {trace_dir}")
            else:
                click.echo(
                    "\nTracing captures rollout data (actions, rewards, model outputs) to a local SQLite DB."
                )
                click.echo("This data can be exported to JSONL for supervised fine-tuning (SFT).")
                enable_tracing = click.confirm("Enable tracing?", default=True)
                if enable_tracing:
                    demo_base = Path(os.environ.get("SYNTH_DEMO_DIR") or Path.cwd())
                    default_trace_dir = str((demo_base / "traces/v3").resolve())
                    trace_dir = click.prompt(
                        "Trace directory", type=str, default=default_trace_dir, show_default=True
                    )
                else:
                    trace_dir = None

        if trace_dir and trace_db is None:
            if auto_trace_enabled:
                demo_base = Path(os.environ.get("SYNTH_DEMO_DIR") or Path.cwd())
                default_trace_db = (demo_base / "traces" / "v3" / "synth_ai.db").resolve()
                try:
                    default_trace_db.parent.mkdir(parents=True, exist_ok=True)
                except Exception as exc:
                    raise click.ClickException(
                        f"Failed to prepare trace DB directory {default_trace_db.parent}: {exc}"
                    ) from exc
                trace_db = str(default_trace_db)
                click.echo(f"[trace] Using trace DB: {trace_db}")
            else:
                demo_base = Path(os.environ.get("SYNTH_DEMO_DIR") or Path.cwd())
                default_trace_db = str((demo_base / "traces/v3/synth_ai.db").resolve())
                trace_db = click.prompt(
                    "Trace DB path", type=str, default=default_trace_db, show_default=True
                )

        choice = module._select_app_choice(app_id, purpose="serve")
        entry = choice.ensure_entry()
        module._serve_entry(
            entry,
            host,
            port,
            env_file,
            reload_flag,
            force,
            trace_dir=trace_dir,
            trace_db=trace_db,
        )
    except DeployCliError:
        raise
    except click.ClickException as err:
        converted = _translate_click_exception(err)
        if converted:
            raise converted from err
        raise


def run_modal_runtime(
    app_id: str | None,
    *,
    command: Literal["deploy", "serve"],
    modal_name: str | None,
    dry_run: bool,
    modal_cli: str,
    env_file: Sequence[str],
    use_demo_dir: bool = True,
) -> None:
    module = _task_apps_module()

    try:
        demo_dir_path = None
        if use_demo_dir:
            demo_dir_path = module._load_demo_directory()
            if demo_dir_path:
                if not demo_dir_path.is_dir():
                    raise click.ClickException(
                        f"Demo directory not found: {demo_dir_path}\nRun 'synth-ai demo' to create a demo."
                    )
                os.chdir(demo_dir_path)
                click.echo(f"Using demo directory: {demo_dir_path}\n")

        purpose = "modal-serve" if command == "serve" else "deploy"
        choice = module._select_app_choice(app_id, purpose=purpose)

        if choice.modal_script:
            env_paths = module._resolve_env_paths_for_script(choice.modal_script, env_file)
            click.echo("Using env file(s): " + ", ".join(str(p.resolve()) for p in env_paths))
            module._run_modal_script(
                choice.modal_script,
                modal_cli,
                command,
                env_paths,
                modal_name=modal_name,
                dry_run=dry_run if command == "deploy" else False,
            )
            if command == "deploy" and not dry_run:
                _maybe_fix_task_url(
                    modal_name=modal_name,
                    demo_dir=str(demo_dir_path) if demo_dir_path else None
                )
            return

        entry = choice.ensure_entry()
        if command == "serve":
            click.echo(f"[modal-serve] serving entry {entry.app_id} from {choice.path}")
            module._modal_serve_entry(entry, modal_name, modal_cli, env_file, original_path=choice.path)
        else:
            module._deploy_entry(entry, modal_name, dry_run, modal_cli, env_file, original_path=choice.path)
            if not dry_run:
                _maybe_fix_task_url(
                    modal_name=modal_name,
                    demo_dir=str(demo_dir_path) if demo_dir_path else None
                )
    except DeployCliError:
        raise
    except click.ClickException as err:
        converted = _translate_click_exception(err)
        if converted:
            raise converted from err
        raise


@click.command(
    "deploy",
    help=DEPLOY_HELP,
    epilog="Run 'uvx synth-ai deploy --help' for detailed usage information.",
)
@click.argument("app_id", type=str, required=False)
@click.option(
    "--runtime",
    type=click.Choice(["modal", "uvicorn"], case_sensitive=False),
    default="modal",
    show_default=True,
    help="Runtime to execute: 'modal' for remote Modal jobs, 'uvicorn' for the local FastAPI server.",
)
@click.option("--name", "modal_name", default=None, help="Override Modal app name")
@click.option("--dry-run", is_flag=True, help="Print modal deploy command without executing")
@click.option("--modal-cli", default="modal", help="Path to modal CLI executable")
@click.option(
    "--modal-mode",
    type=click.Choice(["deploy", "serve"], case_sensitive=False),
    default="deploy",
    show_default=True,
    help="Modal operation to run when --runtime=modal.",
)
@click.option(
    "--env-file",
    multiple=True,
    type=click.Path(),
    help="Env file to load into the container (can be repeated)",
)
@click.option("--host", default="0.0.0.0", show_default=True, help="Host for --runtime=uvicorn")
@click.option("--port", default=None, type=int, help="Port to serve on when --runtime=uvicorn")
@click.option(
    "--reload/--no-reload",
    "reload_flag",
    default=False,
    help="Enable uvicorn auto-reload when --runtime=uvicorn",
)
@click.option(
    "--force/--no-force",
    "force",
    default=False,
    help="Kill any process already bound to the selected port (uvicorn runtime)",
)
@click.option(
    "--trace",
    "trace_dir",
    type=click.Path(),
    default=None,
    help="Enable tracing and write SFT JSONL files when --runtime=uvicorn (default: traces/v3).",
)
@click.option(
    "--trace-db",
    "trace_db",
    type=click.Path(),
    default=None,
    help="Override local trace DB path when --runtime=uvicorn (default: traces/v3/synth_ai.db).",
)
def deploy_command(
    app_id: str | None,
    runtime: str,
    modal_name: str | None,
    dry_run: bool,
    modal_cli: str,
    modal_mode: str,
    env_file: Sequence[str],
    host: str,
    port: int | None,
    reload_flag: bool,
    force: bool,
    trace_dir: str | None,
    trace_db: str | None,
) -> None:
    """Deploy a task app locally or on Modal.
    
    This command deploys your Synth AI task app either to Modal's cloud platform
    or runs it locally with Uvicorn for development. Use --help for detailed usage.
    """

    runtime_normalized = runtime.lower()
    modal_mode_normalized = modal_mode.lower()
    ctx = click.get_current_context()

    def _source(name: str) -> Any:
        if ctx is None:
            return None
        return ctx.get_parameter_source(name)

    def _was_user_provided(name: str) -> bool:
        source = _source(name)
        if ParameterSource is None:
            return bool(source) and str(source) not in {"ParameterSource.DEFAULT", "ParameterSource.NONE"}
        none_sentinel = getattr(ParameterSource, "NONE", None)
        default_sources = {ParameterSource.DEFAULT}
        if none_sentinel is not None:
            default_sources.add(none_sentinel)
        return bool(source) and source not in default_sources

    try:
        if runtime_normalized == "modal":
            uvicorn_only_options = [
                ("host", "--host"),
                ("port", "--port"),
                ("reload_flag", "--reload/--no-reload"),
                ("force", "--force/--no-force"),
                ("trace_dir", "--trace"),
                ("trace_db", "--trace-db"),
            ]
            invalid = [label for param, label in uvicorn_only_options if _was_user_provided(param)]
            if invalid:
                raise click.ClickException(
                    f"{', '.join(invalid)} cannot be used with --runtime=modal."
                )

            if modal_mode_normalized == "serve" and _was_user_provided("dry_run"):
                raise click.ClickException("--dry-run is not supported with --modal-mode=serve.")

            command_choice: Literal["deploy", "serve"] = (
                "serve" if modal_mode_normalized == "serve" else "deploy"
            )
            run_modal_runtime(
                app_id,
                command=command_choice,
                modal_name=modal_name,
                dry_run=dry_run,
                modal_cli=modal_cli,
                env_file=env_file,
            )
            return

        modal_only_options = [
            ("modal_name", "--name"),
            ("dry_run", "--dry-run"),
            ("modal_cli", "--modal-cli"),
            ("modal_mode", "--modal-mode"),
        ]
        invalid = [label for param, label in modal_only_options if _was_user_provided(param)]
        if invalid:
            raise click.ClickException(
                f"{', '.join(invalid)} cannot be used with --runtime=uvicorn."
            )

        run_uvicorn_runtime(app_id, host, port, env_file, reload_flag, force, trace_dir, trace_db)
    except DeployCliError as exc:
        raise click.ClickException(_format_deploy_error(exc)) from exc
    except click.ClickException as err:
        converted = _translate_click_exception(err)
        if converted:
            raise click.ClickException(_format_deploy_error(converted)) from err
        raise


@click.command("modal-serve")
@click.argument("app_id", type=str, required=False)
@click.option("--modal-cli", default="modal", help="Path to modal CLI executable")
@click.option("--name", "modal_name", default=None, help="Override Modal app name (optional)")
@click.option(
    "--env-file",
    multiple=True,
    type=click.Path(),
    help="Env file to load into the container (can be repeated)",
)
def modal_serve_command(
    app_id: str | None, modal_cli: str, modal_name: str | None, env_file: Sequence[str]
) -> None:
    click.echo(f"[modal-serve] requested app_id={app_id or '(auto)'} modal_cli={modal_cli}")
    try:
        run_modal_runtime(
            app_id,
            command="serve",
            modal_name=modal_name,
            dry_run=False,
            modal_cli=modal_cli,
            env_file=env_file,
            use_demo_dir=False,
        )
    except DeployCliError as exc:
        raise click.ClickException(_format_deploy_error(exc)) from exc
    except click.ClickException as err:
        converted = _translate_click_exception(err)
        if converted:
            raise click.ClickException(_format_deploy_error(converted)) from err
        raise
    except SystemExit as exc:  # bubble up with context (legacy argparse would trigger this)
        raise click.ClickException(
            f"Legacy CLI intercepted modal-serve (exit {exc.code}). "
            "Make sure you're running the Click CLI (synth_ai.cli:cli)."
        ) from exc


command = deploy_command


def get_command() -> click.Command:
    return command


def register_task_app_commands(task_app_group: click.Group) -> None:
    task_app_group.add_command(command)
    task_app_group.add_command(modal_serve_command)
