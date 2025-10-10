#!/usr/bin/env python3
"""
Canonical CLI entrypoint for Synth AI (moved from synth_ai/cli.py).
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

import click

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        __pkg_version__ = _pkg_version("synth-ai")
    except PackageNotFoundError:
        try:
            from synth_ai import __version__ as __pkg_version__  # type: ignore
        except Exception:
            __pkg_version__ = "unknown"
except Exception:
    try:
        from synth_ai import __version__ as __pkg_version__  # type: ignore
    except Exception:
        __pkg_version__ = "unknown"


SQLD_VERSION = "v0.26.2"


def find_sqld_binary() -> str | None:
    sqld_path = shutil.which("sqld")
    if sqld_path:
        return sqld_path
    common_paths = [
        "/usr/local/bin/sqld",
        "/usr/bin/sqld",
        os.path.expanduser("~/.local/bin/sqld"),
        os.path.expanduser("~/bin/sqld"),
        os.path.expanduser("~/.turso/bin/sqld"),
    ]
    for path in common_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    return None


def install_sqld() -> str:
    """Install sqld via the Turso CLI, installing the CLI via Homebrew if needed."""

    click.echo("🔧 sqld not found. Attempting automatic install...")

    turso_cli_path = shutil.which("turso")
    brew_path = shutil.which("brew")

    if not turso_cli_path:
        if not brew_path:
            raise click.ClickException(
                "Automatic install requires either Homebrew or an existing Turso CLI.\n"
                "Install manually using one of:\n"
                "  • brew install tursodatabase/tap/turso\n"
                "  • curl -sSfL https://get.tur.so/install.sh | bash\n"
                "Then run 'turso dev' once and re-run this command."
            )

        click.echo("🧰 Installing Turso CLI via Homebrew (tursodatabase/tap/turso)…")
        try:
            subprocess.run(
                [brew_path, "install", "tursodatabase/tap/turso"],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise click.ClickException(
                "Homebrew install failed. Please resolve brew errors and retry."
            ) from exc

        turso_cli_path = shutil.which("turso")
        if not turso_cli_path:
            raise click.ClickException(
                "Homebrew reported success but the 'turso' binary is not on PATH."
            )

    click.echo("📥 Downloading sqld via 'turso dev' (this may take a few seconds)…")

    with tempfile.NamedTemporaryFile(prefix="synth_sqld_", suffix=".db", delete=False) as temp_db:
        temp_db_path = temp_db.name

    env = os.environ.copy()
    env.setdefault("TURSO_NONINTERACTIVE", "1")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    cmd = [
        turso_cli_path,
        "dev",
        f"--db-file={temp_db_path}",
        f"--port={port}",
    ]
    proc: subprocess.Popen[str] | None = None
    stdout_data = ""
    stderr_data = ""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            stdout_data, stderr_data = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                stdout_data, stderr_data = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout_data, stderr_data = proc.communicate()
    finally:
        if proc and proc.returncode not in (0, None) and (stdout_data or stderr_data):
            logging.getLogger(__name__).debug(
                "turso dev stdout: %s\nstderr: %s", stdout_data, stderr_data
            )
        with contextlib.suppress(OSError):
            os.unlink(temp_db_path)

    sqld_path = find_sqld_binary()
    if sqld_path:
        click.echo(f"✅ sqld available at {sqld_path}")
        return sqld_path

    raise click.ClickException(
        "sqld download did not succeed. Run 'turso dev' manually once, "
        "ensure it downloads sqld, and try again."
    )


@click.group(
    help=f"Synth AI v{__pkg_version__} - Software for aiding the best and multiplying the will."
)
@click.version_option(version=__pkg_version__, prog_name="synth-ai")
def cli():
    """Top-level command group for Synth AI."""


# === Legacy demo command group (aliases new rl_demo implementation) ===
@cli.group()
def demo():
    """Demo helpers (deploy, configure, run)."""


def _forward_to_demo(args: list[str]) -> None:
    # Lazy import to avoid loading demo deps unless needed
    try:
        from synth_ai.demos.core import cli as demo_cli  # type: ignore
    except Exception as e:  # pragma: no cover
        click.echo(f"Failed to import demo CLI: {e}")
        sys.exit(1)
    rc = int(demo_cli.main(args) or 0)  # type: ignore[attr-defined]
    if rc != 0:
        sys.exit(rc)


# (prepare command removed; handled by configure)


@demo.command()
@click.option("--local", is_flag=True, help="Run local FastAPI instead of Modal deploy")
@click.option(
    "--app", type=click.Path(), default=None, help="Path to Modal app.py for uv run modal deploy"
)
@click.option("--name", type=str, default="synth-math-demo", help="Modal app name")
@click.option(
    "--script", type=click.Path(), default=None, help="Path to deploy_task_app.sh (optional legacy)"
)
def deploy(local: bool, app: str | None, name: str, script: str | None):
    """Deploy the Math Task App (Modal by default)."""
    args: list[str] = ["rl_demo.deploy"]
    if local:
        args.append("--local")
    if app:
        args.extend(["--app", app])
    if name:
        args.extend(["--name", name])
    if script:
        args.extend(["--script", script])
    _forward_to_demo(args)


@demo.command()
def configure():
    """Print resolved environment and config path."""
    _forward_to_demo(["rl_demo.configure"])


@demo.command()
def setup():
    """Perform SDK handshake and write keys to .env."""
    _forward_to_demo(["rl_demo.setup"])


@demo.command()
@click.option("--template", type=str, default=None, help="Template id to instantiate")
@click.option("--dest", type=str, default=None, help="Destination directory for files")
@click.option("--force", is_flag=True, help="Overwrite existing files in destination")
def init(template: str | None, dest: str | None, force: bool):
    """Copy demo task app template into the current directory."""
    args: list[str] = ["demo.init"]
    if template:
        args.extend(["--template", template])
    if dest:
        args.extend(["--dest", dest])
    if force:
        args.append("--force")
    _forward_to_demo(args)


@demo.command()
@click.option("--batch-size", type=int, default=None)
@click.option("--group-size", type=int, default=None)
@click.option("--model", type=str, default=None)
@click.option("--timeout", type=int, default=600)
def run(batch_size: int | None, group_size: int | None, model: str | None, timeout: int):
    """Kick off a short RL job using the prepared TOML."""
    args = ["rl_demo.run"]
    if batch_size is not None:
        args.extend(["--batch-size", str(batch_size)])
    if group_size is not None:
        args.extend(["--group-size", str(group_size)])
    if model:
        args.extend(["--model", model])
    if timeout:
        args.extend(["--timeout", str(timeout)])
    _forward_to_demo(args)


@cli.command(name="setup")
def setup_command():
    """Perform SDK handshake and write keys to .env."""
    _forward_to_demo(["rl_demo.setup"])


@cli.command()
@click.option("--db-file", default="traces/v3/synth_ai.db", help="Database file path")
@click.option("--sqld-port", default=8080, type=int, help="Port for sqld HTTP interface")
@click.option("--env-port", default=8901, type=int, help="Port for environment service")
@click.option("--no-sqld", is_flag=True, help="Skip starting sqld daemon")
@click.option("--no-env", is_flag=True, help="Skip starting environment service")
@click.option(
    "--reload/--no-reload",
    default=False,
    help="Enable auto-reload (default: off). Or set SYNTH_RELOAD=1",
)
@click.option(
    "--force/--no-force",
    default=True,
    help="Kill any process already bound to --env-port without prompting",
)
def serve_deprecated(
    db_file: str,
    sqld_port: int,
    env_port: int,
    no_sqld: bool,
    no_env: bool,
    reload: bool,
    force: bool,
):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    click.echo(
        "⚠️  'synth-ai serve' now targets task apps; use 'synth-ai serve' for task apps or 'synth-ai serve-deprecated' for this legacy service.",
        err=True,
    )
    processes = []

    def signal_handler(sig, frame):
        click.echo("\n🛑 Shutting down services...")
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if not no_sqld:
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"sqld.*--http-listen-addr.*:{sqld_port}"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                sqld_bin = find_sqld_binary() or install_sqld()
                click.echo(f"🗄️  Starting sqld (local only) on port {sqld_port}")
                proc = subprocess.Popen(
                    [
                        sqld_bin,
                        "--db-path",
                        db_file,
                        "--http-listen-addr",
                        f"127.0.0.1:{sqld_port}",
                    ],
                    stdout=open("sqld.log", "w"),  # noqa: SIM115
                    stderr=subprocess.STDOUT,
                )
                processes.append(proc)
                time.sleep(2)
        except FileNotFoundError:
            pass

    if not no_env:
        click.echo("")
        click.echo(f"🚀 Starting Synth-AI Environment Service on port {env_port}")
        click.echo("")

        # Ensure port is free
        try:
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                in_use = s.connect_ex(("127.0.0.1", env_port)) == 0
        except Exception:
            in_use = False
        if in_use:
            pids: list[str] = []
            try:
                out = subprocess.run(
                    ["lsof", "-ti", f":{env_port}"], capture_output=True, text=True
                )
                if out.returncode == 0 and out.stdout.strip():
                    pids = [p for p in out.stdout.strip().splitlines() if p]
            except FileNotFoundError:
                pids = []
            if force:
                if pids:
                    subprocess.run(["kill", "-9", *pids], check=False)
                    time.sleep(0.5)
            else:
                suffix = f" PIDs: {', '.join(pids)}" if pids else ""
                if click.confirm(
                    f"⚠️  Port {env_port} is in use.{suffix} Kill and continue?", default=True
                ):
                    if pids:
                        subprocess.run(["kill", "-9", *pids], check=False)
                        time.sleep(0.5)
                else:
                    click.echo("❌ Aborting.")
                    sys.exit(1)

        env = os.environ.copy()
        env["SYNTH_LOGGING"] = "true"
        click.echo("📦 Environment:")
        click.echo(f"   Python: {sys.executable}")
        click.echo(f"   Working directory: {os.getcwd()}")
        click.echo("")
        click.echo("🔄 Starting services...")
        click.echo(f"   - sqld daemon: http://127.0.0.1:{sqld_port}")
        click.echo(f"   - Environment service: http://127.0.0.1:{env_port}")
        click.echo("")
        click.echo("💡 Tips:")
        click.echo("   - Check sqld.log if database issues occur")
        click.echo("   - Use Ctrl+C to stop all services")
        reload_enabled = reload or (os.getenv("SYNTH_RELOAD", "0") == "1")
        click.echo(
            "   - Auto-reload ENABLED (code changes restart service)"
            if reload_enabled
            else "   - Auto-reload DISABLED (stable in-memory sessions)"
        )
        click.echo("")

        uvicorn_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "synth_ai.environments.service.app:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(env_port),
            "--log-level",
            "info",
        ]
        if reload_enabled:
            uvicorn_cmd.append("--reload")
            if os.path.exists("synth_ai"):
                uvicorn_cmd.extend(["--reload-dir", "synth_ai"])
        proc = subprocess.Popen(uvicorn_cmd, env=env)
        processes.append(proc)

    if processes:
        click.echo("\n✨ All services started! Press Ctrl+C to stop.")
        try:
            for proc in processes:
                proc.wait()
        except KeyboardInterrupt:
            pass
    else:
        click.echo("No services to start.")
