#!/usr/bin/env python3
"""
Synth AI CLI - Command line interface for Synth AI services.
"""

import sys
import os
import subprocess
import signal
import time
import shutil
from pathlib import Path
from typing import Optional
import logging

import click

logger = logging.getLogger(__name__)


def find_sqld_binary() -> Optional[str]:
    """Find the sqld binary in common locations."""
    # Check if sqld is in PATH
    sqld_path = shutil.which("sqld")
    if sqld_path:
        return sqld_path

    # Check common installation locations
    common_paths = [
        "/usr/local/bin/sqld",
        "/usr/bin/sqld",
        os.path.expanduser("~/.local/bin/sqld"),
        os.path.expanduser("~/bin/sqld"),
        # Package-specific location
        os.path.join(os.path.dirname(__file__), "bin", "sqld"),
    ]

    for path in common_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path

    return None


def install_sqld():
    """Install sqld binary if not found."""
    sqld_bin = find_sqld_binary()
    if sqld_bin:
        return sqld_bin

    click.echo("🔧 sqld not found. Installing...")

    # Try to use the bundled install script first
    package_install_script = os.path.join(os.path.dirname(__file__), "install_sqld.sh")

    try:
        if os.path.exists(package_install_script):
            # Use bundled script
            subprocess.run(["bash", package_install_script], check=True)
        else:
            # Create install script inline
            install_script = """#!/bin/bash
set -e

SQLD_VERSION="v0.26.2"
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

# Map architecture names
case "$ARCH" in
    x86_64) ARCH="x86_64" ;;
    aarch64|arm64) ARCH="aarch64" ;;
    *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

# Construct download URL
URL="https://github.com/tursodatabase/libsql/releases/download/libsql-server-${SQLD_VERSION}/sqld-${OS}-${ARCH}.tar.xz"

echo "📥 Downloading sqld ${SQLD_VERSION} for ${OS}-${ARCH}..."

# Download and extract
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"
curl -L -o sqld.tar.xz "$URL"
tar -xf sqld.tar.xz

# Install to user's local bin
mkdir -p ~/.local/bin
mv sqld ~/.local/bin/
chmod +x ~/.local/bin/sqld

# Clean up
cd -
rm -rf "$TMP_DIR"

echo "✅ sqld installed to ~/.local/bin/sqld"
"""

            # Write and execute install script
            with open("/tmp/install_sqld.sh", "w") as f:
                f.write(install_script)

            subprocess.run(["bash", "/tmp/install_sqld.sh"], check=True)
            os.unlink("/tmp/install_sqld.sh")

        # Add ~/.local/bin to PATH if needed
        local_bin = os.path.expanduser("~/.local/bin")
        if local_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{local_bin}:{os.environ.get('PATH', '')}"

        return os.path.expanduser("~/.local/bin/sqld")

    except Exception as e:
        click.echo(f"❌ Failed to install sqld: {e}", err=True)
        click.echo(
            "Please install sqld manually from: https://github.com/tursodatabase/libsql", err=True
        )
        sys.exit(1)


@click.group()
def cli():
    """Synth AI - Software for aiding the best and multiplying the will."""
    pass


@cli.group("env")
def env():
    """Environment management commands."""
    pass


@env.command("register")
@click.option("--name", required=True, help="Environment name (e.g., 'MyEnv-v1')")
@click.option("--module", "module_path", required=True, help="Python module path (e.g., 'my_package.env')")
@click.option("--class-name", "class_name", required=True, help="Environment class name")
@click.option("--description", help="Optional description")
@click.option("--service-url", default="http://localhost:8901", help="Environment service URL")
def register_env(name: str, module_path: str, class_name: str, description: Optional[str], service_url: str):
    """Register a new environment with the service."""
    import requests
    
    payload = {
        "name": name,
        "module_path": module_path,
        "class_name": class_name,
    }
    if description:
        payload["description"] = description
    
    try:
        response = requests.post(f"{service_url}/registry/environments", json=payload, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        click.echo(f"✅ {result['message']}")
        
    except requests.exceptions.ConnectionError:
        click.echo(f"❌ Could not connect to environment service at {service_url}")
        click.echo("💡 Make sure the service is running: synth-ai serve")
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        click.echo(f"❌ Registration failed: {error_detail}")
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@env.command("list")
@click.option("--service-url", default="http://localhost:8901", help="Environment service URL")
def list_envs(service_url: str):
    """List all registered environments."""
    import requests
    
    try:
        response = requests.get(f"{service_url}/registry/environments", timeout=10)
        response.raise_for_status()
        
        result = response.json()
        environments = result["environments"]
        
        if not environments:
            click.echo("No environments registered.")
            return
        
        click.echo(f"\n📦 Registered Environments ({result['total_count']}):")
        click.echo("=" * 60)
        
        for env in environments:
            click.echo(f"🌍 {env['name']}")
            click.echo(f"   Class: {env['class_name']}")
            click.echo(f"   Module: {env['module']}")
            if env['description']:
                click.echo(f"   Description: {env['description']}")
            click.echo()
            
    except requests.exceptions.ConnectionError:
        click.echo(f"❌ Could not connect to environment service at {service_url}")
        click.echo("💡 Make sure the service is running: synth-ai serve")
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@env.command("unregister")
@click.option("--name", required=True, help="Environment name to unregister")
@click.option("--service-url", default="http://localhost:8901", help="Environment service URL")
def unregister_env(name: str, service_url: str):
    """Unregister an environment from the service."""
    import requests
    
    try:
        response = requests.delete(f"{service_url}/registry/environments/{name}", timeout=10)
        response.raise_for_status()
        
        result = response.json()
        click.echo(f"✅ {result['message']}")
        
    except requests.exceptions.ConnectionError:
        click.echo(f"❌ Could not connect to environment service at {service_url}")
        click.echo("💡 Make sure the service is running: synth-ai serve")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            click.echo(f"❌ Environment '{name}' not found in registry")
        else:
            error_detail = e.response.json().get("detail", str(e)) if e.response else str(e)
            click.echo(f"❌ Unregistration failed: {error_detail}")
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.option("--url", default="sqlite+aiosqlite:///./synth_ai.db/dbs/default/data", help="Database URL")
def view(url: str):
    """Launch the interactive TUI dashboard."""
    try:
        from .tui.dashboard import SynthDashboard
        app = SynthDashboard(db_url=url)
        app.run()
    except ImportError:
        click.echo("❌ Textual not installed. Install with: pip install textual", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\n👋 Dashboard closed", err=True)

# Note: subcommands (watch, experiments, experiment, usage, traces, status, recent, calc)
# are registered from the package module synth_ai.cli at import time.


@cli.command()
@click.option("--db-file", default="synth_ai.db", help="Database file path")
@click.option("--sqld-port", default=8080, type=int, help="Port for sqld HTTP interface")
@click.option("--env-port", default=8901, type=int, help="Port for environment service")
@click.option("--no-sqld", is_flag=True, help="Skip starting sqld daemon")
@click.option("--no-env", is_flag=True, help="Skip starting environment service")
@click.option("--reload/--no-reload", default=False, help="Enable auto-reload (default: off). Or set SYNTH_RELOAD=1")
@click.option("--force", is_flag=True, help="Kill any process already bound to --env-port without prompting")
def serve(db_file: str, sqld_port: int, env_port: int, no_sqld: bool, no_env: bool, reload: bool, force: bool):
    """Start Synth AI services (sqld daemon and environment service)."""

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    processes = []

    def signal_handler(sig, frame):
        """Handle shutdown gracefully."""
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

    # Start sqld if requested
    if not no_sqld:
        # Check if sqld is already running
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"sqld.*--http-listen-addr.*:{sqld_port}"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                click.echo(f"✅ sqld already running on port {sqld_port}")
                click.echo(f"   Database: {db_file}")
                click.echo(f"   HTTP API: http://127.0.0.1:{sqld_port}")
            else:
                # Find or install sqld
                sqld_bin = find_sqld_binary()
                if not sqld_bin:
                    sqld_bin = install_sqld()

                click.echo(f"🗄️  Starting sqld (local only) on port {sqld_port}")

                # Start sqld
                sqld_cmd = [
                    sqld_bin,
                    "--db-path",
                    db_file,
                    "--http-listen-addr",
                    f"127.0.0.1:{sqld_port}",
                ]

                # Create log file
                sqld_log = open("sqld.log", "w")
                proc = subprocess.Popen(sqld_cmd, stdout=sqld_log, stderr=subprocess.STDOUT)
                processes.append(proc)

                # Wait for sqld to start
                time.sleep(2)

                # Verify it started
                if proc.poll() is not None:
                    click.echo("❌ Failed to start sqld. Check sqld.log for details.", err=True)
                    with open("sqld.log", "r") as f:
                        click.echo("\nLast 10 lines of sqld.log:")
                        lines = f.readlines()
                        for line in lines[-10:]:
                            click.echo(f"  {line.rstrip()}")
                    sys.exit(1)

                click.echo("✅ sqld started successfully!")
                click.echo(f"   Database: {db_file}")
                click.echo(f"   HTTP API: http://127.0.0.1:{sqld_port}")
                click.echo(f"   Log file: {os.path.abspath('sqld.log')}")

        except FileNotFoundError:
            click.echo("⚠️  pgrep not found, assuming sqld is not running")

    # Start environment service if requested
    if not no_env:
        click.echo("")
        click.echo(f"🚀 Starting Synth-AI Environment Service on port {env_port}")
        click.echo("")

        # Check if we're in a valid project directory
        if not os.path.exists("synth_ai") and not os.path.exists(
            os.path.join(os.path.dirname(__file__), "environments")
        ):
            click.echo("⚠️  Running from installed package, using package's environment service")
            # Use the installed module path
            env_module = "synth_ai.environments.service.app:app"
        else:
            # Running from source
            env_module = "synth_ai.environments.service.app:app"

        # Ensure env_port is free; offer to kill existing listeners
        try:
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                in_use = s.connect_ex(("127.0.0.1", env_port)) == 0
        except Exception:
            in_use = False

        if in_use:
            # Try to find PIDs using lsof (macOS/Linux)
            pids: list[str] = []
            try:
                out = subprocess.run(["lsof", "-ti", f":{env_port}"], capture_output=True, text=True)
                if out.returncode == 0 and out.stdout.strip():
                    pids = [p for p in out.stdout.strip().splitlines() if p]
            except FileNotFoundError:
                pids = []

            if force:
                if pids:
                    subprocess.run(["kill", "-9", *pids], check=False)
                    time.sleep(0.5)
            else:
                pid_info = f" PIDs: {', '.join(pids)}" if pids else ""
                if click.confirm(f"⚠️  Port {env_port} is in use.{pid_info} Kill and continue?", default=True):
                    if pids:
                        subprocess.run(["kill", "-9", *pids], check=False)
                        time.sleep(0.5)
                else:
                    click.echo("❌ Aborting. Re-run with --force to auto-kill or choose a different --env-port.")
                    sys.exit(1)

        # Set environment variables
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
        # Determine reload behavior: CLI flag overrides env var, default is off
        reload_enabled = reload or (os.getenv("SYNTH_RELOAD", "0") == "1")
        if reload_enabled:
            click.echo("   - Auto-reload ENABLED (code changes restart service)")
        else:
            click.echo("   - Auto-reload DISABLED (stable in-memory sessions)")
        click.echo("")

        # Start uvicorn
        uvicorn_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            env_module,
            "--host",
            "0.0.0.0",
            "--port",
            str(env_port),
            "--log-level",
            "info",
        ]

        if reload_enabled:
            uvicorn_cmd.append("--reload")
            # If running from source, add reload directory
            if os.path.exists("synth_ai"):
                uvicorn_cmd.extend(["--reload-dir", "synth_ai"])

        proc = subprocess.Popen(uvicorn_cmd, env=env)
        processes.append(proc)

    # Wait for processes
    if processes:
        click.echo("\n✨ All services started! Press Ctrl+C to stop.")
        try:
            for proc in processes:
                proc.wait()
        except KeyboardInterrupt:
            pass
    else:
        click.echo("No services to start.")


if __name__ == "__main__":
    cli()
