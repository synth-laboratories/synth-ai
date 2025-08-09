#!/usr/bin/env python3
"""
Canonical CLI entrypoint for Synth AI (moved from synth_ai/cli.py).
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Optional

import click


def find_sqld_binary() -> Optional[str]:
    sqld_path = shutil.which("sqld")
    if sqld_path:
        return sqld_path
    common_paths = [
        "/usr/local/bin/sqld",
        "/usr/bin/sqld",
        os.path.expanduser("~/.local/bin/sqld"),
        os.path.expanduser("~/bin/sqld"),
    ]
    for path in common_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    return None


def install_sqld() -> str:
    click.echo("🔧 sqld not found. Installing...")
    script = """#!/bin/bash
set -e
SQLD_VERSION="v0.26.2"
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
case "$ARCH" in
  x86_64) ARCH="x86_64" ;;
  aarch64|arm64) ARCH="aarch64" ;;
  *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac
URL="https://github.com/tursodatabase/libsql/releases/download/libsql-server-${SQLD_VERSION}/sqld-${OS}-${ARCH}.tar.xz"
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"
curl -L -o sqld.tar.xz "$URL"
tar -xf sqld.tar.xz
mkdir -p ~/.local/bin
mv sqld ~/.local/bin/
chmod +x ~/.local/bin/sqld
cd -
rm -rf "$TMP_DIR"
"""
    path = "/tmp/install_sqld.sh"
    with open(path, "w") as f:
        f.write(script)
    subprocess.run(["bash", path], check=True)
    os.unlink(path)
    local_bin = os.path.expanduser("~/.local/bin")
    if local_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{local_bin}:{os.environ.get('PATH', '')}"
    return os.path.expanduser("~/.local/bin/sqld")


@click.group()
def cli():
    """Synth AI - Software for aiding the best and multiplying the will."""


@cli.command()
@click.option("--db-file", default="traces/v3/synth_ai.db", help="Database file path")
@click.option("--sqld-port", default=8080, type=int, help="Port for sqld HTTP interface")
@click.option("--env-port", default=8901, type=int, help="Port for environment service")
@click.option("--no-sqld", is_flag=True, help="Skip starting sqld daemon")
@click.option("--no-env", is_flag=True, help="Skip starting environment service")
@click.option("--reload/--no-reload", default=False, help="Enable auto-reload (default: off). Or set SYNTH_RELOAD=1")
@click.option("--force/--no-force", default=True, help="Kill any process already bound to --env-port without prompting")
def serve(db_file: str, sqld_port: int, env_port: int, no_sqld: bool, no_env: bool, reload: bool, force: bool):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
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
            result = subprocess.run(["pgrep", "-f", f"sqld.*--http-listen-addr.*:{sqld_port}"], capture_output=True, text=True)
            if result.returncode != 0:
                sqld_bin = find_sqld_binary() or install_sqld()
                click.echo(f"🗄️  Starting sqld (local only) on port {sqld_port}")
                proc = subprocess.Popen([sqld_bin, "--db-path", db_file, "--http-listen-addr", f"127.0.0.1:{sqld_port}"], stdout=open("sqld.log", "w"), stderr=subprocess.STDOUT)
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
                suffix = f" PIDs: {', '.join(pids)}" if pids else ""
                if click.confirm(f"⚠️  Port {env_port} is in use.{suffix} Kill and continue?", default=True):
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
        click.echo("   - Auto-reload ENABLED (code changes restart service)" if reload_enabled else "   - Auto-reload DISABLED (stable in-memory sessions)")
        click.echo("")

        uvicorn_cmd = [
            sys.executable, "-m", "uvicorn", "synth_ai.environments.service.app:app",
            "--host", "0.0.0.0", "--port", str(env_port), "--log-level", "info",
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

