import contextlib
import logging
import os
import shutil
import socket
import subprocess
import tempfile

import click

SQLD_VERSION = "v0.26.2"


def find_sqld_binary() -> str | None:
    """Locate an existing sqld binary on PATH or in common install locations."""

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
