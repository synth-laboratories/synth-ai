"""Cloudflare CLI/bootstrap helpers and tunnel deployment utilities."""

import asyncio
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import click
import httpx
import requests
from synth_ai.cfgs import CloudflareTunnelDeployCfg
from synth_ai.urls import BACKEND_URL_BASE
from synth_ai.utils import log_error, log_event
from synth_ai.utils.apps import get_asgi_app, load_file_to_module
from synth_ai.utils.env import resolve_env_var, write_env_var_to_dotenv
from synth_ai.utils.paths import (
    REPO_ROOT,
    configure_import_paths,
)
from uvicorn._types import ASGIApplication

import uvicorn

# Constants
CLOUDFLARED_BIN_NAME = "cloudflared"
CLOUDFLARED_RELEASES = "https://updatecloudflared.com/launcher"
CLOUDFLARE_DOCS_URL = "https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation"

# Regex for parsing quick tunnel URLs
_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.I)

# Global state - store tunnel process handles for cleanup
_TUNNEL_PROCESSES: dict[int, subprocess.Popen] = {}


# ---------------------------------------------------------------------------
# Cloudflared binary management
# ---------------------------------------------------------------------------


def get_cloudflared_path(prefer_system: bool = True) -> Optional[Path]:
    """Locate the cloudflared binary (managed bin dir, PATH, or common dirs)."""
    bin_dir = Path.home() / ".synth-ai" / "bin"
    candidate = bin_dir / CLOUDFLARED_BIN_NAME
    if candidate.exists() and os.access(candidate, os.X_OK):
        return candidate

    if prefer_system:
        resolved = shutil.which(CLOUDFLARED_BIN_NAME)
        if resolved:
            return Path(resolved)

    common = [
        Path("/usr/local/bin/cloudflared"),
        Path("/opt/homebrew/bin/cloudflared"),
        Path.home() / "bin" / "cloudflared",
    ]
    for path in common:
        if path.exists() and os.access(path, os.X_OK):
            return path
    return None


def ensure_cloudflared_installed(force: bool = False) -> Path:
    """Ensure cloudflared is installed in synth-ai's managed bin directory."""
    existing = get_cloudflared_path(prefer_system=not force)
    if existing and not force:
        return existing

    target_dir = Path.home() / ".synth-ai" / "bin"
    target_dir.mkdir(parents=True, exist_ok=True)

    url = _resolve_cloudflared_download_url()
    tmp_file = _download_file(url)

    if tmp_file.suffixes[-2:] == [".tar", ".gz"]:
        _extract_tarball(tmp_file, target_dir)
    elif tmp_file.suffix == ".gz":
        _extract_gzip(tmp_file, target_dir / CLOUDFLARED_BIN_NAME)
    else:
        shutil.move(str(tmp_file), str(target_dir / CLOUDFLARED_BIN_NAME))

    bin_path = target_dir / CLOUDFLARED_BIN_NAME
    bin_path.chmod(0o755)
    log_event("info", "cloudflared installed", ctx={"path": str(bin_path)})
    return bin_path


def require_cloudflared() -> Path:
    """Return cloudflared binary or raise ClickException with guidance."""
    path = get_cloudflared_path()
    if path:
        return path

    extra = ""
    if platform.system() == "Darwin":
        extra = "Try `brew install cloudflare/cloudflare/cloudflared`."
    elif platform.system() == "Linux":
        extra = "See Cloudflare docs for Linux packages."
    log_error("cloudflared not found", ctx={"hint": extra})
    raise click.ClickException(
        f"Cloudflared CLI missing. Install via Homebrew or follow {CLOUDFLARE_DOCS_URL}."
    )


def run_cloudflared_cmd(args: list[str], *, env: Optional[dict[str, str]] = None) -> subprocess.Popen:
    """Spawn cloudflared subprocess (mirrors synth_ai.modal.run_modal_cmd)."""
    bin_path = require_cloudflared()
    cmd = [str(bin_path), *args]
    log_event("info", "starting cloudflared", ctx={"cmd": cmd})
    try:
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env or os.environ.copy(),
        )
    except FileNotFoundError as exc:
        raise click.ClickException(f"cloudflared binary missing: {exc}") from exc
    except Exception as exc:
        raise click.ClickException(f"Failed to start cloudflared: {exc}") from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_cloudflared_download_url() -> str:
    system = platform.system().lower()
    arch = platform.machine().lower()
    mapping = {"darwin": "macos", "linux": "linux", "windows": "windows"}
    platform_key = mapping.get(system)
    if not platform_key:
        raise RuntimeError(f"Unsupported platform: {system}")

    arch_key = "amd64"
    if arch in ("arm64", "aarch64"):
        arch_key = "arm64"

    resp = requests.get(f"{CLOUDFLARED_RELEASES}/v1/{platform_key}/{arch_key}/versions/stable", timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    url = data.get("url")
    if not url:
        raise RuntimeError("Cloudflared release metadata missing URL")
    return url


def _download_file(url: str) -> Path:
    resp = requests.get(url, timeout=60.0, stream=True)
    resp.raise_for_status()
    suffix = Path(url.split("?")[0]).suffix or ".tmp"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
    return Path(tmp_path)


def _extract_tarball(archive_path: Path, target_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(target_dir)
    archive_path.unlink(missing_ok=True)


def _extract_gzip(gz_path: Path, target: Path) -> None:
    import gzip

    # gzip.open ensures the bytes are decompressed while copying to target
    with gzip.open(gz_path, "rb") as gz_fh, open(target, "wb") as out_fh:
        shutil.copyfileobj(gz_fh, out_fh)
    gz_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tunnel process management
# ---------------------------------------------------------------------------


def open_quick_tunnel(port: int, wait_s: float = 10.0) -> Tuple[str, subprocess.Popen]:
    """
    Open a quick (ephemeral) Cloudflare tunnel.

    Args:
        port: Local port to tunnel to
        wait_s: Maximum time to wait for URL in seconds

    Returns:
        Tuple of (public_url, process_handle)

    Raises:
        RuntimeError: If tunnel fails to start or URL cannot be parsed
    """
    bin_path = require_cloudflared()
    proc = subprocess.Popen(
        [str(bin_path), "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    start = time.time()
    url: Optional[str] = None

    # Stream stdout to detect the trycloudflare URL
    while time.time() - start < wait_s:
        if proc.poll() is not None:
            # Process exited early
            stdout, _ = proc.communicate()
            raise RuntimeError(
                f"cloudflared exited early with code {proc.returncode}. "
                f"Output: {stdout[:500] if stdout else 'no output'}"
            )

        if proc.stdout is None:
            raise RuntimeError("cloudflared process has no stdout")

        line = proc.stdout.readline()
        if not line:
            time.sleep(0.05)
            continue

        match = _URL_RE.search(line)
        if match:
            url = match.group(0)
            break

    if not url:
        proc.terminate()
        stdout, _ = proc.communicate(timeout=2.0)
        raise RuntimeError(
            f"Failed to parse trycloudflare URL from cloudflared output after {wait_s}s. "
            f"Output: {stdout[:500] if stdout else 'no output'}"
        )

    return url, proc


def open_managed_tunnel(tunnel_token: str) -> subprocess.Popen:
    """
    Open a managed (named) Cloudflare tunnel using a token.

    Args:
        tunnel_token: Cloudflare tunnel token from backend API

    Returns:
        Process handle for the tunnel

    Raises:
        RuntimeError: If cloudflared is not installed
    """
    bin_path = require_cloudflared()
    # cloudflared v2023.4+ accepts --token for named tunnels
    return subprocess.Popen(
        [str(bin_path), "tunnel", "run", "--token", tunnel_token],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def stop_tunnel(proc: Optional[subprocess.Popen]) -> None:
    """
    Gracefully stop a tunnel process.

    Args:
        proc: Process handle to terminate, or None
    """
    if proc is None:
        return

    if proc.poll() is None:
        # Process is still running
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            # Force kill if graceful termination fails
            proc.kill()
            proc.wait()


def store_tunnel_credentials(
    tunnel_url: str,
    access_client_id: Optional[str] = None,
    access_client_secret: Optional[str] = None,
    env_file: Optional[Path] = None,
) -> None:
    """
    Store tunnel credentials in .env file for optimizer to use.

    Writes:
    - TASK_APP_URL=<tunnel_url>
    - CF_ACCESS_CLIENT_ID=<client_id> (if Access enabled)
    - CF_ACCESS_CLIENT_SECRET=<client_secret> (if Access enabled)

    Args:
        tunnel_url: Public tunnel URL (e.g., "https://cust-abc123.usesynth.ai")
        access_client_id: Cloudflare Access client ID (optional)
        access_client_secret: Cloudflare Access client secret (optional)
        env_file: Path to .env file (defaults to .env in current directory)
    """
    write_env_var_to_dotenv(
        "TASK_APP_URL",
        tunnel_url,
        output_file_path=env_file,
        print_msg=True,
        mask_msg=False,
    )

    if access_client_id:
        write_env_var_to_dotenv(
            "CF_ACCESS_CLIENT_ID",
            access_client_id,
            output_file_path=env_file,
            print_msg=True,
            mask_msg=True,
        )

    if access_client_secret:
        write_env_var_to_dotenv(
            "CF_ACCESS_CLIENT_SECRET",
            access_client_secret,
            output_file_path=env_file,
            print_msg=True,
            mask_msg=True,
        )


# ---------------------------------------------------------------------------
# Tunnel deployment helpers
# ---------------------------------------------------------------------------


async def create_tunnel(
    synth_api_key: str,
    port: int,
    subdomain: Optional[str] = None,
) -> dict[str, Any]:
    """
    Create a managed Cloudflare tunnel via Synth backend API.

    Args:
        synth_api_key: Synth API key for authentication
        port: Local port the tunnel will forward to
        subdomain: Optional custom subdomain (e.g., "my-company")

    Returns:
        Dict containing:
        - tunnel_token: Token for cloudflared
        - hostname: Public hostname (e.g., "cust-abc123.usesynth.ai")
        - access_client_id: Cloudflare Access client ID (if Access enabled)
        - access_client_secret: Cloudflare Access client secret (if Access enabled)

    Raises:
        RuntimeError: If API request fails
    """
    url = f"{BACKEND_URL_BASE}/api/v1/tunnels/"

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {synth_api_key}"},
                json={
                    "subdomain": subdomain or f"tunnel-{port}",
                    "local_port": port,
                    "local_host": "127.0.0.1",
                },
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Backend API returned {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Failed to connect to backend at {url}: {exc}"
        ) from exc


async def wait_for_health_check(
    host: str,
    port: int,
    api_key: str,
    timeout: float = 30.0,
) -> None:
    """
    Wait for task app health endpoint to be ready.

    Args:
        host: Host to check
        port: Port to check
        api_key: API key for authentication
        timeout: Maximum time to wait in seconds

    Raises:
        RuntimeError: If health check fails or times out
    """
    health_url = f"http://{host}:{port}/health"
    headers = {"X-API-Key": api_key}
    start = time.time()

    while time.time() - start < timeout:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(health_url, headers=headers)
                # Accept both 200 (success) and 400 (auth error means server is up)
                if response.status_code in (200, 400):
                    return
        except (httpx.RequestError, httpx.TimeoutException):
            pass

        await asyncio.sleep(0.5)

    raise RuntimeError(
        f"Health check failed: {health_url} not ready after {timeout}s. "
        "Make sure your task app has a /health endpoint."
    )


def start_uvicorn_background(
    app: ASGIApplication,
    host: str,
    port: int,
    daemon: bool = True,
) -> None:
    """
    Start uvicorn server in a background thread.

    Args:
        app: ASGI application
        host: Host to bind to
        port: Port to bind to
        daemon: If True, thread dies when main process exits. If False, thread keeps running.
    """
    import threading

    def serve():
        try:
            uvicorn.run(
                app,
                host=host,
                port=port,
                reload=False,
                log_level="info",
            )
        except Exception as exc:
            # Log error but don't raise (background thread)
            print(f"Uvicorn error: {exc}", flush=True)

    thread = threading.Thread(
        target=serve,
        name=f"synth-uvicorn-tunnel-{port}",
        daemon=daemon,
    )
    thread.start()


async def deploy_app_tunnel(
    cfg: CloudflareTunnelDeployCfg,
    env_file: Optional[Path] = None,
    keep_alive: bool = False,
    wait: bool = False,
    health_check_timeout: float = 30.0,
) -> str:
    """
    Deploy task app via Cloudflare Tunnel.

    This function provides a clean abstraction that handles:
    1. Starting the local task app (uvicorn) in background
    2. Optionally waiting for health check (only if wait=True)
    3. Opening tunnel (quick or managed)
    4. Writing tunnel URL and Access credentials to .env
    5. Optionally keeping processes alive (blocking vs non-blocking mode)

    By default (wait=False), this function is non-blocking and returns immediately
    after starting the tunnel. This is designed for AI agent use to prevent indefinite stalls.
    Processes run in the background and will continue until explicitly stopped.

    When `wait=True` or `keep_alive=True`, this function blocks and keeps the tunnel running
    until interrupted (Ctrl+C). Use this for interactive use or when you need to wait
    for the deployment to complete.

    Args:
        cfg: Tunnel deployment configuration
        env_file: Optional path to .env file (defaults to .env in current directory)
        keep_alive: (Deprecated) If True, block and keep tunnel alive until interrupted.
                   Use `wait` instead.
        wait: If True, wait for health check and block until interrupted.
             If False (default), return immediately after deployment (background mode).
        health_check_timeout: Maximum time to wait for health check (only used if wait=True)

    Returns:
        Public tunnel URL

    Raises:
        RuntimeError: If deployment fails at any step

    Example:
        # Non-blocking (background mode, returns immediately) - DEFAULT
        url = await deploy_app_tunnel(cfg, wait=False)

        # Blocking (waits for health check and keeps tunnel alive)
        url = await deploy_app_tunnel(cfg, wait=True)
    """

    ensure_cloudflared_installed()

    os.environ["ENVIRONMENT_API_KEY"] = cfg.env_api_key
    if cfg.trace:
        os.environ["TASKAPP_TRACING_ENABLED"] = "1"
    else:
        os.environ.pop("TASKAPP_TRACING_ENABLED", None)

    configure_import_paths(cfg.task_app_path, REPO_ROOT)
    module = load_file_to_module(cfg.task_app_path, f"_synth_tunnel_task_app_{cfg.task_app_path.stem}")
    app = get_asgi_app(module)

    # Always use non-daemon thread so it survives when main process exits
    start_uvicorn_background(app, cfg.host, cfg.port, daemon=False)
    
    # Only wait for health check if wait mode is enabled (for AI agents, skip to avoid stalls)
    if wait or keep_alive:
        await wait_for_health_check(cfg.host, cfg.port, cfg.env_api_key, timeout=health_check_timeout)
    else:
        # In background mode, give it a short moment to start, but don't wait for full health check
        # This prevents indefinite stalls while still allowing the server to start
        import asyncio
        await asyncio.sleep(1.0)  # Brief delay to let server start

    tunnel_proc: Optional[subprocess.Popen] = None
    try:
        if cfg.mode == "quick":
            # Quick tunnel: ephemeral, no backend API call
            url, tunnel_proc = open_quick_tunnel(cfg.port)
            _TUNNEL_PROCESSES[cfg.port] = tunnel_proc
            store_tunnel_credentials(url, None, None, env_file)
            # Record tunnel for scan command
            try:
                from synth_ai.utils.tunnel_records import record_tunnel
                record_tunnel(
                    url=url,
                    port=cfg.port,
                    mode="quick",
                    pid=tunnel_proc.pid if tunnel_proc else None,
                    hostname=url.replace("https://", "").split("/")[0] if url.startswith("https://") else None,
                    local_host=cfg.host,
                    task_app_path=str(cfg.task_app_path) if cfg.task_app_path else None,
                )
            except Exception:
                pass  # Fail silently - records are optional
        else:
            # Managed tunnel: provision via backend API
            synth_api_key = resolve_env_var("SYNTH_API_KEY")
            data = await create_tunnel(synth_api_key, cfg.port, cfg.subdomain)

            tunnel_token = data["tunnel_token"]
            hostname = data["hostname"]
            access_client_id = data.get("access_client_id")
            access_client_secret = data.get("access_client_secret")

            tunnel_proc = open_managed_tunnel(tunnel_token)
            _TUNNEL_PROCESSES[cfg.port] = tunnel_proc

            url = f"https://{hostname}"
            store_tunnel_credentials(url, access_client_id, access_client_secret, env_file)
            # Record tunnel for scan command
            try:
                from synth_ai.utils.tunnel_records import record_tunnel
                record_tunnel(
                    url=url,
                    port=cfg.port,
                    mode="managed",
                    pid=tunnel_proc.pid if tunnel_proc else None,
                    hostname=hostname,
                    local_host=cfg.host,
                    task_app_path=str(cfg.task_app_path) if cfg.task_app_path else None,
                )
            except Exception:
                pass  # Fail silently - records are optional

        # If wait or keep_alive is True, block and keep processes alive until interrupted
        if wait or keep_alive:
            _keep_tunnel_alive(cfg.port, url)
        else:
            # Background mode: print URL and return immediately
            # Processes will keep running in background
            print(f"✓ Tunnel ready: {url}")
            print(f"⏳ Tunnel running in background (PID: {tunnel_proc.pid if tunnel_proc else 'N/A'})")
            print("   Press Ctrl+C in this process to stop, or use: pkill -f cloudflared")

        return url

    except Exception as exc:
        # Clean up tunnel process on error
        if tunnel_proc:
            stop_tunnel(tunnel_proc)
            _TUNNEL_PROCESSES.pop(cfg.port, None)
        # Remove record if it was created
        try:
            from synth_ai.utils.tunnel_records import remove_tunnel_record
            remove_tunnel_record(cfg.port)
        except Exception:
            pass
        raise RuntimeError(f"Failed to deploy tunnel: {exc}") from exc


def _keep_tunnel_alive(port: int, url: str) -> None:
    """
    Keep tunnel processes alive until interrupted.

    This function blocks and monitors the tunnel process, similar to how
    local deployments block. Users can interrupt with Ctrl+C to stop.

    Args:
        port: Port the tunnel is running on
        url: Public tunnel URL (for display)
    """

    def signal_handler(signum, frame):  # noqa: ARG001
        """Handle SIGINT/SIGTERM to cleanup gracefully."""
        if port in _TUNNEL_PROCESSES:
            stop_tunnel(_TUNNEL_PROCESSES[port])
            _TUNNEL_PROCESSES.pop(port, None)
        sys.exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"✓ Tunnel ready: {url}")
    print("⏳ Keeping tunnel running... (Press Ctrl+C to stop)")

    try:
        # Monitor tunnel process and keep alive
        while True:
            if port in _TUNNEL_PROCESSES:
                proc = _TUNNEL_PROCESSES[port]
                if isinstance(proc, subprocess.Popen) and proc.poll() is not None:
                    print(f"❌ Tunnel process exited with code {proc.returncode}")
                    break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup on exit
        if port in _TUNNEL_PROCESSES:
            stop_tunnel(_TUNNEL_PROCESSES[port])
            _TUNNEL_PROCESSES.pop(port, None)
        print("\n🛑 Tunnel stopped")
