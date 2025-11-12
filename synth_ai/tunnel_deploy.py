"""
Cloudflare Tunnel deployment handler.

Manages the full lifecycle of deploying a task app via Cloudflare Tunnel:
1. Start local task app (uvicorn) in background
2. Wait for health check
3. Open tunnel (quick or managed)
4. Write credentials to .env
5. Optionally keep processes alive (blocking mode)

This module provides a clean abstraction that shields users from:
- Process management details (uvicorn threads, cloudflared subprocesses)
- Health check polling logic
- Credential storage
- Cleanup on errors
"""
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
from synth_ai.cfgs import CloudflareTunnelDeployCfg
from synth_ai.tunnel import open_managed_tunnel, open_quick_tunnel, stop_tunnel
from synth_ai.utils.apps import get_asgi_app, load_file_to_module
from synth_ai.utils.env import resolve_env_var
from synth_ai.utils.paths import REPO_ROOT, configure_import_paths
from synth_ai.utils.tunnel import store_tunnel_credentials
from uvicorn._types import ASGIApplication

import uvicorn

_TUNNEL_PROCESSES: dict[int, subprocess.Popen] = {}  # Store tunnel process handles for cleanup


async def _wait_for_health_check(
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


def _start_uvicorn_background(
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
) -> str:
    """
    Deploy task app via Cloudflare Tunnel.
    
    This function provides a clean abstraction that handles:
    1. Starting the local task app (uvicorn) in background
    2. Waiting for health check
    3. Opening tunnel (quick or managed)
    4. Writing tunnel URL and Access credentials to .env
    5. Optionally keeping processes alive (blocking vs non-blocking mode)
    
    When `keep_alive=True`, this function blocks and keeps the tunnel running
    until interrupted (Ctrl+C). This is similar to how `deploy_app_uvicorn`
    blocks for local deployments.
    
    When `keep_alive=False`, this function returns immediately after deployment.
    Processes run in the background and will continue until explicitly stopped
    or the parent process exits. Use this for headless/background deployments.
    
    Args:
        cfg: Tunnel deployment configuration
        env_file: Optional path to .env file (defaults to .env in current directory)
        keep_alive: If True, block and keep tunnel alive until interrupted.
                   If False, return immediately after deployment (background mode).
    
    Returns:
        Public tunnel URL
    
    Raises:
        RuntimeError: If deployment fails at any step
    
    Example:
        # Non-blocking (background mode, returns immediately)
        url = await deploy_app_tunnel(cfg, keep_alive=False)
        
        # Blocking (keeps tunnel alive, similar to local deployment)
        url = await deploy_app_tunnel(cfg, keep_alive=True)
    """
    # Set up environment
    os.environ["ENVIRONMENT_API_KEY"] = cfg.env_api_key
    if cfg.trace:
        os.environ["TASKAPP_TRACING_ENABLED"] = "1"
    else:
        os.environ.pop("TASKAPP_TRACING_ENABLED", None)
    
    # Load and configure task app
    configure_import_paths(cfg.task_app_path, REPO_ROOT)
    module = load_file_to_module(
        cfg.task_app_path,
        f"_synth_tunnel_task_app_{cfg.task_app_path.stem}",
    )
    app = get_asgi_app(module)
    
    # Start uvicorn in background
    # Use daemon=False if keep_alive=True (so thread doesn't die when we block)
    # Use daemon=True if keep_alive=False (background mode, dies with parent)
    _start_uvicorn_background(app, cfg.host, cfg.port, daemon=not keep_alive)
    
    # Wait for health check (with API key for authentication)
    await _wait_for_health_check(cfg.host, cfg.port, cfg.env_api_key)
    
    tunnel_proc = None
    try:
        if cfg.mode == "quick":
            # Quick tunnel: ephemeral, no backend API call
            url, tunnel_proc = open_quick_tunnel(cfg.port)
            _TUNNEL_PROCESSES[cfg.port] = tunnel_proc
            store_tunnel_credentials(url, None, None, env_file)
        else:
            # Managed tunnel: provision via backend API
            try:
                from synth_ai.api.tunnel import create_tunnel  # type: ignore[import-untyped]
            except ImportError as err:
                raise RuntimeError(
                    "Managed tunnel mode requires synth_ai.api.tunnel module. "
                    "This is only available in the managed tunnel implementation."
                ) from err
            
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
        
        # If keep_alive is True, block and keep processes alive until interrupted
        if keep_alive:
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
    import subprocess
    
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

