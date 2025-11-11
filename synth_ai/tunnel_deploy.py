"""
Cloudflare Tunnel deployment handler.

Manages the full lifecycle of deploying a task app via Cloudflare Tunnel:
1. Start local task app (uvicorn) in background
2. Wait for health check
3. Open tunnel (quick or managed)
4. Write credentials to .env
"""
import asyncio
import os
import time
from pathlib import Path
from typing import Optional

import httpx

from synth_ai.api.tunnel import create_tunnel
from synth_ai.cfgs import CloudflareTunnelDeployCfg
from synth_ai.tunnel import open_managed_tunnel, open_quick_tunnel, stop_tunnel
from synth_ai.utils.apps import get_asgi_app, load_file_to_module
from synth_ai.utils.paths import REPO_ROOT, configure_import_paths
from synth_ai.utils.tunnel import store_tunnel_credentials
from synth_ai.utils.env import resolve_env_var
from uvicorn._types import ASGIApplication

import uvicorn


_TUNNEL_PROCESSES: dict[int, object] = {}  # Store tunnel process handles


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
) -> None:
    """
    Start uvicorn server in a background thread.
    
    Args:
        app: ASGI application
        host: Host to bind to
        port: Port to bind to
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
        daemon=True,
    )
    thread.start()


async def deploy_app_tunnel(
    cfg: CloudflareTunnelDeployCfg,
    env_file: Optional[Path] = None,
) -> str:
    """
    Deploy task app via Cloudflare Tunnel.
    
    This function:
    1. Starts the local task app (uvicorn) in background
    2. Waits for health check
    3. Opens tunnel (quick or managed)
    4. Writes tunnel URL and Access credentials to .env
    
    Args:
        cfg: Tunnel deployment configuration
        env_file: Optional path to .env file (defaults to .env in current directory)
    
    Returns:
        Public tunnel URL
    
    Raises:
        RuntimeError: If deployment fails at any step
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
    _start_uvicorn_background(app, cfg.host, cfg.port)
    
    # Wait for health check (with API key for authentication)
    await _wait_for_health_check(cfg.host, cfg.port, cfg.env_api_key)
    
    tunnel_proc = None
    try:
        if cfg.mode == "quick":
            # Quick tunnel: ephemeral, no backend API call
            url, tunnel_proc = open_quick_tunnel(cfg.port)
            _TUNNEL_PROCESSES[cfg.port] = tunnel_proc
            store_tunnel_credentials(url, None, None, env_file)
            return url
        
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
        
        return url
    
    except Exception as exc:
        # Clean up tunnel process on error
        if tunnel_proc:
            stop_tunnel(tunnel_proc)
            _TUNNEL_PROCESSES.pop(cfg.port, None)
        raise RuntimeError(f"Failed to deploy tunnel: {exc}") from exc

