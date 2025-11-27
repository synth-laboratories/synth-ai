"""In-process task app support for local development and demos."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import threading
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import httpx
import uvicorn
from uvicorn._types import ASGIApplication

from synth_ai.core.apps.common import get_asgi_app, load_module
from synth_ai.core.integrations.cloudflare import (
    ensure_cloudflared_installed,
    open_quick_tunnel_with_dns_verification,
    stop_tunnel,
    wait_for_health_check,
)
from synth_ai.core.paths import REPO_ROOT, configure_import_paths
from synth_ai.sdk.task.server import TaskAppConfig, create_task_app

logger = logging.getLogger(__name__)

# Global registry for signal handlers
_registered_instances: set[InProcessTaskApp] = set()


def _find_available_port(host: str, start_port: int, max_attempts: int = 100) -> int:
    """
    Find an available port starting from start_port.
    
    Args:
        host: Host to bind to
        start_port: Starting port number
        max_attempts: Maximum number of ports to try
        
    Returns:
        Available port number
        
    Raises:
        RuntimeError: If no available port found
    """
    for offset in range(max_attempts):
        port = start_port + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
                return port
        except OSError:
            continue
    
    raise RuntimeError(
        f"Could not find available port starting from {start_port} "
        f"(tried {max_attempts} ports)"
    )


def _is_port_available(host: str, port: int) -> bool:
    """Check if a port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
    except OSError:
        return False




def _kill_process_on_port(host: str, port: int) -> None:
    """
    Attempt to kill any process using the specified port.
    
    Note: This is a best-effort operation and may not work on all systems.
    """
    import subprocess
    import sys
    
    try:
        if sys.platform == "win32":
            # Windows
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, check=False
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) > 4:
                        pid = parts[-1]
                        try:
                            subprocess.run(
                                ["taskkill", "/F", "/PID", pid],
                                capture_output=True,
                                check=False,
                            )
                            logger.info(f"Killed process {pid} on port {port}")
                        except Exception:
                            pass
        else:
            # Unix-like (macOS, Linux)
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split()
                for pid in pids:
                    try:
                        subprocess.run(
                            ["kill", "-9", pid],
                            capture_output=True,
                            check=False,
                        )
                        logger.info(f"Killed process {pid} on port {port}")
                    except Exception:
                        pass
    except Exception as e:
        logger.debug(f"Could not kill process on port {port}: {e}")


async def _verify_tunnel_ready(
    tunnel_url: str,
    api_key: str,
    *,
    max_retries: int | None = None,
    retry_delay: float | None = None,
    timeout_per_request: float = 10.0,
    verify_tls: bool = True,
) -> bool:
    """
    Verify that a Cloudflare tunnel is actually routing traffic (not just DNS-resolvable).
    
    A tunnel can resolve via DNS before HTTP routing is ready. This helper polls both
    /health and /task_info until they return 200 or retries are exhausted.
    
    Environment variables for tuning:
        SYNTH_TUNNEL_VERIFY_MAX_RETRIES: Number of retry attempts (default: 30)
        SYNTH_TUNNEL_VERIFY_DELAY_SECS: Delay between retries in seconds (default: 2.0)
    
    With defaults, waits up to 60 seconds for tunnel to become ready.
    """
    # Allow env var overrides for reliability tuning
    if max_retries is None:
        max_retries = int(os.getenv("SYNTH_TUNNEL_VERIFY_MAX_RETRIES", "30"))
    if retry_delay is None:
        retry_delay = float(os.getenv("SYNTH_TUNNEL_VERIFY_DELAY_SECS", "2.0"))
    
    # Initial delay before first check - tunnels often need a moment after DNS resolves
    initial_delay = float(os.getenv("SYNTH_TUNNEL_VERIFY_INITIAL_DELAY_SECS", "3.0"))
    if initial_delay > 0:
        logger.debug(f"Waiting {initial_delay}s for tunnel to stabilize before verification...")
        await asyncio.sleep(initial_delay)
    
    base = tunnel_url.rstrip("/")
    headers = {
        "X-API-Key": api_key,
        "Authorization": f"Bearer {api_key}",
    }
    aliases = (os.getenv("ENVIRONMENT_API_KEY_ALIASES") or "").strip()
    if aliases:
        headers["X-API-Keys"] = ",".join(
            [api_key, *[p.strip() for p in aliases.split(",") if p.strip()]]
        )
    
    logger.info(f"Verifying tunnel is routing traffic (max {max_retries} attempts, {retry_delay}s delay)...")

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout_per_request, verify=verify_tls) as client:
                health = await client.get(f"{base}/health", headers=headers)
                task_info = await client.get(f"{base}/task_info", headers=headers)

            if health.status_code == 200 and task_info.status_code == 200:
                logger.info(
                    f"Tunnel ready after {attempt + 1} attempt(s): "
                    f"health={health.status_code}, task_info={task_info.status_code}"
                )
                return True

            logger.debug(
                "Tunnel not ready (attempt %s/%s): health=%s task_info=%s",
                attempt + 1,
                max_retries,
                health.status_code,
                task_info.status_code,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "Tunnel readiness check failed (attempt %s/%s): %s",
                attempt + 1,
                max_retries,
                exc,
            )

        if attempt < max_retries - 1:
            # Log progress periodically (every 5 attempts)
            if (attempt + 1) % 5 == 0:
                elapsed = (attempt + 1) * retry_delay
                remaining = (max_retries - attempt - 1) * retry_delay
                logger.info(
                    f"Still waiting for tunnel... ({elapsed:.0f}s elapsed, {remaining:.0f}s remaining)"
                )
            await asyncio.sleep(retry_delay)

    logger.warning(f"Tunnel verification exhausted after {max_retries} attempts")
    return False


async def _verify_preconfigured_url_ready(
    tunnel_url: str,
    api_key: str,
    *,
    extra_headers: Optional[dict[str, str]] = None,
    max_retries: int = 10,
    retry_delay: float = 1.0,
    timeout_per_request: float = 10.0,
) -> bool:
    """
    Verify that a preconfigured tunnel URL is routing traffic.
    
    This is similar to _verify_tunnel_ready but designed for external tunnel
    providers (ngrok, etc.) where we don't control the tunnel setup.
    
    Args:
        tunnel_url: The external tunnel URL to verify
        api_key: API key for task app authentication
        extra_headers: Additional headers for the tunnel (e.g., auth tokens)
        max_retries: Number of retry attempts
        retry_delay: Delay between retries in seconds
        timeout_per_request: Timeout for each HTTP request
        
    Returns:
        True if tunnel is accessible, False otherwise
    """
    base = tunnel_url.rstrip("/")
    headers = {
        "X-API-Key": api_key,
        "Authorization": f"Bearer {api_key}",
    }
    
    # Add any extra headers (e.g., custom auth tokens)
    if extra_headers:
        headers.update(extra_headers)
    
    logger.info(f"Verifying preconfigured URL is accessible (max {max_retries} attempts)...")
    
    for attempt in range(max_retries):
        try:
            # Use verify=False since external tunnels may have different certs
            async with httpx.AsyncClient(timeout=timeout_per_request, verify=False) as client:
                health = await client.get(f"{base}/health", headers=headers)
            
            # Accept various success codes - the tunnel is working if we get any response
            if health.status_code in (200, 400, 401, 403, 404, 405):
                logger.info(
                    f"Preconfigured URL ready after {attempt + 1} attempt(s): "
                    f"health={health.status_code}"
                )
                return True
            
            logger.debug(
                "Preconfigured URL not ready (attempt %s/%s): health=%s",
                attempt + 1,
                max_retries,
                health.status_code,
            )
        except Exception as exc:
            logger.debug(
                "Preconfigured URL check failed (attempt %s/%s): %s",
                attempt + 1,
                max_retries,
                exc,
            )
        
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)
    
    logger.warning(f"Preconfigured URL verification exhausted after {max_retries} attempts")
    return False


class InProcessTaskApp:
    """
    Context manager for running task apps in-process with automatic tunneling.
    
    This class simplifies local development and demos by:
    1. Starting a task app server in a background thread
    2. Opening a tunnel automatically (Cloudflare by default, or use preconfigured URL)
    3. Providing the tunnel URL for GEPA/MIPRO jobs
    4. Cleaning up everything on exit
    
    Supports multiple input methods:
    - FastAPI app instance (most direct)
    - TaskAppConfig object
    - Config factory function (Callable[[], TaskAppConfig])
    - Task app file path (fallback for compatibility)
    
    Tunnel modes:
    - "quick": Cloudflare quick tunnel (default for local dev)
    - "named": Cloudflare named/managed tunnel
    - "local": No tunnel, use localhost URL directly
    - "preconfigured": Use externally-provided URL (set via preconfigured_url param or
      SYNTH_TASK_APP_URL env var). Useful for ngrok or other external tunnel providers.
    
    Example:
        ```python
        from synth_ai.sdk.task.in_process import InProcessTaskApp
        from heartdisease_task_app import build_config
        
        # Default: use Cloudflare quick tunnel
        async with InProcessTaskApp(
            config_factory=build_config,
            port=8114,
        ) as task_app:
            print(f"Task app running at: {task_app.url}")
        
        # Use preconfigured URL (e.g., from ngrok, localtunnel, etc.)
        async with InProcessTaskApp(
            config_factory=build_config,
            port=8000,
            tunnel_mode="preconfigured",
            preconfigured_url="https://abc123.ngrok.io",
        ) as task_app:
            print(f"Task app running at: {task_app.url}")
        ```
    """

    def __init__(
        self,
        *,
        app: Optional[ASGIApplication] = None,
        config: Optional[TaskAppConfig] = None,
        config_factory: Optional[Callable[[], TaskAppConfig]] = None,
        task_app_path: Optional[Path | str] = None,
        port: int = 8114,
        host: str = "127.0.0.1",
        tunnel_mode: str = "quick",
        preconfigured_url: Optional[str] = None,
        preconfigured_auth_header: Optional[str] = None,
        preconfigured_auth_token: Optional[str] = None,
        api_key: Optional[str] = None,
        health_check_timeout: float = 30.0,
        auto_find_port: bool = True,
        skip_tunnel_verification: bool = False,
        on_start: Optional[Callable[[InProcessTaskApp], None]] = None,
        on_stop: Optional[Callable[[InProcessTaskApp], None]] = None,
    ):
        """
        Initialize in-process task app.
        
        Args:
            app: FastAPI app instance (most direct)
            config: TaskAppConfig object
            config_factory: Callable that returns TaskAppConfig
            task_app_path: Path to task app .py file (fallback)
            port: Local port to run server on
            host: Host to bind to (default: 127.0.0.1, use 0.0.0.0 for external access)
            tunnel_mode: Tunnel mode - "quick", "named", "local", or "preconfigured"
            preconfigured_url: External tunnel URL to use when tunnel_mode="preconfigured".
                              Can also be set via SYNTH_TASK_APP_URL env var.
            preconfigured_auth_header: Optional auth header name for preconfigured URL
                                       (e.g., "x-custom-auth-token")
            preconfigured_auth_token: Optional auth token value for preconfigured URL
            api_key: API key for health checks (defaults to ENVIRONMENT_API_KEY env var)
            health_check_timeout: Max time to wait for health check in seconds
            auto_find_port: If True, automatically find available port if requested port is busy
            skip_tunnel_verification: If True, skip HTTP verification of tunnel connectivity.
                                      Useful when the tunnel URL is known to be valid.
            on_start: Optional callback called when task app starts (receives self)
            on_stop: Optional callback called when task app stops (receives self)
            
        Raises:
            ValueError: If multiple or no input methods provided, or invalid parameters
            FileNotFoundError: If task_app_path doesn't exist
        """
        # Validate: exactly one input method
        inputs = [app, config, config_factory, task_app_path]
        if sum(x is not None for x in inputs) != 1:
            raise ValueError(
                "Must provide exactly one of: app, config, config_factory, or task_app_path"
            )

        # Validate port range
        if not (1024 <= port <= 65535):
            raise ValueError(f"Port must be in range [1024, 65535], got {port}")

        # Validate host (allow 0.0.0.0 for container environments)
        allowed_hosts = ("127.0.0.1", "localhost", "0.0.0.0")
        if host not in allowed_hosts:
            raise ValueError(
                f"Host must be one of {allowed_hosts} for security reasons, got {host}"
            )

        # Validate tunnel_mode
        valid_modes = ("local", "quick", "named", "preconfigured")
        if tunnel_mode not in valid_modes:
            raise ValueError(f"tunnel_mode must be one of {valid_modes}, got {tunnel_mode}")

        # Validate task_app_path if provided
        if task_app_path:
            path = Path(task_app_path)
            if not path.exists():
                raise FileNotFoundError(f"Task app path does not exist: {task_app_path}")
            if path.suffix != ".py":
                raise ValueError(
                    f"Task app path must be a .py file, got {task_app_path}"
                )

        self._app_input = app
        self._config = config
        self._config_factory = config_factory
        self._task_app_path = Path(task_app_path) if task_app_path else None

        self.port = port
        self.host = host
        self.tunnel_mode = tunnel_mode
        self.preconfigured_url = preconfigured_url
        self.preconfigured_auth_header = preconfigured_auth_header
        self.preconfigured_auth_token = preconfigured_auth_token
        self.api_key = api_key
        self.health_check_timeout = health_check_timeout
        self.auto_find_port = auto_find_port
        self.skip_tunnel_verification = skip_tunnel_verification
        self.on_start = on_start
        self.on_stop = on_stop

        self.url: Optional[str] = None
        self._tunnel_proc: Optional[Any] = None
        self._app: Optional[ASGIApplication] = None
        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._server_thread: Optional[Any] = None
        self._original_port = port  # Track original requested port
        self._is_preconfigured = False  # Track if using preconfigured URL

    async def __aenter__(self) -> InProcessTaskApp:
        """Start task app and tunnel."""
        logger.info(f"Starting in-process task app on {self.host}:{self.port}")

        # Handle port conflicts
        if not _is_port_available(self.host, self.port):
            if self.auto_find_port:
                logger.warning(
                    f"Port {self.port} is in use, attempting to find available port..."
                )
                self.port = _find_available_port(self.host, self.port)
                logger.info(f"Using port {self.port} instead")
            else:
                # Try to kill process on port
                logger.warning(
                    f"Port {self.port} is in use, attempting to free it..."
                )
                _kill_process_on_port(self.host, self.port)
                await asyncio.sleep(0.5)  # Brief wait for port to free
                
                if not _is_port_available(self.host, self.port):
                    raise RuntimeError(
                        f"Port {self.port} is still in use. "
                        "Set auto_find_port=True to automatically find an available port."
                    )

        # 1. Get FastAPI app from whichever input method was provided
        if self._app_input:
            # Direct app - use as-is
            self._app = self._app_input

        elif self._config:
            # TaskAppConfig - create app from it
            self._app = create_task_app(self._config)  # type: ignore[assignment]

        elif self._config_factory:
            # Callable - call it to get config, then create app
            config = self._config_factory()
            self._app = create_task_app(config)  # type: ignore[assignment]

        elif self._task_app_path:
            # File path - load module and extract app
            configure_import_paths(self._task_app_path, REPO_ROOT)
            module = load_module(
                self._task_app_path,
                f"_inprocess_{self._task_app_path.stem}_{id(self)}",
            )
            
            # Try to get app directly first
            try:
                self._app = get_asgi_app(module)  # type: ignore[assignment]
            except RuntimeError:
                # If no app found, try to get build_config function
                build_config = getattr(module, "build_config", None)
                if build_config and callable(build_config):
                    config = build_config()
                    self._app = create_task_app(config)  # type: ignore[assignment]
                else:
                    # Try registry lookup as last resort
                    from synth_ai.sdk.task.apps import registry
                    app_id = getattr(module, "APP_ID", None) or self._task_app_path.stem
                    entry = registry.get(app_id)
                    if entry and entry.config_factory:
                        config = entry.config_factory()
                        self._app = create_task_app(config)  # type: ignore[assignment]
                    else:
                        raise RuntimeError(
                            f"Task app at {self._task_app_path} must expose either:\n"
                            f"  - An ASGI app via `app = FastAPI(...)` or factory function\n"
                            f"  - A `build_config()` function that returns TaskAppConfig\n"
                            f"  - Be registered with register_task_app()"
                        ) from None

        # 2. Start uvicorn in background thread
        # Use daemon=True for local testing to allow quick exit
        # The thread will be killed when the process exits
        logger.debug(f"Starting uvicorn server on {self.host}:{self.port}")

        config = uvicorn.Config(
            self._app,  # type: ignore[arg-type]
            host=self.host,
            port=self.port,
            reload=False,
            log_level="info",
        )
        self._uvicorn_server = uvicorn.Server(config)

        def serve():
            try:
                self._uvicorn_server.run()  # type: ignore[attr-defined]
            except Exception as exc:
                logger.debug(f"Uvicorn server stopped: {exc}")
        
        self._server_thread = threading.Thread(
            target=serve,
            name=f"synth-uvicorn-{self.port}",
            daemon=True,  # Daemon thread dies when main process exits
        )
        self._server_thread.start()

        # 3. Wait for health check
        api_key = self.api_key or self._get_api_key()
        logger.debug(f"Waiting for health check on {self.host}:{self.port}")
        await wait_for_health_check(
            self.host, self.port, api_key, timeout=self.health_check_timeout
        )
        logger.info(f"Health check passed for {self.host}:{self.port}")

        # 4. Determine tunnel mode (env var can override)
        mode = os.getenv("SYNTH_TUNNEL_MODE", self.tunnel_mode)
        
        # Check for preconfigured URL via env var
        env_preconfigured_url = os.getenv("SYNTH_TASK_APP_URL")
        if env_preconfigured_url:
            mode = "preconfigured"
            self.preconfigured_url = env_preconfigured_url
            logger.info(f"Using preconfigured URL from SYNTH_TASK_APP_URL: {env_preconfigured_url}")
        
        override_host = os.getenv("SYNTH_TUNNEL_HOSTNAME")
        
        if mode == "preconfigured":
            # Preconfigured mode: use externally-provided URL (e.g., ngrok, localtunnel)
            # This bypasses Cloudflare entirely - the caller is responsible for the tunnel
            if not self.preconfigured_url:
                raise ValueError(
                    "tunnel_mode='preconfigured' requires preconfigured_url parameter "
                    "or SYNTH_TASK_APP_URL environment variable"
                )
            
            self.url = self.preconfigured_url.rstrip("/")
            self._tunnel_proc = None
            self._is_preconfigured = True
            logger.info(f"Using preconfigured tunnel URL: {self.url}")
            
            # Optionally verify the preconfigured URL is accessible
            if not self.skip_tunnel_verification:
                api_key = self.api_key or self._get_api_key()
                
                # Build headers including any custom auth for the tunnel
                extra_headers: dict[str, str] = {}
                if self.preconfigured_auth_header and self.preconfigured_auth_token:
                    extra_headers[self.preconfigured_auth_header] = self.preconfigured_auth_token
                
                ready = await _verify_preconfigured_url_ready(
                    self.url,
                    api_key,
                    extra_headers=extra_headers,
                    max_retries=10,  # Fewer retries - external URL should work quickly
                    retry_delay=1.0,
                )
                if ready:
                    logger.info(f"Preconfigured URL verified and ready: {self.url}")
                else:
                    logger.warning(
                        f"Preconfigured URL {self.url} may not be accessible. "
                        "Proceeding anyway - set skip_tunnel_verification=True to suppress this warning."
                    )
        elif mode == "local":
            # Local mode: skip tunnel, use localhost
            self.url = f"http://{self.host}:{self.port}"
            self._tunnel_proc = None
            logger.info(f"Using local mode: {self.url}")
        elif mode == "named":
            # Named tunnel mode: use a pre-configured Cloudflare named tunnel
            # Fetches the customer's tunnel hostname from the backend using their Synth API key
            # The named tunnel must be running separately (cloudflared tunnel run <name>)
            ensure_cloudflared_installed()
            
            # For tunnel config, we need the SYNTH_API_KEY (not ENVIRONMENT_API_KEY)
            from synth_ai.core.env import get_api_key as get_synth_api_key
            synth_api_key = get_synth_api_key()
            
            # Fetch customer's tunnel config from backend
            tunnel_config = await self._fetch_tunnel_config(synth_api_key)
            
            # For task app auth, use the environment API key
            api_key = self.api_key or self._get_api_key()
            named_host = tunnel_config.get("hostname")
            
            if not named_host:
                raise RuntimeError(
                    "No tunnel configured for your account. "
                    "Create one via the SDK: `synth tunnels create --subdomain myapp`, "
                    "or in the dashboard at https://app.usesynth.ai/tunnels, "
                    "or use tunnel_mode='quick' for automatic quick tunnels (less reliable)."
                )
            
            self.url = f"https://{named_host}"
            self._tunnel_proc = None  # Named tunnel runs separately
            
            # Verify the named tunnel is accessible
            ready = await _verify_tunnel_ready(
                self.url,
                api_key,
                max_retries=5,  # Fewer retries - named tunnel should work immediately
                retry_delay=2.0,
                verify_tls=_should_verify_tls(),
            )
            if not ready:
                tunnel_token = tunnel_config.get("tunnel_token", "")
                token_hint = f"\n  cloudflared tunnel run --token {tunnel_token[:20]}..." if tunnel_token else ""
                raise RuntimeError(
                    f"Your tunnel {self.url} is not accessible. "
                    f"Ensure cloudflared is running:{token_hint}\n"
                    "Or run: `synth tunnels run`"
                )
            logger.info(f"Using named tunnel: {self.url}")
        elif mode == "quick":
            # Quick tunnel mode: create tunnel with DNS verification and retry
            # Cloudflare quick tunnels can be flaky - retry with fresh tunnels if needed
            ensure_cloudflared_installed()
            
            api_key = self.api_key or self._get_api_key()
            max_tunnel_attempts = int(os.getenv("SYNTH_TUNNEL_MAX_ATTEMPTS", "3"))
            
            for tunnel_attempt in range(max_tunnel_attempts):
                if tunnel_attempt > 0:
                    logger.warning(
                        f"Tunnel attempt {tunnel_attempt + 1}/{max_tunnel_attempts} - "
                        "requesting fresh tunnel..."
                    )
                    # Kill the previous tunnel process if it exists
                    if self._tunnel_proc:
                        try:
                            self._tunnel_proc.terminate()
                            await asyncio.sleep(1)
                        except Exception:
                            pass
                
                logger.info("Opening Cloudflare quick tunnel...")
                try:
                    self.url, self._tunnel_proc = await open_quick_tunnel_with_dns_verification(
                        self.port, api_key=api_key
                    )
                except Exception as e:
                    logger.warning(f"Tunnel creation failed: {e}")
                    if tunnel_attempt == max_tunnel_attempts - 1:
                        raise
                    continue
                
                # Apply hostname override if provided
                if override_host:
                    parsed = urlparse(self.url)
                    self.url = f"{parsed.scheme}://{override_host}"
                    logger.info(f"Overriding hostname: {self.url}")
                
                logger.info(f"Tunnel opened: {self.url}")

                # Extra guard: wait for tunnel HTTP routing to become ready (not just DNS)
                ready = await _verify_tunnel_ready(
                    self.url,
                    api_key,
                    verify_tls=_should_verify_tls(),
                )
                if ready:
                    logger.info(f"Tunnel verified and ready: {self.url}")
                    break
                else:
                    logger.warning(
                        f"Tunnel {self.url} not routing traffic after verification. "
                        f"{'Retrying with fresh tunnel...' if tunnel_attempt < max_tunnel_attempts - 1 else 'Giving up.'}"
                    )
                    if tunnel_attempt == max_tunnel_attempts - 1:
                        raise RuntimeError(
                            f"Failed to establish working tunnel after {max_tunnel_attempts} attempts. "
                            f"Last tunnel URL: {self.url}. "
                            "This may indicate Cloudflare rate limiting or network issues. "
                            "Try: SYNTH_TUNNEL_MODE=local if the backend can reach localhost, "
                            "or use a named Cloudflare tunnel instead of quick tunnels."
                        )
        else:
            raise ValueError(f"Unknown SYNTH_TUNNEL_MODE: {mode}")

        # Register for signal handling
        _registered_instances.add(self)

        # Call on_start callback if provided
        if self.on_start:
            try:
                self.on_start(self)
            except Exception as e:
                logger.warning(f"on_start callback raised exception: {e}")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop tunnel and server."""
        logger.info("Stopping in-process task app...")

        # Unregister from signal handling
        _registered_instances.discard(self)

        # Call on_stop callback if provided
        if self.on_stop:
            try:
                self.on_stop(self)
            except Exception as e:
                logger.warning(f"on_stop callback raised exception: {e}")

        # Stop tunnel
        if self._tunnel_proc:
            logger.debug("Stopping Cloudflare tunnel...")
            stop_tunnel(self._tunnel_proc)
            self._tunnel_proc = None
            logger.info("Tunnel stopped")
        
        # Stop the uvicorn server thread gracefully to avoid killing the host process
        if self._server_thread and self._server_thread.is_alive():
            logger.debug("Stopping uvicorn server thread...")
            if self._uvicorn_server:
                self._uvicorn_server.should_exit = True
            self._server_thread.join(timeout=5.0)
            if self._server_thread.is_alive():
                if self._uvicorn_server:
                    # Last resort if graceful shutdown hangs
                    self._uvicorn_server.force_exit = True
                self._server_thread.join(timeout=1.0)
            if self._server_thread.is_alive():
                logger.warning(
                    "Uvicorn server thread did not stop cleanly; "
                    "it will exit with the main process"
                )
            self._server_thread = None
            self._uvicorn_server = None

    def _get_api_key(self) -> str:
        """Get API key from environment or default."""
        import os
        
        # Try to load .env file if available
        try:
            from dotenv import load_dotenv
            load_dotenv(override=False)
        except ImportError:
            pass

        return os.getenv("ENVIRONMENT_API_KEY", "test")

    async def _fetch_tunnel_config(self, api_key: str) -> dict:
        """Fetch the customer's tunnel configuration from the backend.
        
        Uses the existing /api/v1/tunnels/tunnel endpoint to get the customer's
        active tunnels. Returns the first active tunnel's config.
        
        Returns a dict with:
            - hostname: The customer's configured tunnel hostname (e.g., "myapp.usesynth.ai")
            - tunnel_token: The cloudflared tunnel token for running the tunnel
            - local_port: The local port the tunnel routes to
            - local_host: The local host the tunnel routes to
        """
        from synth_ai.core.env import get_backend_url
        
        backend_url = get_backend_url()
        url = f"{backend_url}/api/v1/tunnels/"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "X-API-Key": api_key,
                    },
                )
                
                if resp.status_code == 404:
                    logger.debug("No tunnels found for this API key")
                    return {}
                
                if resp.status_code == 401:
                    raise RuntimeError(
                        "Invalid API key. Please check your SYNTH_API_KEY."
                    )
                
                resp.raise_for_status()
                tunnels = resp.json()
                
                # Return the first active tunnel
                if tunnels and len(tunnels) > 0:
                    tunnel = tunnels[0]
                    return {
                        "hostname": tunnel.get("hostname"),
                        "tunnel_token": tunnel.get("tunnel_token"),
                        "local_port": tunnel.get("local_port", 8000),
                        "local_host": tunnel.get("local_host", "127.0.0.1"),
                    }
                
                return {}
                
            except httpx.HTTPStatusError as e:
                logger.warning(f"Failed to fetch tunnel config: {e}")
                return {}
            except Exception as e:
                logger.debug(f"Tunnel config fetch failed: {e}")
                return {}


def _setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        """Handle SIGINT/SIGTERM by cleaning up all registered instances."""
        logger.info(f"Received signal {signum}, cleaning up {len(_registered_instances)} instances...")
        for instance in list(_registered_instances):
            try:
                # Trigger cleanup
                if instance._tunnel_proc:
                    stop_tunnel(instance._tunnel_proc)
                    instance._tunnel_proc = None
            except Exception as e:
                logger.error(f"Error cleaning up instance: {e}")
        _registered_instances.clear()

    # Register handlers (only once)
    if not hasattr(_setup_signal_handlers, "_registered"):
        signal.signal(signal.SIGINT, signal_handler)  # type: ignore[misc]
        signal.signal(signal.SIGTERM, signal_handler)  # type: ignore[misc]
        _setup_signal_handlers._registered = True  # type: ignore[attr-defined]


def _should_verify_tls() -> bool:
    """Return True unless explicitly disabled via env."""
    val = (os.getenv("SYNTH_TUNNEL_VERIFY_TLS") or "true").strip().lower()
    return val not in ("0", "false", "no", "off")


# Set up signal handlers on module import
_setup_signal_handlers()
