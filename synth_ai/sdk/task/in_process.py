"""In-process task app support for local development and demos."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import httpx
import uvicorn
from uvicorn._types import ASGIApplication

from synth_ai.core.apps.common import get_asgi_app, load_module
from synth_ai.core.integrations.cloudflare import (
    create_tunnel,
    ensure_cloudflared_installed,
    open_managed_tunnel,
    open_quick_tunnel_with_dns_verification,
    rotate_tunnel,
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


async def _resolve_via_public_dns(hostname: str, timeout: float = 5.0) -> Optional[str]:
    """
    Resolve hostname using public DNS servers (1.1.1.1, 8.8.8.8).

    This bypasses local DNS caching issues that can cause NXDOMAIN errors
    when the local resolver has stale cached responses.

    Returns the first resolved IP address, or None if resolution fails.
    """
    loop = asyncio.get_event_loop()

    for dns_server in ("1.1.1.1", "8.8.8.8"):
        try:
            result = await loop.run_in_executor(
                None,
                lambda server=dns_server: subprocess.run(
                    ["dig", f"@{server}", "+short", hostname],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                ),
            )
            if result.returncode == 0 and result.stdout.strip():
                # Return first IP (dig may return multiple)
                first_ip = result.stdout.strip().splitlines()[0].strip()
                if first_ip and not first_ip.endswith("."):  # Skip CNAME responses
                    logger.debug(f"Resolved {hostname} via {dns_server}: {first_ip}")
                    return first_ip
        except FileNotFoundError:
            # dig not available, try socket resolution instead
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: socket.gethostbyname(hostname),
                )
                if result:
                    logger.debug(f"Resolved {hostname} via system DNS: {result}")
                    return result
            except socket.gaierror:
                pass
        except subprocess.TimeoutExpired:
            logger.debug(f"DNS timeout resolving {hostname} via {dns_server}")
        except Exception as e:
            logger.debug(f"DNS resolution error for {hostname} via {dns_server}: {e}")

    return None


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

    IMPORTANT: Uses public DNS (1.1.1.1) to bypass local DNS cache issues.
    Local DNS may have stale NXDOMAIN cached even after tunnel DNS is created.

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

    # Parse hostname from URL
    parsed = urlparse(tunnel_url)
    hostname = parsed.netloc
    scheme = parsed.scheme or "https"

    headers = {
        "X-API-Key": api_key,
        "Authorization": f"Bearer {api_key}",
        "Host": hostname,  # Always set Host header for IP-based requests
    }
    aliases = (os.getenv("ENVIRONMENT_API_KEY_ALIASES") or "").strip()
    if aliases:
        headers["X-API-Keys"] = ",".join(
            [api_key, *[p.strip() for p in aliases.split(",") if p.strip()]]
        )

    logger.info(f"Verifying tunnel is routing traffic (max {max_retries} attempts, {retry_delay}s delay)...")

    # Track resolved IP to avoid re-resolving every attempt
    resolved_ip: Optional[str] = None

    for attempt in range(max_retries):
        try:
            # Try to resolve IP via public DNS if we don't have it yet
            if resolved_ip is None:
                resolved_ip = await _resolve_via_public_dns(hostname)
                if resolved_ip:
                    logger.info(f"Resolved tunnel hostname via public DNS: {hostname} -> {resolved_ip}")

            # If we have a resolved IP, use curl with --resolve for proper SNI handling
            # httpx connecting to an IP directly fails SSL handshake due to SNI issues
            if resolved_ip:
                loop = asyncio.get_event_loop()

                # Use curl with --resolve to bypass local DNS while maintaining proper SNI
                async def curl_check(path: str) -> int:
                    try:
                        result = await loop.run_in_executor(
                            None,
                            lambda: subprocess.run(
                                [
                                    "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                                    "--resolve", f"{hostname}:443:{resolved_ip}",
                                    "-H", f"X-API-Key: {api_key}",
                                    "-H", f"Authorization: Bearer {api_key}",
                                    f"https://{hostname}{path}",
                                ],
                                capture_output=True,
                                text=True,
                                timeout=timeout_per_request,
                            ),
                        )
                        return int(result.stdout.strip()) if result.returncode == 0 else 0
                    except Exception:
                        return 0

                health_status = await curl_check("/health")
                task_info_status = await curl_check("/task_info")

                if health_status == 200 and task_info_status == 200:
                    logger.info(
                        f"Tunnel ready after {attempt + 1} attempt(s): "
                        f"health={health_status}, task_info={task_info_status}"
                    )
                    return True

                logger.debug(
                    "Tunnel not ready (attempt %s/%s): health=%s task_info=%s",
                    attempt + 1,
                    max_retries,
                    health_status,
                    task_info_status,
                )
            else:
                # Fall back to hostname-based request (local DNS)
                base = tunnel_url.rstrip("/")
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
            # Clear resolved IP on connection errors - might need to re-resolve
            if "connect" in str(exc).lower() or "resolve" in str(exc).lower():
                resolved_ip = None

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
            async with httpx.AsyncClient(timeout=timeout_per_request) as client:
                health = await client.get(f"{base}/health", headers=headers)
            
            # Only accept 200 for health checks - other codes may indicate misrouting
            if health.status_code == 200:
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
        skip_tunnel_verification: bool = True,  # Default True - verification is unreliable
        force_new_tunnel: bool = False,
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
            force_new_tunnel: If True, create a fresh tunnel instead of reusing existing one.
                             Use this when an existing managed tunnel is stale/broken.
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
        self.force_new_tunnel = force_new_tunnel
        self.on_start = on_start
        self.on_stop = on_stop

        self.url: Optional[str] = None
        self._tunnel_proc: Optional[Any] = None
        self._app: Optional[ASGIApplication] = None
        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._server_thread: Optional[Any] = None
        self._original_port = port  # Track original requested port
        self._is_preconfigured = False  # Track if using preconfigured URL
        self._dns_verified_by_backend = False  # Track if backend verified DNS propagation

    async def __aenter__(self) -> InProcessTaskApp:
        """Start task app and tunnel."""
        
        # For named tunnels, pre-fetch tunnel config to get the correct port
        # (existing tunnels are configured for a specific port)
        mode = os.getenv("SYNTH_TUNNEL_MODE", self.tunnel_mode)
        if mode == "named":
            try:
                from synth_ai.core.env import get_api_key as get_synth_api_key
                synth_api_key = get_synth_api_key()
                if synth_api_key is None:
                    raise ValueError("SYNTH_API_KEY is required for named tunnel mode")
                tunnel_config = await self._fetch_tunnel_config(synth_api_key)
                tunnel_port = tunnel_config.get("local_port")
                if tunnel_config.get("hostname") and tunnel_port and tunnel_port != self.port:
                    logger.info(
                        f"Existing managed tunnel is configured for port {tunnel_port}, "
                        f"adjusting from requested port {self.port}"
                    )
                    self.port = tunnel_port
                # Store config for later use to avoid re-fetching
                self._prefetched_tunnel_config = tunnel_config
            except Exception as e:
                logger.debug(f"Pre-fetch tunnel config failed: {e}")
                self._prefetched_tunnel_config = None
        else:
            self._prefetched_tunnel_config = None
        
        logger.info(f"Starting in-process task app on {self.host}:{self.port}")

        # For named tunnels, the port is baked into the tunnel config - we MUST use it
        tunnel_config = getattr(self, "_prefetched_tunnel_config", None) or {}
        tunnel_port = tunnel_config.get("local_port")
        is_named_tunnel_port = mode == "named" and tunnel_port and tunnel_port == self.port
        
        # Handle port conflicts
        if not _is_port_available(self.host, self.port):
            if is_named_tunnel_port:
                # Named tunnel port is REQUIRED - kill whatever is using it
                print(f"[CLOUDFLARE-FIX] Named tunnel requires port {self.port}, killing existing process...")
                logger.warning(
                    f"Named tunnel is configured for port {self.port}, killing existing process..."
                )
                _kill_process_on_port(self.host, self.port)
                await asyncio.sleep(1.0)  # Wait for port to free
                
                if not _is_port_available(self.host, self.port):
                    raise RuntimeError(
                        f"Named tunnel requires port {self.port} but it's still in use after kill attempt. "
                        "Manually kill the process using this port, or delete and recreate the tunnel."
                    )
                print(f"[CLOUDFLARE-FIX] Port {self.port} freed successfully")
            elif self.auto_find_port:
                print(f"Port {self.port} is in use, attempting to find available port...")
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
            # Named tunnel mode: fully automatic managed tunnel
            # 1. Check for existing tunnel
            # 2. Auto-create if none exists
            # 3. Auto-start cloudflared if not accessible
            # 4. Verify tunnel is working
            ensure_cloudflared_installed()
            
            # For tunnel config, we need the SYNTH_API_KEY (not ENVIRONMENT_API_KEY)
            from synth_ai.core.env import get_api_key as get_synth_api_key
            synth_api_key = get_synth_api_key()
            if synth_api_key is None:
                raise ValueError("SYNTH_API_KEY is required for named tunnel mode")

            # For task app auth, use the environment API key
            api_key = self.api_key or self._get_api_key()

            # Use pre-fetched config (port was already adjusted before server started)
            tunnel_config = getattr(self, "_prefetched_tunnel_config", None) or {}
            if not tunnel_config:
                # Fetch if not pre-fetched (shouldn't happen normally)
                tunnel_config = await self._fetch_tunnel_config(synth_api_key)
            
            named_host = tunnel_config.get("hostname")
            tunnel_token = tunnel_config.get("tunnel_token")
            
            # Track if backend verified DNS (so we can skip local verification)
            dns_verified_by_backend = False

            # Force ROTATE tunnel if requested (deletes old + creates new, stays within limits)
            if self.force_new_tunnel:
                print("[CLOUDFLARE-FIX] force_new_tunnel=True, rotating tunnel...")
                logger.info("force_new_tunnel=True, rotating tunnel (delete+create)")
                try:
                    rotated = await rotate_tunnel(
                        synth_api_key=synth_api_key,
                        port=self.port,
                        reason="force_new_tunnel=True",
                    )
                    named_host = rotated.get("hostname")
                    tunnel_token = rotated.get("tunnel_token")
                    dns_verified_by_backend = rotated.get("dns_verified", False)
                    print(f"[CLOUDFLARE-FIX] Rotated to fresh tunnel: {named_host}")
                    print(f"[CLOUDFLARE-FIX] DNS verified by backend: {dns_verified_by_backend}")
                    logger.info(f"Rotated to fresh managed tunnel: {named_host}, dns_verified={dns_verified_by_backend}")
                except Exception as e:
                    print(f"[CLOUDFLARE-FIX] Rotation failed: {e}, using existing tunnel: {named_host}")
                    logger.warning(f"Rotation failed: {e}, falling back to existing tunnel: {named_host}")
                    if not named_host or not tunnel_token:
                        raise RuntimeError(
                            f"Tunnel rotation failed and no existing tunnel found: {e}\n"
                            "Try using tunnel_mode='quick' instead."
                        ) from e
            # Auto-create tunnel if none exists
            elif not named_host:
                logger.info("No managed tunnel found, creating one automatically...")
                try:
                    # Generate subdomain from port or use default
                    subdomain = f"task-app-{self.port}"
                    new_tunnel = await create_tunnel(
                        synth_api_key=synth_api_key,
                        port=self.port,
                        subdomain=subdomain,
                    )
                    named_host = new_tunnel.get("hostname")
                    tunnel_token = new_tunnel.get("tunnel_token")
                    dns_verified_by_backend = new_tunnel.get("dns_verified", False)
                    logger.info(f"Created managed tunnel: {named_host}, dns_verified={dns_verified_by_backend}")
                except Exception as e:
                    # If tunnel creation fails, suggest using quick tunnels
                    raise RuntimeError(
                        f"Failed to create managed tunnel: {e}\n"
                        "This may be because the backend doesn't have Cloudflare configured.\n"
                        "Options:\n"
                        "  1. Use tunnel_mode='quick' for automatic quick tunnels\n"
                        "  2. Ask your admin to configure Cloudflare credentials on the backend"
                    ) from e
            
            if not named_host or not tunnel_token:
                raise RuntimeError(
                    "Tunnel configuration incomplete (missing hostname or token). "
                    "Try deleting and recreating the tunnel, or use tunnel_mode='quick'."
                )

            self.url = f"https://{named_host}"
            # Store dns_verified for use by job (to skip health check)
            self._dns_verified_by_backend = dns_verified_by_backend

            print(f"[CLOUDFLARE] Named tunnel URL: {self.url}")

            # CRITICAL: For Cloudflare managed tunnels, DNS will NOT resolve until cloudflared connects.
            # The DNS record exists in Cloudflare, but proxied CNAMEs to .cfargotunnel.com only
            # resolve when the tunnel has an active cloudflared connection.
            # Therefore, we MUST start cloudflared FIRST, then verify the tunnel works.

            # First, check if cloudflared is already running (tunnel might be accessible)
            ready = await _verify_tunnel_ready(
                self.url,
                api_key,
                max_retries=1,  # Single quick check
                retry_delay=0.5,
                verify_tls=_should_verify_tls(),
            )

            if ready:
                # Tunnel already accessible - cloudflared must be running elsewhere
                self._tunnel_proc = None
                print(f"[CLOUDFLARE] Tunnel already accessible (cloudflared running externally)")
                logger.info(f"Tunnel {self.url} is already accessible (cloudflared running externally)")
            else:
                # Tunnel not accessible - start cloudflared FIRST, then verify
                print(f"[CLOUDFLARE] Starting cloudflared (DNS requires active tunnel connection)...")
                logger.info(f"Starting cloudflared for {self.url}...")
                try:
                    self._tunnel_proc = open_managed_tunnel(tunnel_token)
                    print(f"[CLOUDFLARE] cloudflared started, PID={self._tunnel_proc.pid}")
                    logger.info(f"Started cloudflared (PID: {self._tunnel_proc.pid})")
                except Exception as e:
                    print(f"[CLOUDFLARE] ERROR starting cloudflared: {e}")
                    raise RuntimeError(
                        f"Failed to start cloudflared: {e}\n"
                        "Make sure cloudflared is installed: brew install cloudflare/cloudflare/cloudflared"
                    ) from e

                # Wait for cloudflared to connect and tunnel to become accessible
                print(f"[CLOUDFLARE] Waiting for tunnel to become accessible...")
                ready = await _verify_tunnel_ready(
                    self.url,
                    api_key,
                    max_retries=15,  # Up to ~30 seconds for tunnel to connect
                    retry_delay=2.0,
                    verify_tls=_should_verify_tls(),
                )

                if not ready:
                    # Tunnel still not accessible after starting cloudflared
                    # Clean up and try auto-rotation
                    if self._tunnel_proc:
                        stop_tunnel(self._tunnel_proc)
                        self._tunnel_proc = None

                    print(f"[CLOUDFLARE] Tunnel {self.url} not accessible, attempting rotation...")
                    logger.warning(f"Tunnel {self.url} failed to connect. Attempting rotation...")

                    try:
                        rotated = await rotate_tunnel(
                            synth_api_key=synth_api_key,
                            port=self.port,
                            reason=f"Tunnel {named_host} failed to connect",
                        )
                        named_host = rotated.get("hostname")
                        tunnel_token = rotated.get("tunnel_token")

                        if not named_host or not tunnel_token:
                            raise RuntimeError("Rotation returned incomplete tunnel config")

                        self.url = f"https://{named_host}"
                        print(f"[CLOUDFLARE] Rotated to new tunnel: {self.url}")

                        # Start cloudflared with the new token
                        self._tunnel_proc = open_managed_tunnel(tunnel_token)
                        print(f"[CLOUDFLARE] Started cloudflared for rotated tunnel, PID={self._tunnel_proc.pid}")

                        # Verify the new tunnel
                        ready = await _verify_tunnel_ready(
                            self.url,
                            api_key,
                            max_retries=15,
                            retry_delay=2.0,
                            verify_tls=_should_verify_tls(),
                        )

                        if not ready:
                            if self._tunnel_proc:
                                stop_tunnel(self._tunnel_proc)
                                self._tunnel_proc = None
                            raise RuntimeError(
                                f"Rotated tunnel {self.url} also failed. "
                                "Try using tunnel_mode='quick' instead."
                            )

                        print(f"[CLOUDFLARE] Rotated tunnel ready: {self.url}")

                    except Exception as rotate_err:
                        raise RuntimeError(
                            f"Tunnel failed and rotation failed: {rotate_err}\n"
                            "Try using tunnel_mode='quick' instead."
                        ) from rotate_err
                else:
                    print(f"[CLOUDFLARE] Tunnel connected and ready: {self.url}")

            logger.info(f"Using managed tunnel: {self.url}")
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
        
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
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
