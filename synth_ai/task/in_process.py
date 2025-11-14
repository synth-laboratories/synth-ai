"""In-process task app support for local development and demos."""

from __future__ import annotations

import asyncio
import logging
import signal
import socket
from pathlib import Path
from typing import Any, Callable, Optional

from synth_ai.cloudflare import (
    ensure_cloudflared_installed,
    open_quick_tunnel,
    start_uvicorn_background,
    stop_tunnel,
    wait_for_health_check,
)
from synth_ai.task.server import TaskAppConfig, create_task_app
from synth_ai.utils.apps import get_asgi_app, load_file_to_module
from synth_ai.utils.paths import REPO_ROOT, configure_import_paths
from uvicorn._types import ASGIApplication

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


class InProcessTaskApp:
    """
    Context manager for running task apps in-process with automatic Cloudflare tunnels.
    
    This class simplifies local development and demos by:
    1. Starting a task app server in a background thread
    2. Opening a Cloudflare tunnel automatically
    3. Providing the tunnel URL for GEPA/MIPRO jobs
    4. Cleaning up everything on exit
    
    Supports multiple input methods:
    - FastAPI app instance (most direct)
    - TaskAppConfig object
    - Config factory function (Callable[[], TaskAppConfig])
    - Task app file path (fallback for compatibility)
    
    Example:
        ```python
        from synth_ai.task.in_process import InProcessTaskApp
        from heartdisease_task_app import build_config
        
        async with InProcessTaskApp(
            config_factory=build_config,
            port=8114,
        ) as task_app:
            print(f"Task app running at: {task_app.url}")
            # Use task_app.url for GEPA jobs
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
        api_key: Optional[str] = None,
        health_check_timeout: float = 30.0,
        auto_find_port: bool = True,
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
            host: Host to bind to (default: 127.0.0.1, must be localhost)
            tunnel_mode: Tunnel mode ("quick" for ephemeral tunnels)
            api_key: API key for health checks (defaults to ENVIRONMENT_API_KEY env var)
            health_check_timeout: Max time to wait for health check in seconds
            auto_find_port: If True, automatically find available port if requested port is busy
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

        # Validate host (must be localhost for security)
        allowed_hosts = ("127.0.0.1", "localhost", "0.0.0.0")
        if host not in allowed_hosts:
            raise ValueError(
                f"Host must be one of {allowed_hosts} for security reasons, got {host}"
            )

        # Validate tunnel_mode
        if tunnel_mode not in ("quick",):
            raise ValueError(f"tunnel_mode must be 'quick', got {tunnel_mode}")

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
        self.api_key = api_key
        self.health_check_timeout = health_check_timeout
        self.auto_find_port = auto_find_port
        self.on_start = on_start
        self.on_stop = on_stop

        self.url: Optional[str] = None
        self._tunnel_proc: Optional[Any] = None
        self._app: Optional[ASGIApplication] = None
        self._original_port = port  # Track original requested port

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
            self._app = create_task_app(self._config)

        elif self._config_factory:
            # Callable - call it to get config, then create app
            config = self._config_factory()
            self._app = create_task_app(config)

        elif self._task_app_path:
            # File path - load module and extract app
            configure_import_paths(self._task_app_path, REPO_ROOT)
            module = load_file_to_module(
                self._task_app_path,
                f"_inprocess_{self._task_app_path.stem}_{id(self)}",
            )
            
            # Try to get app directly first
            try:
                self._app = get_asgi_app(module)
            except RuntimeError:
                # If no app found, try to get build_config function
                build_config = getattr(module, "build_config", None)
                if build_config and callable(build_config):
                    config = build_config()
                    self._app = create_task_app(config)
                else:
                    # Try registry lookup as last resort
                    from synth_ai.task.apps import registry
                    app_id = getattr(module, "APP_ID", None) or self._task_app_path.stem
                    entry = registry.get(app_id)
                    if entry and entry.config_factory:
                        config = entry.config_factory()
                        self._app = create_task_app(config)
                    else:
                        raise RuntimeError(
                            f"Task app at {self._task_app_path} must expose either:\n"
                            f"  - An ASGI app via `app = FastAPI(...)` or factory function\n"
                            f"  - A `build_config()` function that returns TaskAppConfig\n"
                            f"  - Be registered with register_task_app()"
                        ) from None

        # 2. Start uvicorn in background thread
        # Use daemon=False to ensure cleanup on exit
        logger.debug(f"Starting uvicorn server on {self.host}:{self.port}")
        start_uvicorn_background(self._app, self.host, self.port, daemon=False)

        # 3. Wait for health check
        api_key = self.api_key or self._get_api_key()
        logger.debug(f"Waiting for health check on {self.host}:{self.port}")
        await wait_for_health_check(
            self.host, self.port, api_key, timeout=self.health_check_timeout
        )
        logger.info(f"Health check passed for {self.host}:{self.port}")

        # 4. Ensure cloudflared is installed
        ensure_cloudflared_installed()

        # 5. Open tunnel
        if self.tunnel_mode == "quick":
            logger.info("Opening Cloudflare quick tunnel...")
            self.url, self._tunnel_proc = open_quick_tunnel(self.port, wait_s=15.0)
            logger.info(f"Tunnel opened: {self.url}")
        else:
            raise ValueError(f"Unsupported tunnel_mode: {self.tunnel_mode}")

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

    def _get_api_key(self) -> str:
        """Get API key from environment or default."""
        import os

        return os.getenv("ENVIRONMENT_API_KEY", "test")


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


# Set up signal handlers on module import
_setup_signal_handlers()
