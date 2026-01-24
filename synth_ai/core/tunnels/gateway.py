"""Local gateway for routing tunnel traffic.

The gateway is a local HTTP proxy that:
1. Listens on a fixed port (default 8016)
2. Routes traffic based on route prefixes
3. Strips the prefix before forwarding
4. Provides a stable target for Cloudflare tunnel ingress

This allows multiple leases to share a single tunnel without
Cloudflare configuration changes.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, Optional

from .errors import GatewayError, GatewayPortInUseError, GatewayStartError
from .ports import is_port_available, kill_port
from .types import GatewayState, GatewayStatus

logger = logging.getLogger(__name__)

# Default gateway port
DEFAULT_GATEWAY_PORT = 8016


class TunnelGateway:
    """Local gateway for routing tunnel traffic.

    The gateway maintains a routing table that maps route prefixes
    to local targets. When a request comes in, it:
    1. Matches the path against route prefixes
    2. Strips the prefix from the path
    3. Forwards the request to the target

    Example:
        gateway = TunnelGateway(port=8016)
        await gateway.start()

        # Add a route
        gateway.add_route("/s/abc123", "127.0.0.1", 8001)

        # Request to https://hostname/s/abc123/api/task
        # is forwarded to http://127.0.0.1:8001/api/task
    """

    def __init__(self, port: int = DEFAULT_GATEWAY_PORT):
        """Initialize the gateway.

        Args:
            port: Port to listen on
        """
        self.port = port
        self._routes: dict[str, tuple[str, int]] = {}
        self._state = GatewayState.STOPPED
        self._server: Optional[Any] = None
        self._server_thread: Optional[threading.Thread] = None
        self._server_loop: Optional[asyncio.AbstractEventLoop] = None
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def status(self) -> GatewayStatus:
        """Get the current gateway status."""
        return GatewayStatus(
            state=self._state,
            port=self.port,
            routes=dict(self._routes),
            error=self._error,
        )

    @property
    def is_running(self) -> bool:
        """Check if the gateway is running."""
        return self._state == GatewayState.RUNNING

    def add_route(
        self,
        route_prefix: str,
        target_host: str,
        target_port: int,
    ) -> None:
        """Add or update a route.

        Args:
            route_prefix: The route prefix to match (e.g., "/s/abc123")
            target_host: Target host to forward to
            target_port: Target port to forward to
        """
        with self._lock:
            self._routes[route_prefix] = (target_host, target_port)
            logger.info(
                "[GATEWAY] Added route: %s -> %s:%d",
                route_prefix,
                target_host,
                target_port,
            )

    def remove_route(self, route_prefix: str) -> bool:
        """Remove a route.

        Args:
            route_prefix: The route prefix to remove

        Returns:
            True if the route was removed, False if not found
        """
        with self._lock:
            if route_prefix in self._routes:
                del self._routes[route_prefix]
                logger.info("[GATEWAY] Removed route: %s", route_prefix)
                return True
            return False

    def get_target(self, path: str) -> Optional[tuple[str, int, str]]:
        """Get the target for a path.

        Args:
            path: The request path

        Returns:
            Tuple of (host, port, stripped_path) or None if no match
        """
        with self._lock:
            for prefix, (host, port) in self._routes.items():
                if path.startswith(prefix):
                    # Strip the prefix from the path
                    stripped = path[len(prefix) :]
                    if not stripped:
                        stripped = "/"
                    elif not stripped.startswith("/"):
                        stripped = "/" + stripped
                    return (host, port, stripped)
            return None

    async def start(self, force: bool = False) -> None:
        """Start the gateway server.

        Args:
            force: If True, kill existing process on the port

        Raises:
            GatewayPortInUseError: If the port is in use and force=False
            GatewayStartError: If the server fails to start
        """
        if self._state == GatewayState.RUNNING:
            logger.debug("[GATEWAY] Already running on port %d", self.port)
            return

        self._state = GatewayState.STARTING
        self._error = None

        # Check port availability
        if not is_port_available(self.port):
            if force:
                logger.warning("[GATEWAY] Killing existing process on port %d", self.port)
                kill_port(self.port)
                await asyncio.sleep(0.5)  # Give it time to release
            else:
                self._state = GatewayState.ERROR
                self._error = f"Port {self.port} is in use"
                raise GatewayPortInUseError(self.port)

        try:
            # Create the ASGI app for routing
            app = self._create_app()

            # Start uvicorn in a background task
            import uvicorn

            config = uvicorn.Config(
                app,
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
                access_log=False,
            )
            server = uvicorn.Server(config)

            # Run uvicorn in a separate thread with its own event loop
            # This prevents deadlocks when sync code (e.g., subprocess.run) blocks the main thread
            self._server = server

            def run_server() -> None:
                logger.debug("[GATEWAY] Thread starting, creating new event loop")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._server_loop = loop
                logger.debug("[GATEWAY] Thread event loop created, starting uvicorn.serve()")
                try:
                    loop.run_until_complete(server.serve())
                    logger.debug("[GATEWAY] Thread: uvicorn.serve() completed normally")
                except Exception as e:
                    logger.error("[GATEWAY] Thread: uvicorn.serve() failed: %s", e)
                finally:
                    logger.debug("[GATEWAY] Thread: Closing event loop")
                    loop.close()
                    logger.debug("[GATEWAY] Thread: Event loop closed, thread exiting")

            logger.debug("[GATEWAY] Creating daemon thread for uvicorn server")
            self._server_thread = threading.Thread(
                target=run_server,
                daemon=True,
                name="gateway-uvicorn",
            )
            self._server_thread.start()
            logger.debug(
                "[GATEWAY] Thread started: name=%s daemon=%s alive=%s",
                self._server_thread.name,
                self._server_thread.daemon,
                self._server_thread.is_alive(),
            )

            # Wait for startup
            await asyncio.sleep(0.3)

            if not is_port_available(self.port):
                # Port is now in use by us
                self._state = GatewayState.RUNNING
                logger.info("[GATEWAY] Started on port %d", self.port)
            else:
                raise GatewayStartError("Server did not bind to port", port=self.port)

        except GatewayError:
            raise
        except Exception as e:
            self._state = GatewayState.ERROR
            self._error = str(e)
            raise GatewayStartError(str(e), port=self.port) from e

    async def stop(self) -> None:
        """Stop the gateway server."""
        if self._state == GatewayState.STOPPED:
            return

        logger.info("[GATEWAY] Stopping gateway on port %d", self.port)

        if self._server:
            self._server.should_exit = True
            if self._server_thread and self._server_thread.is_alive():
                # Wait for thread to finish (uvicorn will exit when should_exit=True)
                self._server_thread.join(timeout=5.0)

        self._server = None
        self._server_thread = None
        self._server_loop = None
        self._state = GatewayState.STOPPED
        self._routes.clear()
        logger.info("[GATEWAY] Stopped")

    def _create_app(self) -> Callable[..., Any]:
        """Create the ASGI application for routing."""
        import httpx

        async def app(
            scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]
        ) -> None:
            if scope["type"] == "lifespan":
                # Handle lifespan
                while True:
                    message = await receive()
                    if message["type"] == "lifespan.startup":
                        await send({"type": "lifespan.startup.complete"})
                    elif message["type"] == "lifespan.shutdown":
                        await send({"type": "lifespan.shutdown.complete"})
                        return
                return

            if scope["type"] != "http":
                return

            path = scope.get("path", "/")
            method = scope.get("method", "GET")

            # Special endpoint for gateway health (gateway itself only)
            if path == "/__synth/gateway/health":
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"status":"ok","gateway":"running"}',
                    }
                )
                return

            # Route-specific ready endpoint: /{route_prefix}/__synth/ready
            # Returns 200 only if target app is reachable
            if path.endswith("/__synth/ready"):
                route_prefix = path.rsplit("/__synth/ready", 1)[0]
                if route_prefix in self._routes:
                    target_host, target_port = self._routes[route_prefix]
                    # Extract API key from incoming request to forward to health probe
                    probe_headers: dict[str, str] = {}
                    for header_name, header_value in scope.get("headers", []):
                        name_lower = header_name.decode("latin-1").lower()
                        if name_lower in ("x-api-key", "x-api-keys", "authorization"):
                            probe_headers[header_name.decode("latin-1")] = header_value.decode(
                                "latin-1"
                            )
                    # Probe the target app
                    try:
                        async with httpx.AsyncClient(
                            timeout=httpx.Timeout(5.0, connect=2.0),
                            trust_env=False,
                        ) as client:
                            resp = await client.get(
                                f"http://{target_host}:{target_port}/health", headers=probe_headers
                            )
                            if resp.status_code < 500:
                                await send(
                                    {
                                        "type": "http.response.start",
                                        "status": 200,
                                        "headers": [(b"content-type", b"application/json")],
                                    }
                                )
                                await send(
                                    {
                                        "type": "http.response.body",
                                        "body": f'{{"status":"ok","route":"{route_prefix}","target":"{target_host}:{target_port}"}}'.encode(),
                                    }
                                )
                                return
                    except Exception as e:
                        logger.debug("[GATEWAY] Ready probe failed for %s: %s", route_prefix, e)
                    # Target not reachable
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 503,
                            "headers": [(b"content-type", b"application/json")],
                        }
                    )
                    await send(
                        {
                            "type": "http.response.body",
                            "body": f'{{"status":"unavailable","route":"{route_prefix}","error":"target_unreachable"}}'.encode(),
                        }
                    )
                    return
                # Route not found
                await send(
                    {
                        "type": "http.response.start",
                        "status": 404,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"status":"not_found","error":"route_not_found"}',
                    }
                )
                return

            # Find target
            target = self.get_target(path)
            if not target:
                # No route found - return 404
                await send(
                    {
                        "type": "http.response.start",
                        "status": 404,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"error":"no_route","message":"No route found for this path"}',
                    }
                )
                return

            target_host, target_port, stripped_path = target

            # Build the target URL
            query_string = scope.get("query_string", b"").decode()
            target_url = f"http://{target_host}:{target_port}{stripped_path}"
            if query_string:
                target_url = f"{target_url}?{query_string}"

            # Collect request body
            body_parts = []
            while True:
                message = await receive()
                body_parts.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            body = b"".join(body_parts)

            # Build headers (exclude hop-by-hop headers)
            hop_by_hop = {
                b"connection",
                b"keep-alive",
                b"proxy-authenticate",
                b"proxy-authorization",
                b"te",
                b"trailers",
                b"transfer-encoding",
                b"upgrade",
            }
            headers = [(k, v) for k, v in scope.get("headers", []) if k.lower() not in hop_by_hop]

            # Forward the request
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.request(
                        method=method,
                        url=target_url,
                        headers=headers,
                        content=body,
                    )

                    # Send response
                    response_headers = [
                        (
                            k.encode() if isinstance(k, str) else k,
                            v.encode() if isinstance(v, str) else v,
                        )
                        for k, v in response.headers.items()
                        if k.lower()
                        not in ("content-encoding", "transfer-encoding", "content-length")
                    ]
                    response_headers.append(
                        (b"content-length", str(len(response.content)).encode())
                    )

                    await send(
                        {
                            "type": "http.response.start",
                            "status": response.status_code,
                            "headers": response_headers,
                        }
                    )
                    await send(
                        {
                            "type": "http.response.body",
                            "body": response.content,
                        }
                    )

            except httpx.ConnectError:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 502,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": f'{{"error":"bad_gateway","message":"Cannot connect to {target_host}:{target_port}"}}'.encode(),
                    }
                )
            except Exception as e:
                logger.exception("[GATEWAY] Error forwarding request")
                await send(
                    {
                        "type": "http.response.start",
                        "status": 500,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": f'{{"error":"internal_error","message":"{str(e)}"}}'.encode(),
                    }
                )

        return app


# Global gateway instance
_gateway: Optional[TunnelGateway] = None
_gateway_lock = threading.Lock()


def get_gateway(port: int = DEFAULT_GATEWAY_PORT) -> TunnelGateway:
    """Get or create the global gateway instance.

    Args:
        port: Port for the gateway

    Returns:
        TunnelGateway instance
    """
    global _gateway
    with _gateway_lock:
        if _gateway is None or _gateway.port != port:
            _gateway = TunnelGateway(port=port)
        return _gateway


async def ensure_gateway_running(
    port: int = DEFAULT_GATEWAY_PORT,
    force: bool = False,
) -> TunnelGateway:
    """Ensure the gateway is running.

    Args:
        port: Port for the gateway
        force: If True, kill existing process on the port

    Returns:
        Running TunnelGateway instance
    """
    gateway = get_gateway(port)
    if not gateway.is_running:
        await gateway.start(force=force)
    return gateway
