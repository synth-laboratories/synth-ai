"""Tunnel manager for orchestrating lease-based tunnels.

This module provides the high-level orchestration for the new
lease-based tunnel system. It coordinates:
1. Creating/attaching to leases
2. Starting the local gateway
3. Starting the cloudflared connector
4. Verifying readiness
5. Sending heartbeats
6. Cleanup on close

The manager is designed for the "magic" UX where users just call
`open()` and get a working tunnel URL.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

import httpx

from .backend_client import LeaseClient, get_lease_client
from .connector import ensure_connector_running, get_connector
from .errors import (
    LocalAppNotReadyError,
    TunnelError,
)
from .gateway import ensure_gateway_running, get_gateway
from .types import (
    Diagnostics,
    LeaseState,
    TunnelHandle,
)

logger = logging.getLogger(__name__)

# Default timeouts
DEFAULT_LOCAL_READY_TIMEOUT = 30.0
DEFAULT_PUBLIC_READY_TIMEOUT = 60.0
HEARTBEAT_INTERVAL = 30

# Client instance ID storage
CLIENT_INSTANCE_FILE = Path.home() / ".synth" / "client_instance_id"


def _get_client_instance_id() -> str:
    """Get or create a stable client instance ID for this machine.

    The ID is stored in ~/.synth/client_instance_id and persists
    across sessions. This ensures tunnel reuse works correctly.
    """
    try:
        CLIENT_INSTANCE_FILE.parent.mkdir(parents=True, exist_ok=True)

        if CLIENT_INSTANCE_FILE.exists():
            instance_id = CLIENT_INSTANCE_FILE.read_text().strip()
            if len(instance_id) >= 8:
                return instance_id

        # Generate new ID
        instance_id = f"client-{uuid.uuid4().hex[:16]}"
        CLIENT_INSTANCE_FILE.write_text(instance_id)
        logger.info("[MANAGER] Generated new client instance ID: %s", instance_id)
        return instance_id
    except Exception as e:
        # Fall back to a session-based ID if we can't persist
        logger.warning("[MANAGER] Could not persist client instance ID: %s", e)
        return f"session-{uuid.uuid4().hex[:16]}"


class TunnelManager:
    """Manager for lease-based tunnel operations.

    This class orchestrates the full tunnel lifecycle:
    1. Create lease with backend
    2. Start local gateway
    3. Configure gateway route
    4. Start cloudflared connector
    5. Verify local app readiness
    6. Verify public URL readiness
    7. Send periodic heartbeats
    8. Cleanup on close

    Example:
        async with TunnelManager() as manager:
            handle = await manager.open(local_port=8001)
            print(f"Public URL: {handle.url}")
            # ... use the tunnel ...
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        backend_url: Optional[str] = None,
        client_instance_id: Optional[str] = None,
    ):
        """Initialize the tunnel manager.

        Args:
            api_key: Synth API key (defaults to SYNTH_API_KEY env var)
            backend_url: Backend URL (defaults to production)
            client_instance_id: Override the auto-generated client ID
        """
        self._api_key = api_key
        self._backend_url = backend_url
        self._client_instance_id = client_instance_id or _get_client_instance_id()
        self._client: Optional[LeaseClient] = None
        self._active_handles: dict[str, TunnelHandle] = {}
        self._heartbeat_tasks: dict[str, asyncio.Task[Any]] = {}
        self._closed = False

    async def _get_client(self) -> LeaseClient:
        """Get or create the lease client."""
        if self._client is None:
            self._client = get_lease_client(
                api_key=self._api_key,
                backend_url=self._backend_url,
            )
        return self._client

    async def open(
        self,
        local_port: int,
        *,
        local_host: str = "127.0.0.1",
        app_name: Optional[str] = None,
        ttl_seconds: int = 3600,
        verify_local: bool = True,
        verify_public: bool = True,
        local_timeout: float = DEFAULT_LOCAL_READY_TIMEOUT,
        public_timeout: float = DEFAULT_PUBLIC_READY_TIMEOUT,
        progress: bool = False,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> TunnelHandle:
        """Open a tunnel to the local port.

        This is the main entry point for creating tunnels. It handles
        the full lifecycle automatically.

        Args:
            local_port: Local port to tunnel
            local_host: Local host to tunnel (must be 127.0.0.1 or localhost)
            app_name: Optional name for the app (for debugging)
            ttl_seconds: Requested lease TTL
            verify_local: Whether to verify local app is ready
            verify_public: Whether to verify public URL is accessible
            local_timeout: Timeout for local readiness check
            public_timeout: Timeout for public readiness check
            progress: If True, print status updates
            on_status: Callback for status updates

        Returns:
            TunnelHandle with public URL and control methods
        """
        if self._closed:
            raise TunnelError("Manager is closed")

        # Slow path threshold - show extra progress after this many seconds
        slow_threshold = 5.0
        _slow_message_shown = False

        def status(msg: str, hint: str | None = None) -> None:
            """Show status message, with optional hint for slow paths."""
            nonlocal _slow_message_shown
            if progress:
                import sys

                print(f"  {msg}", flush=True)
                sys.stderr.flush()
            if on_status:
                on_status(msg)
            logger.info("[MANAGER] %s", msg)

            # Show hint if provided and we haven't shown a slow message yet
            if hint and not _slow_message_shown:
                elapsed = time.monotonic() - start_time
                if elapsed > slow_threshold:
                    _slow_message_shown = True
                    if progress:
                        print(f"    (hint: {hint})", flush=True)

        start_time = time.monotonic()

        def log_step(event: str, **fields: Any) -> None:
            elapsed = time.monotonic() - start_time
            payload = " ".join(f"{k}={v}" for k, v in fields.items())
            logger.debug("[MANAGER] step=%s elapsed=%.3fs %s", event, elapsed, payload)

        def slow_path_hint() -> None:
            """Show a hint if we're on the slow path."""
            nonlocal _slow_message_shown
            elapsed = time.monotonic() - start_time
            if elapsed > slow_threshold and not _slow_message_shown and progress:
                _slow_message_shown = True
                print("    (taking longer than usual - use TUNNEL_DEBUG=1 for details)", flush=True)

        client = await self._get_client()
        gateway = get_gateway()
        connector = get_connector()

        status("Creating tunnel lease...")
        log_step("lease_create_start")

        # 1. Create lease
        lease = await client.create_lease(
            client_instance_id=self._client_instance_id,
            local_host=local_host,
            local_port=local_port,
            app_name=app_name,
            requested_ttl_seconds=ttl_seconds,
        )
        log_step(
            "lease_create_ok",
            lease_id=lease.lease_id,
            hostname=lease.hostname,
            route_prefix=lease.route_prefix,
        )

        try:
            # 2. Start gateway if needed
            if not gateway.is_running:
                status("Starting local gateway...")
                log_step("gateway_start")
                await ensure_gateway_running(lease.gateway_port, force=True)
                log_step("gateway_started", port=lease.gateway_port)

            # 3. Configure gateway route
            gateway.add_route(lease.route_prefix, local_host, local_port)
            log_step(
                "gateway_route_added", route=lease.route_prefix, target=f"{local_host}:{local_port}"
            )

            # 4. Verify local app is ready (optional)
            if verify_local:
                status(
                    f"Waiting for local app on port {local_port}...",
                    hint=f"ensure your app is listening on {local_host}:{local_port}",
                )
                log_step(
                    "local_verify_start", target=f"{local_host}:{local_port}", timeout=local_timeout
                )
                await self._verify_local_ready(local_host, local_port, local_timeout)
                log_step("local_verify_ok")

            # 5. Start connector if needed
            if not connector.is_connected:
                status(
                    "Connecting to Cloudflare edge...",
                    hint="first connection may take 10-20s",
                )
                log_step("connector_start")
                await ensure_connector_running(lease.tunnel_token)
                log_step("connector_connected", connected=connector.is_connected)
            else:
                status("Reusing existing connection...")
                log_step("connector_reuse")

            # Register lease with connector
            connector.register_lease(lease.lease_id)
            log_step("lease_registered", lease_id=lease.lease_id)

            # 6. Verify public URL via /__synth/ready endpoint (optional)
            if verify_public:
                status(
                    "Verifying public URL...",
                    hint="DNS propagation may take a few seconds",
                )
                log_step(
                    "public_verify_start",
                    hostname=lease.hostname,
                    route=lease.route_prefix,
                    timeout=public_timeout,
                )
                await self._verify_public_ready(
                    hostname=lease.hostname,
                    route_prefix=lease.route_prefix,
                    timeout=public_timeout,
                )
                log_step("public_verify_done")

            # 7. Send initial heartbeat
            await client.heartbeat(
                lease.lease_id,
                connected_to_edge=connector.is_connected,
                gateway_ready=gateway.is_running,
                local_ready=True,
            )
            log_step("heartbeat_sent", lease_id=lease.lease_id)

            # Update lease state
            lease.state = LeaseState.ACTIVE

            # Create handle
            handle = TunnelHandle(
                url=lease.public_url,
                hostname=lease.hostname,
                local_port=local_port,
                lease=lease,
                connector=connector.status,
                gateway=gateway.status,
            )

            # Store handle
            self._active_handles[lease.lease_id] = handle
            log_step("handle_stored", lease_id=lease.lease_id)

            # 8. Start heartbeat task
            log_step("heartbeat_task_creating", lease_id=lease.lease_id)
            try:
                task = asyncio.create_task(
                    self._heartbeat_loop(lease.lease_id),
                    name=f"heartbeat-{lease.lease_id[:8]}",
                )
                self._heartbeat_tasks[lease.lease_id] = task
                log_step(
                    "heartbeat_task_created",
                    lease_id=lease.lease_id,
                    task_name=task.get_name(),
                    task_done=task.done(),
                )
            except Exception as e:
                log_step("heartbeat_task_create_failed", error=str(e))
                raise

            status(f"Public URL: {lease.public_url}")
            log_step("open_complete", lease_id=lease.lease_id)

            # Detailed return tracing
            log_step("preparing_return", handle_url=handle.url)
            logger.info(
                "[MANAGER] TRACE: About to return handle. "
                "If you see this but not 'returned' in caller, hang is in return/await."
            )
            import sys

            sys.stdout.flush()
            sys.stderr.flush()

            return handle

        except Exception:
            # Cleanup on failure
            gateway.remove_route(lease.route_prefix)
            with contextlib.suppress(Exception):
                await client.release(lease.lease_id)
            raise

    async def _verify_local_ready(
        self,
        host: str,
        port: int,
        timeout: float,
    ) -> None:
        """Verify local app is responding to health checks."""
        start = asyncio.get_event_loop().time()
        url = f"http://{host}:{port}/health"

        # Ignore proxy env vars for local health checks to avoid hangs.
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(5.0, connect=2.0),
            trust_env=False,
        ) as client:
            while True:
                elapsed = asyncio.get_event_loop().time() - start
                if elapsed >= timeout:
                    raise LocalAppNotReadyError(port, timeout=timeout)

                try:
                    resp = await client.get(url)
                    if resp.status_code < 500:
                        return  # Any non-5xx is considered ready
                except httpx.ConnectError:
                    pass  # Not ready yet
                except Exception as e:
                    logger.debug("[MANAGER] Local health check error: %s", e)

                await asyncio.sleep(0.5)

    async def _verify_public_ready(
        self,
        hostname: str,
        route_prefix: str,
        timeout: float,
    ) -> None:
        """Verify public URL is accessible through the tunnel via /__synth/ready.

        This endpoint returns 200 only if the gateway can reach the target app,
        providing true end-to-end verification.
        """
        from .cloudflare import resolve_hostname_with_explicit_resolvers

        ready_url = f"https://{hostname}{route_prefix}/__synth/ready"
        start = asyncio.get_event_loop().time()
        last_error: str | None = None

        # Resolve hostname to IP to bypass system DNS lag
        try:
            resolved_ip = await resolve_hostname_with_explicit_resolvers(hostname)
        except Exception as e:
            logger.debug("[MANAGER] DNS resolution failed for %s: %s", hostname, e)
            resolved_ip = None

        headers = {}
        if self._api_key:
            headers["X-API-Key"] = self._api_key

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            verify=True,
        ) as client:
            while True:
                elapsed = asyncio.get_event_loop().time() - start
                if elapsed >= timeout:
                    logger.warning(
                        "[MANAGER] Public readiness timeout after %.1fs: %s (last error: %s)",
                        timeout,
                        ready_url,
                        last_error,
                    )
                    return  # Don't fail - tunnel may still work

                try:
                    # Use resolved IP if available to bypass DNS lag.
                    # Ensure SNI uses the hostname to avoid TLS handshake failures.
                    if resolved_ip:
                        resp = await client.get(
                            f"https://{resolved_ip}{route_prefix}/__synth/ready",
                            headers={**headers, "Host": hostname},
                            extensions={"sni_hostname": hostname},
                        )
                    else:
                        resp = await client.get(ready_url, headers=headers)

                    if resp.status_code == 200:
                        logger.debug("[MANAGER] Public readiness verified: %s", ready_url)
                        return
                    elif resp.status_code == 503:
                        last_error = "target_unreachable (503)"
                    elif resp.status_code == 404:
                        last_error = "route_not_found (404)"
                    else:
                        last_error = f"status={resp.status_code}"

                except httpx.ConnectError as e:
                    last_error = f"connect_error: {e}"
                except httpx.HTTPError as e:
                    last_error = f"http_error: {e}"
                    # Fallback to hostname-based verification if IP+SNI still fails.
                    if "tls" in str(e).lower() or "handshake" in str(e).lower():
                        resolved_ip = None
                except Exception as e:
                    last_error = f"error: {e}"

                logger.debug("[MANAGER] Public readiness retry: %s (%s)", ready_url, last_error)
                await asyncio.sleep(1.0)

    async def _heartbeat_loop(self, lease_id: str) -> None:
        """Send periodic heartbeats for a lease."""
        logger.info(
            "[MANAGER] HEARTBEAT_LOOP: Starting for lease_id=%s (task scheduled)",
            lease_id[:8],
        )
        try:
            client = await self._get_client()
            logger.debug("[MANAGER] HEARTBEAT_LOOP: Got client")
            connector = get_connector()
            logger.debug("[MANAGER] HEARTBEAT_LOOP: Got connector")
            gateway = get_gateway()
            logger.debug("[MANAGER] HEARTBEAT_LOOP: Got gateway, entering loop")
        except Exception as e:
            logger.error("[MANAGER] HEARTBEAT_LOOP: Setup failed: %s", e)
            raise

        iteration = 0
        while lease_id in self._active_handles:
            try:
                iteration += 1
                logger.debug(
                    "[MANAGER] HEARTBEAT_LOOP: iteration=%d sleeping for %ds",
                    iteration,
                    HEARTBEAT_INTERVAL,
                )
                await asyncio.sleep(HEARTBEAT_INTERVAL)

                if lease_id not in self._active_handles:
                    logger.debug("[MANAGER] HEARTBEAT_LOOP: Lease removed, exiting")
                    break

                logger.debug("[MANAGER] HEARTBEAT_LOOP: Sending heartbeat")
                action, next_interval = await client.heartbeat(
                    lease_id,
                    connected_to_edge=connector.is_connected,
                    gateway_ready=gateway.is_running,
                    local_ready=True,
                )

                if action == "restart_connector":
                    logger.warning("[MANAGER] Backend requested connector restart")
                    # Could implement restart logic here

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("[MANAGER] Heartbeat failed: %s", e)

    async def close(self, lease_id: str) -> None:
        """Close a specific tunnel lease.

        Note: This only closes the lease, not the underlying connector or
        gateway. Use shutdown() or the context manager when completely
        done with tunnels to ensure clean exit.

        Args:
            lease_id: The lease ID to close
        """
        if lease_id not in self._active_handles:
            return

        handle = self._active_handles.pop(lease_id)

        # Cancel heartbeat
        if lease_id in self._heartbeat_tasks:
            self._heartbeat_tasks[lease_id].cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_tasks[lease_id]
            del self._heartbeat_tasks[lease_id]

        # Remove gateway route
        gateway = get_gateway()
        gateway.remove_route(handle.lease.route_prefix)

        # Unregister from connector
        connector = get_connector()
        connector.unregister_lease(lease_id)

        # Release lease
        try:
            client = await self._get_client()
            await client.release(lease_id)
        except Exception as e:
            logger.warning("[MANAGER] Failed to release lease: %s", e)

        handle._closed = True
        logger.info("[MANAGER] Closed tunnel: %s", lease_id[:8])

    async def close_all(self) -> None:
        """Close all active tunnels."""
        lease_ids = list(self._active_handles.keys())
        for lease_id in lease_ids:
            await self.close(lease_id)

    async def shutdown(self) -> None:
        """Shutdown the manager and all resources.

        This stops the connector (cloudflared), gateway, and releases
        all leases. Call this when completely done with tunnels.
        """
        self._closed = True
        await self.close_all()

        # Stop global connector and gateway
        connector = get_connector()
        gateway = get_gateway()

        logger.debug("[MANAGER] Stopping connector...")
        await connector.stop()

        logger.debug("[MANAGER] Stopping gateway...")
        await gateway.stop()

        if self._client:
            await self._client.close()
            self._client = None

        logger.debug("[MANAGER] Shutdown complete")

    async def __aenter__(self) -> TunnelManager:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.shutdown()

    def get_diagnostics(self, lease_id: str) -> Optional[Diagnostics]:
        """Get diagnostic information for a lease.

        Args:
            lease_id: The lease ID

        Returns:
            Diagnostics object or None if lease not found
        """
        handle = self._active_handles.get(lease_id)
        if not handle:
            return None

        connector = get_connector()

        return Diagnostics(
            lease_id=lease_id,
            tunnel_id=handle.lease.managed_tunnel_id,
            client_instance_id=self._client_instance_id,
            hostname=handle.hostname,
            connector_state=handle.connector.state.value,
            gateway_state=handle.gateway.state.value,
            lease_state=handle.lease.state.value,
            last_error=handle.lease.diagnostics_hint,
            logs=connector.get_logs(50),
        )


# Global manager instance
_manager: Optional[TunnelManager] = None


def get_manager(
    api_key: Optional[str] = None,
    backend_url: Optional[str] = None,
) -> TunnelManager:
    """Get or create the global tunnel manager.

    Args:
        api_key: Synth API key
        backend_url: Backend URL

    Returns:
        TunnelManager instance
    """
    global _manager
    if _manager is None:
        _manager = TunnelManager(api_key=api_key, backend_url=backend_url)
    return _manager


@asynccontextmanager
async def open_tunnel(
    local_port: int,
    *,
    api_key: Optional[str] = None,
    backend_url: Optional[str] = None,
    **kwargs: Any,
) -> AsyncIterator[TunnelHandle]:
    """Context manager for opening a tunnel.

    This is the simplest way to create a tunnel:

        async with open_tunnel(8001) as t:
            print(f"Public URL: {t.url}")
            # ... use the tunnel ...

    Args:
        local_port: Local port to tunnel
        api_key: Synth API key
        backend_url: Backend URL
        **kwargs: Additional arguments for TunnelManager.open()

    Yields:
        TunnelHandle with public URL
    """
    manager = get_manager(api_key=api_key, backend_url=backend_url)
    handle = await manager.open(local_port, **kwargs)
    try:
        yield handle
    finally:
        await manager.close(handle.lease.lease_id)


async def quick_open(
    local_port: int,
    *,
    api_key: Optional[str] = None,
    progress: bool = True,
) -> TunnelHandle:
    """Quick one-liner to open a tunnel.

    Note: Remember to close the tunnel when done.

        handle = await quick_open(8001)
        print(f"Public URL: {handle.url}")
        # ... use it ...
        await get_manager().close(handle.lease.lease_id)

    For automatic cleanup, use the context manager instead:

        async with open_tunnel(8001) as handle:
            print(f"Public URL: {handle.url}")

    Args:
        local_port: Local port to tunnel
        api_key: Synth API key
        progress: If True, print status updates

    Returns:
        TunnelHandle with public URL
    """
    manager = get_manager(api_key=api_key)
    return await manager.open(local_port, progress=progress)
