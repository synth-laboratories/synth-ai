"""Rust-core backed tunnel helpers."""

from __future__ import annotations

import asyncio
import socket
import warnings
from typing import Any

import synth_ai_py


def _cloudflare_disabled() -> None:
    warnings.warn(
        "Cloudflare tunnel helpers are deprecated and disabled. Use SynthTunnel or NgrokManaged.",
        DeprecationWarning,
        stacklevel=3,
    )
    raise RuntimeError("Cloudflare tunnel helpers are disabled.")


def get_cloudflared_path(prefer_system: bool = True) -> str | None:
    _ = prefer_system
    _cloudflare_disabled()


def ensure_cloudflared_installed(force: bool = False) -> str:
    _ = force
    _cloudflare_disabled()


def require_cloudflared() -> str:
    _cloudflare_disabled()


def open_quick_tunnel(port: int, wait_s: float = 10.0) -> tuple[str, Any]:
    _ = (port, wait_s)
    _cloudflare_disabled()


async def open_quick_tunnel_with_dns_verification(
    port: int,
    *,
    wait_s: float = 10.0,
    api_key: str | None = None,
) -> tuple[str, Any]:
    _ = (port, wait_s, api_key)
    _cloudflare_disabled()


def open_managed_tunnel(tunnel_token: str) -> Any:
    _ = tunnel_token
    _cloudflare_disabled()


async def open_managed_tunnel_with_connection_wait(
    tunnel_token: str,
    timeout_seconds: float = 30.0,
) -> Any:
    _ = (tunnel_token, timeout_seconds)
    _cloudflare_disabled()


def stop_tunnel(proc: Any) -> None:
    synth_ai_py.stop_tunnel(proc)


async def rotate_tunnel(api_key: str, port: int, backend_url: str | None = None) -> dict[str, Any]:
    _ = (api_key, port, backend_url)
    _cloudflare_disabled()


async def create_tunnel(api_key: str, port: int, subdomain: str | None = None) -> dict[str, Any]:
    _ = (api_key, port, subdomain)
    _cloudflare_disabled()


async def verify_tunnel_dns_resolution(
    tunnel_url: str,
    name: str = "tunnel",
    timeout_seconds: float = 60.0,
    api_key: str | None = None,
) -> None:
    _ = (tunnel_url, name, timeout_seconds, api_key)
    _cloudflare_disabled()


async def wait_for_health_check(
    host: str,
    port: int,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> None:
    await asyncio.to_thread(synth_ai_py.wait_for_health_check, host, port, api_key, timeout)


def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    fn = getattr(synth_ai_py, "is_port_available", None)
    if callable(fn):
        return fn(port, host)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def find_available_port(start_port: int, host: str = "0.0.0.0", max_attempts: int = 100) -> int:
    fn = getattr(synth_ai_py, "find_available_port", None)
    if callable(fn):
        return fn(start_port, host, max_attempts)
    port = start_port
    for _ in range(max_attempts):
        if is_port_available(port, host):
            return port
        port += 1
    raise RuntimeError(f"No available port found starting at {start_port}")


def kill_port(port: int) -> bool:
    fn = getattr(synth_ai_py, "kill_port", None)
    if callable(fn):
        return fn(port)
    return False


def acquire_port(
    port: int,
    on_conflict: str = "fail",
    host: str = "0.0.0.0",
    max_search: int = 100,
) -> int:
    behavior = on_conflict
    if hasattr(on_conflict, "value"):
        behavior = on_conflict.value  # type: ignore[assignment]
    fn = getattr(synth_ai_py, "acquire_port", None)
    if callable(fn):
        return fn(port, behavior, host, max_search)
    behavior = str(behavior)
    if behavior == "find_new":
        return find_available_port(port, host, max_search)
    if behavior == "fail":
        if not is_port_available(port, host):
            raise RuntimeError(f"Port {port} is not available")
        return port
    # Best-effort strict for unsupported behaviors
    if is_port_available(port, host):
        return port
    return find_available_port(port, host, max_search)
