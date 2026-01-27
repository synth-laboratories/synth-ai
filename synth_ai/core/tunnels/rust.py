"""Rust-core backed tunnel helpers."""

from __future__ import annotations

import asyncio
from typing import Any

import synth_ai_py


def get_cloudflared_path(prefer_system: bool = True) -> str | None:
    return synth_ai_py.get_cloudflared_path(prefer_system)


def ensure_cloudflared_installed(force: bool = False) -> str:
    return synth_ai_py.ensure_cloudflared_installed(force)


def require_cloudflared() -> str:
    return synth_ai_py.require_cloudflared()


def open_quick_tunnel(port: int, wait_s: float = 10.0) -> tuple[str, Any]:
    return synth_ai_py.open_quick_tunnel(port, wait_s)


async def open_quick_tunnel_with_dns_verification(
    port: int,
    *,
    wait_s: float = 10.0,
    api_key: str | None = None,
) -> tuple[str, Any]:
    return await asyncio.to_thread(
        synth_ai_py.open_quick_tunnel_with_dns_verification, port, wait_s, True, api_key
    )


def open_managed_tunnel(tunnel_token: str) -> Any:
    return synth_ai_py.open_managed_tunnel(tunnel_token)


async def open_managed_tunnel_with_connection_wait(
    tunnel_token: str,
    timeout_seconds: float = 30.0,
) -> Any:
    return await asyncio.to_thread(
        synth_ai_py.open_managed_tunnel_with_connection_wait,
        tunnel_token,
        timeout_seconds,
    )


def stop_tunnel(proc: Any) -> None:
    synth_ai_py.stop_tunnel(proc)


async def rotate_tunnel(api_key: str, port: int, backend_url: str | None = None) -> dict[str, Any]:
    return await asyncio.to_thread(synth_ai_py.rotate_tunnel, api_key, port, backend_url)


async def create_tunnel(
    api_key: str, port: int, subdomain: str | None = None
) -> dict[str, Any]:
    return await asyncio.to_thread(synth_ai_py.create_tunnel, api_key, port, subdomain)


async def verify_tunnel_dns_resolution(
    tunnel_url: str,
    name: str = "tunnel",
    timeout_seconds: float = 60.0,
    api_key: str | None = None,
) -> None:
    _ = name
    await asyncio.to_thread(
        synth_ai_py.verify_tunnel_dns_resolution, tunnel_url, timeout_seconds, api_key
    )


async def wait_for_health_check(
    host: str,
    port: int,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> None:
    await asyncio.to_thread(synth_ai_py.wait_for_health_check, host, port, api_key, timeout)


def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    return synth_ai_py.is_port_available(port, host)


def find_available_port(start_port: int, host: str = "0.0.0.0", max_attempts: int = 100) -> int:
    return synth_ai_py.find_available_port(start_port, host, max_attempts)


def kill_port(port: int) -> bool:
    return synth_ai_py.kill_port(port)


def acquire_port(
    port: int,
    on_conflict: str = "fail",
    host: str = "0.0.0.0",
    max_search: int = 100,
) -> int:
    behavior = on_conflict
    if hasattr(on_conflict, "value"):
        behavior = on_conflict.value  # type: ignore[assignment]
    return synth_ai_py.acquire_port(port, behavior, host, max_search)
