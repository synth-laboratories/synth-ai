"""Rust-core backed tunnel helpers."""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import synth_ai_py


def stop_tunnel(proc: Any) -> None:
    synth_ai_py.stop_tunnel(proc)


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
