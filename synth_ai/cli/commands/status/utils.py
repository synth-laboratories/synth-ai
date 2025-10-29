"""Shared utilities for status commands."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

import click
from rich.console import Console

from .config import DEFAULT_TIMEOUT, BackendConfig, resolve_backend_config

T = TypeVar("T")

console = Console()


def parse_relative_time(value: str | None) -> str | None:
    """Convert relative time expressions (e.g., '5m', '2h', '1d') to ISO strings."""
    if not value:
        return None
    token = value.strip().lower()
    if not token:
        return None
    multiplier = 1.0
    if token.endswith("ms"):
        multiplier = 0.001
        token = token[:-2]
    elif token.endswith("s"):
        multiplier = 1.0
        token = token[:-1]
    elif token.endswith("m"):
        multiplier = 60.0
        token = token[:-1]
    elif token.endswith("h"):
        multiplier = 3600.0
        token = token[:-1]
    elif token.endswith("d"):
        multiplier = 86400.0
        token = token[:-1]

    try:
        seconds = float(token) * multiplier
    except ValueError:
        return value

    dt = datetime.now(UTC) - timedelta(seconds=seconds)
    return dt.isoformat()


def ensure_async(fn: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    """Decorator to run an async callable via asyncio.run inside Click commands."""

    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


def resolve_context_config(
    ctx: click.Context,
    *,
    base_url: str | None,
    api_key: str | None,
    timeout: float | None,
) -> BackendConfig:
    if base_url is not None or api_key is not None or timeout not in (None, DEFAULT_TIMEOUT):
        return resolve_backend_config(base_url=base_url, api_key=api_key, timeout=timeout)
    obj = ctx.find_object(dict)
    if obj and isinstance(obj.get("status_backend_config"), BackendConfig):
        return obj["status_backend_config"]
    return resolve_backend_config(base_url=base_url, api_key=api_key, timeout=timeout)


def warn(message: str) -> None:
    console.print(f"[yellow]{message}[/yellow]")


def bail(message: str) -> None:
    raise click.ClickException(message)


def common_options() -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Apply shared backend CLI options to a command."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        options = [
            click.option(
                "--base-url",
                envvar="SYNTH_STATUS_BASE_URL",
                default=None,
                help="Override the Synth backend base URL for this command.",
            ),
            click.option(
                "--api-key",
                envvar="SYNTH_STATUS_API_KEY",
                default=None,
                help="API key for the Synth backend.",
            ),
            click.option(
                "--timeout",
                default=DEFAULT_TIMEOUT,
                show_default=True,
                type=float,
                help="HTTP request timeout in seconds.",
            ),
        ]
        for option in reversed(options):
            func = option(func)
        return func

    return decorator
