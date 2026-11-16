"""Real MCP transports for the J1 MCP driver: stdio child and hosted /mcp.

See SYN-5693. Both expose the blocking `ToolCaller` protocol over one
long-lived `ClientSession`. The session is entered and exited by a single
host task on a private event-loop thread (anyio scopes must not cross tasks);
calls are submitted to that loop.

The stdio server (`synth-ai-mcp`) takes its configuration from its environment
only. It has no client-injection seam, so the caller supplies the child
environment (origin + credential); nothing is placed in argv.
"""

from __future__ import annotations

# ruff: noqa: UP045 -- Explicit Optional at acceptance contract boundaries.
import asyncio
import threading
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Optional

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent
from pydantic import JsonValue

from .j1 import AdapterError, refusal
from .limits import MCP_CALL_SECONDS
from .mcp_driver import MCPRefusal, ToolOutcome

__all__ = ["HostedEndpoint", "SessionToolCaller", "StdioCommand", "hosted_session", "stdio_session"]

SessionOpener = Callable[[], AbstractAsyncContextManager[ClientSession]]


@dataclass(frozen=True, slots=True)
class StdioCommand:
    # e.g. (<venv>/bin/synth-ai-mcp, "--env", "local"); config lives in env.
    executable: Path
    arguments: tuple[str, ...]
    environment: Mapping[str, str]
    working_directory: Path


@dataclass(frozen=True, slots=True)
class HostedEndpoint:
    url: str
    # Opaque bearer for /mcp (Core mcpoauth resolves an MCP OAuth access token).
    bearer: str


def stdio_session(command: StdioCommand) -> SessionOpener:
    @asynccontextmanager
    async def open_session() -> AsyncGenerator[ClientSession]:
        parameters = StdioServerParameters(
            command=str(command.executable),
            args=list(command.arguments),
            env=dict(command.environment),
            cwd=command.working_directory,
        )
        async with (
            stdio_client(parameters) as (reader, writer),
            ClientSession(reader, writer) as session,
        ):
            await session.initialize()
            yield session

    return open_session


def hosted_session(endpoint: HostedEndpoint) -> SessionOpener:
    @asynccontextmanager
    async def open_session() -> AsyncGenerator[ClientSession]:
        async with (
            create_mcp_http_client(
                headers={"Authorization": "Bearer " + endpoint.bearer}
            ) as http_client,
            streamable_http_client(endpoint.url, http_client=http_client) as (
                reader,
                writer,
                _session_id,
            ),
            ClientSession(reader, writer) as session,
        ):
            await session.initialize()
            yield session

    return open_session


def outcome_of(result: CallToolResult) -> ToolOutcome:
    texts: Sequence[str] = [item.text for item in result.content if isinstance(item, TextContent)]
    if len(texts) != 1 or len(result.content) != 1:
        raise refusal(
            MCPRefusal.RESULT_INVALID,
            stage="tools/call",
            cause="tool result must carry exactly one text content",
            run_id="transport",
        )
    return ToolOutcome(result.isError, texts[0])


class SessionToolCaller:
    """Blocking `ToolCaller` over one MCP session hosted on a private loop."""

    def __init__(self, opener: SessionOpener) -> None:
        self._opener = opener
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="mcp-session", daemon=True)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop: Optional[asyncio.Event] = None
        self._session: Optional[ClientSession] = None
        self._names: frozenset[str] = frozenset()
        self._failure: Optional[BaseException] = None

    def _run(self) -> None:
        asyncio.run(self._host())

    async def _host(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        try:
            async with self._opener() as session:
                listing = await session.list_tools()
                self._names = frozenset(tool.name for tool in listing.tools)
                self._session = session
                self._ready.set()
                await self._stop.wait()
        except (McpError, httpx.HTTPError, OSError, ExceptionGroup) as error:
            # Retained for the caller; it is re-raised as a typed refusal.
            self._failure = error
        finally:
            self._session = None
            self._ready.set()

    def __enter__(self) -> SessionToolCaller:
        self._thread.start()
        if not self._ready.wait(MCP_CALL_SECONDS) or self._session is None:
            raise self._transport_refusal("session_open")
        return self

    def __exit__(
        self,
        kind: Optional[type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(MCP_CALL_SECONDS)

    def _transport_refusal(self, stage: str) -> AdapterError:
        cause = "no session" if self._failure is None else type(self._failure).__name__
        return refusal(MCPRefusal.TRANSPORT_FAILED, stage=stage, cause=cause, run_id="transport")

    def tools(self) -> frozenset[str]:
        return self._names

    def call(self, name: str, arguments: Mapping[str, JsonValue]) -> ToolOutcome:
        session, loop = self._session, self._loop
        if session is None or loop is None:
            raise self._transport_refusal(name)
        future = asyncio.run_coroutine_threadsafe(session.call_tool(name, dict(arguments)), loop)
        try:
            result = future.result(MCP_CALL_SECONDS)
        except (McpError, httpx.HTTPError, FutureTimeout) as error:
            future.cancel()
            raise refusal(
                MCPRefusal.TRANSPORT_FAILED,
                stage=name,
                cause=type(error).__name__,
                run_id="transport",
            ) from error
        return outcome_of(result)
