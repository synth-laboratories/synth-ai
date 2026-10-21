"""Closed strict DEEP progress notifications within the active MCP request."""

from contextlib import contextmanager
from contextvars import ContextVar

from synth_ai.errors import ContractError
from synth_ai.sdk.index.types.search import StrictSearchProgress

_PUBLISH = ContextVar("synth_mcp_strict_progress", default=None)


@contextmanager
def progress_scope(publish):
    token = _PUBLISH.set(publish)
    try:
        yield
    finally:
        _PUBLISH.reset(token)


async def publish_progress(value: StrictSearchProgress):
    publish = _PUBLISH.get()
    if publish is None:
        raise ContractError(
            "Strict DEEP MCP delivery requires an active progress token",
            code="progress_token_required",
        )
    await publish(value)
