"""Typed calls to Mnemosyne actor MCP tools (SYN-5338 / SYN-5347 client side).

These tools are served by the actor-isolated ``swarms_trace_v1`` program over
MCP ``tools/call``; they are not core.v1 HTTP operations, so nothing here is
part of an exported contract set or its fingerprints. The caller supplies an
``ActorToolCaller`` (the actor's MCP session); this module never opens one.

Limits enforced client-side, mirroring the backend owner
(``packages/orchestra_port/mnemosyne_trace_tools.py``):

* retained native archive: at most 8 MiB (``PROJECT_TRACE_ARCHIVE_BYTES_MAX``);
* one ``mnemosyne_read_project_trace`` range window: at most 1 MiB
  (``PROJECT_TRACE_CHUNK_BYTES_MAX``); larger archives are downloaded as
  digest-bound, resumable ranges (see ``trace_chunks``);
* ``mnemosyne_compare_project_traces``: 2 to 8 distinct pinned associations
  (see ``trace_compare``); the tool is only called when advertised.

Every refusal is a ``ContractError`` (or ``CapabilityUnavailableError``) with a
stable ``code``, ``stage`` and ``cause``; nothing is retried silently.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Collection, Mapping
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from synth_ai.errors import CapabilityUnavailableError, ContractError

SERVICE = "mnemosyne.actor_mcp"
READ_PROJECT_TRACE_TOOL = "mnemosyne_read_project_trace"
COMPARE_PROJECT_TRACES_TOOL = "mnemosyne_compare_project_traces"
PROJECT_TRACE_ARCHIVE_BYTES_MAX = 8 * 1024 * 1024
PROJECT_TRACE_CHUNK_BYTES_MAX = 1024 * 1024
PROJECT_TRACE_COMPARE_MIN = 2
PROJECT_TRACE_COMPARE_MAX = 8


class ActorTraceRefusal(StrEnum):
    TOOL_NOT_ADVERTISED = "actor_trace_tool_not_advertised"
    TOOL_ERROR = "actor_trace_tool_error"
    RESULT_INVALID = "actor_trace_result_invalid"
    ARCHIVE_TOO_LARGE = "actor_trace_archive_too_large"
    CHUNK_TOO_LARGE = "actor_trace_chunk_too_large"
    CHUNK_MISMATCH = "actor_trace_chunk_mismatch"
    CHUNK_DIGEST_MISMATCH = "actor_trace_chunk_digest_mismatch"
    ARCHIVE_DIGEST_MISMATCH = "actor_trace_archive_digest_mismatch"
    PARTIAL_INVALID = "actor_trace_partial_invalid"
    COMPARE_MISMATCH = "actor_trace_compare_mismatch"


def refuse(code: ActorTraceRefusal, stage: str, cause: str) -> ContractError:
    return ContractError(
        f"Mnemosyne actor trace refused at {stage}: {cause} [{code.value}]",
        code=code.value,
        stage=stage,
        cause=cause,
        service=SERVICE,
    )


@runtime_checkable
class ActorToolCaller(Protocol):
    """One actor MCP session: advertised tool names and ``tools/call`` results.

    ``call_tool`` returns the JSON-RPC ``result`` object (``content``,
    ``structuredContent``, ``isError``) for one call.
    """

    def tool_names(self) -> Collection[str]: ...

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]: ...


@runtime_checkable
class AsyncActorToolCaller(Protocol):
    def tool_names(self) -> Collection[str]: ...

    def call_tool(
        self, name: str, arguments: Mapping[str, Any]
    ) -> Awaitable[Mapping[str, Any]]: ...


def require_tool(caller: ActorToolCaller | AsyncActorToolCaller, name: str) -> None:
    """Capability gate: an unadvertised tool is never called."""
    if name not in set(caller.tool_names()):
        raise CapabilityUnavailableError(
            f"Actor MCP session does not advertise {name} "
            f"[{ActorTraceRefusal.TOOL_NOT_ADVERTISED.value}]",
            code=ActorTraceRefusal.TOOL_NOT_ADVERTISED.value,
            stage="capability",
            cause=f"{name} not in tools/list",
            service=SERVICE,
        )


def structured_result(result: Mapping[str, Any], stage: str) -> str:
    """Return the result JSON text; refuse tool errors and text/structured drift.

    Callers validate the text with ``model_validate_json`` so strict wire models
    see JSON inputs (arrays as tuples), as on every other SDK response path.
    """
    if not isinstance(result, Mapping):
        raise refuse(ActorTraceRefusal.RESULT_INVALID, stage, "result is not an object")
    if result.get("isError") is not False:
        raise refuse(ActorTraceRefusal.TOOL_ERROR, stage, _error_text(result))
    structured = result.get("structuredContent")
    if not isinstance(structured, Mapping):
        raise refuse(ActorTraceRefusal.RESULT_INVALID, stage, "structuredContent missing")
    content = result.get("content")
    texts = (
        [item.get("text") for item in content if isinstance(item, Mapping)]
        if isinstance(content, list)
        else []
    )
    if len(texts) != 1 or not isinstance(texts[0], str):
        raise refuse(ActorTraceRefusal.RESULT_INVALID, stage, "expected one text content")
    try:
        mirrored = json.loads(texts[0])
    except ValueError:
        raise refuse(ActorTraceRefusal.RESULT_INVALID, stage, "text content is not JSON") from None
    if mirrored != structured:
        raise refuse(ActorTraceRefusal.RESULT_INVALID, stage, "text and structured content differ")
    return texts[0]


def _error_text(result: Mapping[str, Any]) -> str:
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                return item["text"][:512]
    return "tool returned isError"
