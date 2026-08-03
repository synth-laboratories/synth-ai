"""The legacy Research Intern ``/sessions`` plane is gated behind explicit opt-in.

QA for the Sync Intern release treats any legacy ``/smr/research-intern/sessions``
hit as a hard fail, so the SDK raises :class:`LegacyInternSessionsDisabledError`
by default and the MCP server does not even register the legacy tools unless
``SYNTH_ALLOW_LEGACY_INTERN_SESSIONS=1`` is set.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from synth_ai.mcp.research.tools.research_intern import (
    LEGACY_INTERN_SESSION_TOOL_NAMES,
    build_research_intern_tools,
)
from synth_ai.sdk.research.research_intern import (
    LEGACY_INTERN_SESSIONS_ENV,
    AsyncResearchInternSessionsAPI,
    LegacyInternSessionsDisabledError,
    ResearchInternSessionsAPI,
    legacy_intern_sessions_enabled,
)


class _TransportReachedError(Exception):
    """Sentinel proving a call passed the gate and reached the transport."""


class _SentinelTransport:
    """Fails loudly if any method is invoked; never performs network I/O."""

    def __getattr__(self, name: str) -> Any:
        def _fail(*args: Any, **kwargs: Any) -> Any:
            raise _TransportReachedError(name)

        return _fail


def _sync_api(*, allow_legacy: bool = False) -> ResearchInternSessionsAPI:
    return ResearchInternSessionsAPI(
        cast(Any, _SentinelTransport()),
        cast(Any, None),
        allow_legacy=allow_legacy,
    )


def _async_api(*, allow_legacy: bool = False) -> AsyncResearchInternSessionsAPI:
    return AsyncResearchInternSessionsAPI(
        cast(Any, _SentinelTransport()),
        cast(Any, None),
        allow_legacy=allow_legacy,
    )


@pytest.fixture(autouse=True)
def _no_env_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LEGACY_INTERN_SESSIONS_ENV, raising=False)


def test_legacy_sessions_disabled_by_default() -> None:
    api = _sync_api()
    with pytest.raises(LegacyInternSessionsDisabledError) as excinfo:
        api.list()
    assert excinfo.value.operation == "list"
    assert "sync-sessions" in str(excinfo.value)
    assert LEGACY_INTERN_SESSIONS_ENV in str(excinfo.value)


def test_every_legacy_sync_method_is_gated_at_entry() -> None:
    api = _sync_api()
    calls: list[tuple[str, tuple[Any, ...]]] = [
        ("create", (cast(Any, None),)),
        ("list", ()),
        ("retrieve", ("sess",)),
        ("append_event", ("sess", cast(Any, None))),
        ("list_events", ("sess",)),
        ("watch", ("sess",)),
        ("sync", ("sess",)),
        ("turn", ("sess", cast(Any, None))),
        ("close", ("sess", cast(Any, None))),
        ("publish_trace", ("sess", cast(Any, None))),
        ("create_reactive", (cast(Any, None),)),
        ("connect", ("sess",)),
    ]
    for name, args in calls:
        with pytest.raises(LegacyInternSessionsDisabledError):
            getattr(api, name)(*args)
    with pytest.raises(LegacyInternSessionsDisabledError):
        next(iter(api.stream_events("sess")))


def test_every_legacy_async_method_is_gated_at_entry() -> None:
    api = _async_api()
    calls: list[tuple[str, tuple[Any, ...]]] = [
        ("create", (cast(Any, None),)),
        ("list", ()),
        ("retrieve", ("sess",)),
        ("append_event", ("sess", cast(Any, None))),
        ("list_events", ("sess",)),
        ("watch", ("sess",)),
        ("sync", ("sess",)),
        ("turn", ("sess", cast(Any, None))),
        ("close", ("sess", cast(Any, None))),
        ("publish_trace", ("sess", cast(Any, None))),
        ("create_reactive", (cast(Any, None),)),
        ("connect", ("sess",)),
    ]
    for name, args in calls:
        with pytest.raises(LegacyInternSessionsDisabledError):
            asyncio.run(getattr(api, name)(*args))

    async def _first_stream_frame() -> None:
        async for _ in api.stream_events("sess"):
            break

    with pytest.raises(LegacyInternSessionsDisabledError):
        asyncio.run(_first_stream_frame())


def test_constructor_flag_opt_in_opens_the_gate() -> None:
    api = _sync_api(allow_legacy=True)
    with pytest.raises(_TransportReachedError):
        api.retrieve("sess")


def test_env_var_opt_in_opens_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LEGACY_INTERN_SESSIONS_ENV, "1")
    assert legacy_intern_sessions_enabled()
    api = _sync_api()
    with pytest.raises(_TransportReachedError):
        api.retrieve("sess")


def test_async_opt_in_opens_the_gate() -> None:
    api = _async_api(allow_legacy=True)
    with pytest.raises(_TransportReachedError):
        asyncio.run(api.retrieve("sess"))


def test_client_flag_threads_through_to_sessions_api() -> None:
    from synth_ai.sdk.research.client import Client

    gated = Client(api_key="sk-test")
    with pytest.raises(LegacyInternSessionsDisabledError):
        gated.intern.sessions._require_legacy_enabled("list")

    allowed = Client(api_key="sk-test", allow_legacy_intern_sessions=True)
    allowed.intern.sessions._require_legacy_enabled("list")


def test_front_door_client_flag_threads_through_to_sessions_api() -> None:
    from synth_ai import AsyncSynthClient, SynthClient

    sync_client = SynthClient(
        api_key="sk-test",
        allow_legacy_intern_sessions=True,
    )
    sync_client.research.intern.sessions._require_legacy_enabled("list")

    async_client = AsyncSynthClient(
        api_key="sk-test",
        allow_legacy_intern_sessions=True,
    )
    async_client.research.intern.sessions._require_legacy_enabled("list")


def test_mcp_tool_list_excludes_legacy_tools_by_default() -> None:
    names = {tool.name for tool in build_research_intern_tools(cast(Any, None))}
    assert not names & LEGACY_INTERN_SESSION_TOOL_NAMES
    assert "intern_sync_create" in names
    assert "intern_async_ensure" in names
    assert "research_provision_research_intern" in names


def test_mcp_tool_list_includes_legacy_tools_with_env_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(LEGACY_INTERN_SESSIONS_ENV, "1")
    names = {tool.name for tool in build_research_intern_tools(cast(Any, None))}
    assert names >= LEGACY_INTERN_SESSION_TOOL_NAMES
