"""Public resource reads preserve exact identity and incomplete disposition."""

from unittest.mock import AsyncMock, Mock

import pytest

from synth_ai.sdk.research.research_intern import (
    ResearchInternSyncRuntimeAPI,
    ResearchInternAsyncRuntimeAPI,
    AsyncResearchInternSyncRuntimeAPI,
    AsyncResearchInternAsyncRuntimeAPI,
)


def payload(kind):
    return dict(
        runtime_kind=kind,
        runtime_id="recorded",
        observed_at="2026-09-12T00:00:00Z",
        coverage="registered-runtime-resources-v1",
        coverage_complete=False,
        incomplete_reasons=["runtime_host_disposition_unavailable"],
        resources=[
            dict(
                resource_kind="intern_runtime",
                resource_id="recorded",
                relation="self",
                disposition="unknown",
                reason="runtime_host_disposition_unavailable",
            )
        ],
    )


@pytest.mark.parametrize(
    "kind,api,path",
    [
        (
            "sync",
            ResearchInternSyncRuntimeAPI,
            "/smr/research-intern/sync-sessions/recorded/resources",
        ),
        (
            "async",
            ResearchInternAsyncRuntimeAPI,
            "/smr/research-intern/async-assignments/recorded/resources",
        ),
    ],
)
def test_exact_recorded_resource_read(kind, api, path):
    transport = Mock()
    transport.execute.return_value = payload(kind)
    result = api(transport).resources("recorded")
    assert transport.execute.call_args.args[0].path == path
    assert not result.coverage_complete
    assert result.resources[0].disposition == "unknown"
    transport.execute.return_value = {**payload(kind), "runtime_id": "replacement"}
    with pytest.raises(ValueError, match="identity drifted"):
        api(transport).resources("recorded")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,api",
    [("sync", AsyncResearchInternSyncRuntimeAPI), ("async", AsyncResearchInternAsyncRuntimeAPI)],
)
async def test_async_transport_uses_same_inventory_contract(kind, api):
    transport = Mock()
    transport.execute = AsyncMock(return_value=payload(kind))
    result = await api(transport).resources("recorded")
    assert result.runtime_id == "recorded" and not result.coverage_complete
    transport.execute.return_value = {**payload(kind), "provider_handle": "must not escape"}
    with pytest.raises(ValueError):
        await api(transport).resources("recorded")
