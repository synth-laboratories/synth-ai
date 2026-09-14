"""Exact runtime usage reads preserve pending billing and reject identity drift."""

from unittest.mock import AsyncMock, Mock

import pytest

from synth_ai.sdk.research.research_intern import (
    ResearchInternAsyncRuntimeAPI,
    ResearchInternSyncRuntimeAPI,
    AsyncResearchInternAsyncRuntimeAPI,
    AsyncResearchInternSyncRuntimeAPI,
)


def receipt(kind):
    return dict(
        origin_runtime_kind=kind, origin_runtime_id="recorded", org_id="org",
        spend_cents=0, token_count=0, run_count=0,
        billing=dict(billing_state="pending", stalled_row_count=0,
                     stalled_spend_cents=0, billing_failed_row_count=0,
                     show_stalled_banner=False),
    )


@pytest.mark.parametrize("kind,api", [("sync", ResearchInternSyncRuntimeAPI), ("async", ResearchInternAsyncRuntimeAPI)])
def test_sync_transport_usage(kind, api):
    transport = Mock()
    transport.execute.return_value = receipt(kind)
    result = api(transport).usage("recorded")
    assert result.billing.billing_state.value == "pending"
    segment = "sync-sessions" if kind == "sync" else "async-assignments"
    assert transport.execute.call_args.args[0].path == f"/smr/research-intern/{segment}/recorded/usage"
    transport.execute.return_value["origin_runtime_id"] = "replacement"
    with pytest.raises(ValueError, match="identity drifted"):
        api(transport).usage("recorded")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,api", [("sync", AsyncResearchInternSyncRuntimeAPI), ("async", AsyncResearchInternAsyncRuntimeAPI)])
async def test_async_transport_usage(kind, api):
    transport = Mock(execute=AsyncMock(return_value=receipt(kind)))
    result = await api(transport).usage("recorded")
    assert result.billing.billing_state.value == "pending"
    segment = "sync-sessions" if kind == "sync" else "async-assignments"
    assert transport.execute.call_args.args[0].path == f"/smr/research-intern/{segment}/recorded/usage"
    transport.execute.return_value["origin_runtime_kind"] = "async" if kind == "sync" else "sync"
    with pytest.raises(ValueError, match="identity drifted"):
        await api(transport).usage("recorded")
