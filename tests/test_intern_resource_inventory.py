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


def complete_payload(kind):
    """A covered runtime: complete coverage, still-pending cleanup and epochs."""
    return {
        **payload(kind),
        "coverage_complete": True,
        "incomplete_reasons": [],
        "creator_set": "intern-local-docker-creators-v1",
        "profile": "local_docker",
        "disposition_complete": False,
        "resources": [
            dict(
                resource_kind="intern_runtime",
                resource_id="recorded",
                relation="self",
                disposition="retained",
                reason="runtime_active",
                cleanup_owner="research_intern:intern-1",
            ),
            dict(
                resource_kind="container_lease",
                resource_id="lease-0",
                relation="owned",
                disposition="pending",
                reason="resource_not_settled",
                cleanup_owner=f"intern_runtime:{kind}:recorded:epoch:0",
                epoch=0,
            ),
        ],
        "epochs": [
            dict(
                epoch=0,
                admission="closed",
                closure_kind="superseded_by_reopen",
                opened_at="2026-09-12T00:00:00Z",
                closed_at="2026-09-12T01:00:00Z",
                effects=[
                    dict(
                        resource_kind="container_lease",
                        resource_id="lease-0",
                        effect="stop_requested",
                        owner=f"intern_runtime:{kind}:recorded:epoch:0",
                        reason="runtime_epoch_closed",
                    )
                ],
            ),
            dict(epoch=1, admission="open", opened_at="2026-09-12T01:00:00Z"),
        ],
    }


@pytest.mark.parametrize(
    "kind,api",
    [("sync", ResearchInternSyncRuntimeAPI), ("async", ResearchInternAsyncRuntimeAPI)],
)
def test_complete_coverage_keeps_retained_and_pending_distinct(kind, api):
    transport = Mock()
    transport.execute.return_value = complete_payload(kind)
    result = api(transport).resources("recorded")
    assert result.coverage_complete and not result.disposition_complete
    assert [item.disposition for item in result.resources] == ["retained", "pending"]
    assert result.resources[1].epoch == 0
    assert [epoch.admission for epoch in result.epochs] == ["closed", "open"]
    assert result.epochs[0].effects[0].effect == "stop_requested"
    # Cleanup can never be complete on incomplete coverage.
    transport.execute.return_value = {
        **payload(kind), "disposition_complete": True,
    }
    with pytest.raises(ValueError, match="cleanup without coverage"):
        api(transport).resources("recorded")
    for bad in (
        {"disposition": "deleted"},
        {"epoch": -1},
    ):
        broken = complete_payload(kind)
        broken["resources"][1] = {**broken["resources"][1], **bad}
        transport.execute.return_value = broken
        with pytest.raises(ValueError):
            api(transport).resources("recorded")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,api",
    [("sync", AsyncResearchInternSyncRuntimeAPI), ("async", AsyncResearchInternAsyncRuntimeAPI)],
)
async def test_async_transport_decodes_epochs_and_effects(kind, api):
    transport = Mock()
    transport.execute = AsyncMock(return_value=complete_payload(kind))
    result = await api(transport).resources("recorded")
    assert result.creator_set == "intern-local-docker-creators-v1"
    assert result.epochs[0].closure_kind == "superseded_by_reopen"
    assert result.resources[0].cleanup_owner == "research_intern:intern-1"


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


# Lockstep with the backend-authored bounded contract. These mirrors live in
# contracts/intern_resources.py, outside the research_intern pairing table.
INVENTORY_MIRRORS = {
    "InternResourceInventory": "InternResourceInventory",
    "InternResourceDisposition": "InternResourceDisposition",
    "InternRuntimeResourceEpoch": "InternRuntimeResourceEpoch",
    "InternResourceStopEffect": "InternResourceStopEffect",
}


@pytest.mark.parametrize(("mirror_name", "schema_name"), sorted(INVENTORY_MIRRORS.items()))
def test_inventory_mirrors_match_backend_contract(mirror_name, schema_name):
    import json
    from pathlib import Path

    from synth_ai.sdk.research.contracts import intern_resources

    contract = Path(__file__).resolve().parents[1] / "openapi" / "research-v1.json"
    schema = json.loads(contract.read_text(encoding="utf-8"))["components"]["schemas"][
        schema_name
    ]
    mirror = getattr(intern_resources, mirror_name)
    assert set(mirror.model_fields) == set(schema["properties"])
    required = {name for name, field in mirror.model_fields.items() if field.is_required()}
    assert not set(schema.get("required", ())) - required
