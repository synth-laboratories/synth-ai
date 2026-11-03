"""SYN-4849: independent execution/publication/settlement/cleanup facets, offline."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from synth_ai import SynthClient
from synth_ai.mcp.catalog import run_status
from synth_ai.mcp.contracts import RunIdentity
from synth_ai.sdk.swarms import RunStatusFacets
from synth_ai.sdk.swarms.types.runs import RunSnapshot

pytestmark = pytest.mark.unit

METADATA = {
    "contract": "swarms.v1",
    "request_id": "facets-fixture",
    "operation_id": None,
    "observed_at": "2026-10-29T12:00:00Z",
}


def _snapshot(facets: dict | None, **overrides) -> dict:
    value = {
        "metadata": METADATA,
        "run_id": "run-1",
        "project_id": "p",
        "state": "failed",
        "model": "gpt-6-luna",
        "provider": "openai",
        "result_available": False,
        "execution_closed": False,
        "accounting_final": False,
        "cleanup_complete": False,
        **overrides,
    }
    if facets is not None:
        value["status_facets"] = facets
    return value


REFUSED = {
    "execution": {
        "state": "never_started",
        "outcome": "failed",
        "cause": {
            "code": "program_family_context_unbound",
            "authority": "orchestra.program_create",
        },
        "closed_at": "2026-10-29T12:00:01Z",
    },
    "publication": {
        "state": "unknown",
        "cause": {"code": "obligations_not_observed", "authority": "not_observed"},
    },
    "settlement": {
        "state": "final",
        "cause": {
            "code": "backend_owner_absent_closure",
            "authority": "backend.owner_absent_closure",
        },
        "customer_charge_final": True,
        "supplier_settlement_final": True,
    },
    "cleanup": {
        "state": "complete",
        "cause": {
            "code": "orchestra_program_create_refused",
            "authority": "backend.never_launched_stop",
        },
        "completed_at": "2026-10-29T12:00:01Z",
    },
}


def test_facets_decode_independently_of_the_top_level_state():
    snapshot = RunSnapshot.model_validate_json(json.dumps(_snapshot(REFUSED)))
    facets = snapshot.status_facets
    assert isinstance(facets, RunStatusFacets)
    # The top-level fields keep their meaning; the facets add the split.
    assert snapshot.state == "failed" and snapshot.execution_closed is False
    assert facets.execution.state == "never_started"
    assert facets.execution.cause.code == "program_family_context_unbound"
    assert facets.execution.started_at is None
    assert facets.settlement.state == "final"
    assert facets.cleanup.completed_at.isoformat() == "2026-10-29T12:00:01+00:00"


def test_absent_facets_serialize_to_the_unchanged_bytes():
    bare = RunSnapshot.model_validate_json(json.dumps(_snapshot(None)))
    assert bare.status_facets is None
    assert "status_facets" not in bare.model_dump(mode="json")


def test_absent_facet_fields_are_omitted_never_null():
    snapshot = RunSnapshot.model_validate_json(json.dumps(_snapshot(REFUSED)))
    wire = snapshot.model_dump(mode="json")["status_facets"]
    assert "started_at" not in wire["execution"]
    assert "counts" not in wire["publication"]
    assert "operation_id" not in wire["settlement"]
    assert wire == REFUSED | {
        "execution": REFUSED["execution"] | {"closed_at": "2026-10-29T12:00:01Z"},
        "cleanup": REFUSED["cleanup"] | {"completed_at": "2026-10-29T12:00:01Z"},
    }


def test_future_facet_states_are_preserved_not_rejected():
    future = json.loads(json.dumps(REFUSED))
    future["publication"] = {
        "state": "embargoed",
        "cause": {"code": "embargo", "authority": "forge.publication"},
        "counts": {"open": 0, "closed": 1, "waived": 0},
    }
    snapshot = RunSnapshot.model_validate_json(json.dumps(_snapshot(future)))
    assert snapshot.status_facets.publication.state == "embargoed"
    assert snapshot.status_facets.publication.counts.closed == 1


def test_sdk_status_and_mcp_readback_carry_the_facets():
    def respond(request):
        return httpx.Response(
            200, headers={"X-Synth-Contract": "swarms.v1"}, json=_snapshot(REFUSED)
        )

    client = SynthClient(api_key="fixture-key", base_url="https://backend.test")
    client._transport.client.close()
    client._transport.client = httpx.Client(transport=httpx.MockTransport(respond))
    client._transport._negotiate = lambda *args: None
    with client:
        status = client.swarms.get("run-1").status()
    assert status.status_facets.execution.state == "never_started"

    class Run:
        async def status(self):
            return RunSnapshot.model_validate_json(json.dumps(_snapshot(REFUSED)))

    async def get(run_id):
        assert run_id == "run-1"
        return Run()

    mcp = asyncio.run(
        run_status(
            SimpleNamespace(swarms=SimpleNamespace(get=get)),
            RunIdentity.model_validate({"run_id": "run-1"}),
        )
    )
    assert (
        mcp.model_dump(mode="json")["status_facets"]
        == status.model_dump(mode="json")["status_facets"]
    )
