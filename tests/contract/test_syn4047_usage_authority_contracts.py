"""SYN-4047: preserve the backend's actual settled-usage authority vocabulary."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from synth_ai.sdk.research.contracts.usage import UsageFreshness, UsageSource
from synth_ai.sdk.research.swarms import SwarmsAPI


def test_actual_settled_usage_survives_public_swarm_wrapper__syn4047():
    wire = json.loads(
        (Path(__file__).parents[1] / "fixtures/native_usage/settled_usage_summary.json").read_text()
    )
    transport = Mock()
    transport.execute.return_value = wire
    usage = SwarmsAPI(transport).usage(wire["run_id"])
    request = transport.execute.call_args.args[0]
    assert request.path == f"/smr/runs/{wire['run_id']}/usage-summary"
    assert usage.swarm_id == wire["run_id"]
    assert usage.project_id == wire["project_id"]
    assert usage.money.to_wire() == wire["money"]
    assert usage.tokens.to_wire() == wire["tokens"]
    assert [actor.to_wire() for actor in usage.actors] == wire["actors"]
    assert usage.freshness.source.value == "admission_settlements"
    assert usage.freshness.record_count == wire["freshness"]["record_count"]
    assert usage.freshness.run_is_terminal is True


@pytest.mark.parametrize(
    "source",
    [
        "admission_settlements",
        "usage_facts_and_admission_settlements",
        "spend_ledger_and_admission_settlements",
    ],
)
def test_settled_authority_roundtrips_without_reclassification__syn4047(source):
    wire = {"source": source, "as_of": None, "record_count": 1, "run_is_terminal": True}
    value = UsageFreshness.from_wire(wire)
    assert value.to_wire() == wire, "SYN-4047: settlement authority was lost or renamed"
    assert value.source.value == source


def test_closed_usage_authorities_match_the_producer__syn4047():
    assert {value.value for value in UsageSource} == {
        "none",
        "usage_facts",
        "spend_ledger",
        "admission_settlements",
        "usage_facts_and_admission_settlements",
        "spend_ledger_and_admission_settlements",
    }, "SYN-4047: valid backend settlement source is absent from the SDK"


def test_unknown_usage_authority_remains_refused__syn4047():
    with pytest.raises(ValueError):
        UsageFreshness.from_wire(
            {
                "source": "invented_authority",
                "as_of": None,
                "record_count": 1,
                "run_is_terminal": True,
            }
        )
