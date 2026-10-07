"""SYN-4039: preserve the backend's actual settled-usage authority vocabulary."""

import pytest
from synth_ai.sdk.research.contracts.usage import UsageFreshness, UsageSource


@pytest.mark.parametrize(
    "source",
    [
        "admission_settlements",
        "usage_facts_and_admission_settlements",
        "spend_ledger_and_admission_settlements",
    ],
)
def test_settled_authority_roundtrips_without_reclassification__syn4039(source):
    wire = {"source": source, "as_of": None, "record_count": 1, "run_is_terminal": True}
    value = UsageFreshness.from_wire(wire)
    assert value.to_wire() == wire, "SYN-4039: settlement authority was lost or renamed"
    assert value.source.value == source


def test_closed_usage_authorities_match_the_producer__syn4039():
    assert {value.value for value in UsageSource} == {
        "none",
        "usage_facts",
        "spend_ledger",
        "admission_settlements",
        "usage_facts_and_admission_settlements",
        "spend_ledger_and_admission_settlements",
    }, "SYN-4039: valid backend settlement source is absent from the SDK"


def test_unknown_usage_authority_remains_refused__syn4039():
    with pytest.raises(ValueError):
        UsageFreshness.from_wire(
            {
                "source": "invented_authority",
                "as_of": None,
                "record_count": 1,
                "run_is_terminal": True,
            }
        )
