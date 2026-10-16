"""Regressions already green on current backend dev. See openapi_contract.py."""

import json
from pathlib import Path

SPEC = json.loads(
    (Path(__file__).with_name("fixtures") / "research_openapi.generated.json").read_text()
)


def test_registry_matches_backend__MX03():
    from synth_ai.sdk.research.operations import RESEARCH_OPERATIONS

    public = {
        op["operationId"]
        for item in SPEC["paths"].values()
        for op in item.values()
        if isinstance(op, dict) and "operationId" in op
    }
    # Separate full-contract dataset operations have their own producer authority.
    extra = set(RESEARCH_OPERATIONS) - public
    assert extra <= {"get_dataset_revision", "get_dataset_revision_manifest"}, (
        f"MX-03: SDK-only registry operations {sorted(extra)}"
    )
