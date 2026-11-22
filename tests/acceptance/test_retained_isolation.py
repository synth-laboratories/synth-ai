"""Offline controls for retained cross-org evidence without tenant credentials."""

import pytest

from scripts.acceptance.retained_isolation import inspect_cell


def pair(status, code, result):
    refused = {
        "accepted": False,
        "status": status,
        "code": code,
        "cause": "",
        "typed": True,
        "operation": "getRun",
    }
    return {
        "operation": "getRun",
        "result": result,
        "foreign": refused,
        "absent": dict(refused),
        "owner_org_id": "owned-a",
        "foreign_org_id": "owned-b",
        "owner_user_id": "principal-a",
        "foreign_user_id": "principal-b",
    }


def test_equal_typed_resource_404_preserves_reported_pass():
    assert inspect_cell(pair(404, "run_not_found", "pass")) == "pass"


def test_uniform_403_can_only_remain_unproven():
    assert (
        inspect_cell(pair(403, "insufficient_scope", "unproven:uniform_403"))
        == "unproven:uniform_403"
    )
    with pytest.raises(ValueError, match="retained_pass_unsubstantiated"):
        inspect_cell(pair(403, "insufficient_scope", "pass"))


@pytest.mark.parametrize(
    "change", ["same_user", "different_problem", "wrong_operation", "accepted", "untyped"]
)
def test_corrupt_pass_is_refused(change):
    row = pair(404, "run_not_found", "pass")
    if change == "same_user":
        row["foreign_user_id"] = row["owner_user_id"]
    elif change == "different_problem":
        row["absent"]["code"] = "file_not_found"
    elif change == "wrong_operation":
        row["foreign"]["operation"] = "getFile"
    elif change == "accepted":
        row["foreign"]["accepted"] = True
    else:
        row["absent"]["typed"] = False
    with pytest.raises(ValueError, match="retained_pass_unsubstantiated"):
        inspect_cell(row)


def test_unexercised_cell_stays_visible():
    assert (
        inspect_cell({"operation": "getUpload", "result": "uncovered:surface_unmapped"})
        == "uncovered:surface_unmapped"
    )


def test_gate_keeps_all_driver_identities_and_rejects_modified_evidence(tmp_path):
    import hashlib
    import json

    from scripts.acceptance.retained_isolation import inspect_gate

    artifacts = []
    for driver in ("sdk", "cli", "mcp_stdio"):
        directory = tmp_path / driver
        directory.mkdir()
        receipt = directory / "driver-receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "driver": driver,
                    "steps": [
                        {
                            "step": "cross_org_not_found_matrix",
                            "evidence": [pair(403, "insufficient_scope", "unproven:uniform_403")],
                        }
                    ],
                }
            )
        )
        artifacts.append(
            {"path": str(receipt), "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}
        )
    manifest = "a" * 64
    gate = tmp_path / "receipt.json"
    gate.write_text(
        json.dumps(
            {
                "check": "cross_org_isolation",
                "cohort": {"manifest_sha256": manifest},
                "result": {"verdict": "Inconclusive"},
                "artifacts": artifacts,
            }
        )
    )
    result = inspect_gate(gate, manifest)
    assert result["drivers"]["sdk"]["counts"] == {"unproven:uniform_403": 1}
    with pytest.raises(ValueError, match="wrong_retained_gate_or_cohort"):
        inspect_gate(gate, "b" * 64)
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="retained_driver_digest_mismatch"):
        inspect_gate(gate, manifest)
