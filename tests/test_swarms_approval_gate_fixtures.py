"""SYN-4650 W4: the SDK decodes the backend's frozen ``kind="gate"`` approval bytes."""

import hashlib
import json
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError
from synth_ai.sdk.swarms.types.approvals import (
    GATE_REVISION,
    GATE_SCHEMA_VERSION,
    Approval,
    ApprovalKind,
    ApprovalList,
    GateApproval,
)

pytestmark = pytest.mark.unit
PACKAGE = Path(__file__).parent / "fixtures" / "swarms-v1-approval-gate"
MANIFEST = json.loads((PACKAGE / "manifest.json").read_text())
METADATA = {
    "contract": "swarms.v1",
    "request_id": "gate-fixture",
    "operation_id": None,
    "observed_at": "2026-10-29T12:00:00Z",
}
# The SDK decoder is forward-compatible (ResponseBase ignores unknown keys), so a
# gate carrying a stray wait field still decodes as a gate; the backend's
# producer model is the side that refuses it (extra_forbidden).
SDK_IGNORES_EXTRA = {"gate.wait-field.refused.json"}


def test_vendored_bytes_match_the_backend_manifest():
    assert MANIFEST["schema_version"] == GATE_SCHEMA_VERSION
    assert set(MANIFEST["fixtures"]) == {path.name for path in PACKAGE.glob("gate.*.json")}
    for name, entry in MANIFEST["fixtures"].items():
        digest = "sha256:" + hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest()
        assert digest == entry["sha256"], name


@pytest.mark.parametrize("name", sorted(MANIFEST["fixtures"]))
def test_gate_bytes_decode_with_declared_outcome(name):
    entry = MANIFEST["fixtures"][name]
    raw = (PACKAGE / name).read_text()
    if entry["expect"] == "accepted" or name in SDK_IGNORES_EXTRA:
        parsed = GateApproval.model_validate_json(raw)
        assert (parsed.kind, parsed.revision, parsed.state) == ("gate", GATE_REVISION, "awaiting")
        assert parsed.decidable is True
        return
    with pytest.raises(ValidationError) as refused:
        GateApproval.model_validate_json(raw)
    assert entry["refusal"] in str(refused.value)


def test_list_discriminates_gate_items_beside_tool_calls():
    gate = json.loads((PACKAGE / "gate.release.accepted.json").read_text())
    tool_call = {
        "metadata": METADATA,
        "approval_id": "approval-1",
        "run_id": gate["run_id"],
        "revision": 3,
        "state": "awaiting",
        "kind": "tool_call",
        "question": "Allow tool git_push?",
        "tool": "git_push",
        "capability": "git.push.remote",
        "argument_digest": None,
        "actor_id": None,
        "attempt_id": None,
        "parked_at": "2026-10-29T11:59:00Z",
        "deadline_at": None,
        "on_timeout": None,
        "decidable": True,
    }
    listing = {"metadata": METADATA, "run_id": gate["run_id"], "items": [tool_call, gate]}
    parsed = ApprovalList.model_validate_json(json.dumps(listing))
    assert [type(item) for item in parsed.items] == [Approval, GateApproval]
    assert parsed.items[1].gate_kind == "release"
    assert parsed.items[1].target_task_id == gate["target_task_id"]


def test_gate_kind_is_public_and_call_items_cannot_claim_it():
    assert "gate" in get_args(ApprovalKind)
    with pytest.raises(ValidationError):
        Approval.model_validate(
            {
                **json.loads((PACKAGE / "gate.release.accepted.json").read_text()),
                "metadata": METADATA,
            }
        )


def test_unknown_item_kind_is_refused_by_the_list():
    gate = json.loads((PACKAGE / "gate.release.accepted.json").read_text())
    listing = {"metadata": METADATA, "run_id": gate["run_id"], "items": [{**gate, "kind": "vote"}]}
    with pytest.raises(ValidationError):
        ApprovalList.model_validate_json(json.dumps(listing))
