"""Behavioral acceptance for file identity, evidence and admitted scientific writes.

SYN-3990/3992/3994/3995: preserve identity and null outcomes across real public
client/codec boundaries. All transports are in-process; no provider or sockets.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from synth_ai.mcp.research.server import _mcp_jsonable
from synth_ai.sdk.research.contracts.factory_operations import ExperimentBundle, ExperimentHistory
from synth_ai.sdk.research.contracts.forge.contracts import ExactReference, contract_digest
from synth_ai.sdk.research.contracts.forge.operations import PublicWrite
from synth_ai.sdk.research.contracts.forge.records import Measurement, Result
from synth_ai.sdk.research.scientific_records import ExecutionWriteRequest, ScientificRecordsAPI
from synth_ai.sdk.research.session.client import ResearchSession

ORG = "00000000-0000-4000-8000-00000000000a"
PROJECT = "00000000-0000-4000-8000-00000000000b"


def test_file_pages_follow_exact_cursor_without_losing_files__SYN3990():
    client = ResearchSession(api_key="offline-dummy", backend_base="http://offline.invalid")
    token = "  opaque/token==  "
    pages = [
        {"files": [{"file_id": "one"}], "next_cursor": token},
        {"files": [{"file_id": "two"}], "next_cursor": None},
    ]
    with patch.object(client, "_request_json", side_effect=pages) as request:
        assert [row["file_id"] for row in client.list_project_files(PROJECT)] == ["one", "two"]
        assert request.call_args_list[1].kwargs["params"]["cursor"] == token


@pytest.mark.parametrize(
    "page",
    [
        {"files": {}, "next_cursor": None},
        {"files": [], "next_cursor": 123},
        {"files": [], "next_cursor": ""},
    ],
)
def test_invalid_file_page_refuses_without_empty_success__SYN3990(page):
    client = ResearchSession(api_key="offline-dummy", backend_base="http://offline.invalid")
    with patch.object(client, "_request_json", return_value=page):
        with pytest.raises((ValueError, RuntimeError)):
            client.list_project_files(PROJECT)


@pytest.mark.parametrize(
    "outcome,value", [("null", None), ("negative", -0.5), ("inconclusive", None)]
)
def test_execution_write_preserves_null_and_scorer_intent__SYN3992(outcome, value):
    reference = ExactReference(
        authority="forge", kind="trial", record_id="trial", revision="1", digest_sha256="a" * 64
    )
    scorer = ExactReference(
        authority="orchestra",
        kind="scorer",
        record_id="scorer-v1",
        revision="1",
        digest_sha256="b" * 64,
    )
    result = Result(
        trial=reference,
        outcome=outcome,
        evaluator=scorer,
        measurements=(Measurement(name="accuracy", value=value, unit="fraction"),),
        missing_evidence=("numeric measurement unavailable",) if value is None else (),
    )
    values = {
        "operation_id": "original",
        "record_id": "result",
        "expected_revision": 0,
        "payload": result.model_dump(mode="json"),
    }
    write = PublicWrite(**values, request_digest_sha256=contract_digest(values))
    submitted = ExecutionWriteRequest(
        organization_id=ORG, admission_id="adm_" + "0" * 26, write=write
    )
    receipt = {
        "schema_version": "forge.receipt.v1",
        "scope": {"organization_id": ORG, "project_id": PROJECT},
        "operation_id": "original",
        "payload_digest_sha256": "c" * 64,
        "reference": {
            "authority": "forge",
            "kind": "result",
            "record_id": "result",
            "revision": "1",
            "digest_sha256": contract_digest(result),
        },
        "cursor": 1,
        "recorded_at": "2026-10-06T00:00:00Z",
    }
    captured = []

    def request(method, path, **kwargs):
        captured.append((method, path, kwargs))
        return receipt

    answer = ScientificRecordsAPI(SimpleNamespace(request_json=request)).execution_write(
        PROJECT, submitted
    )
    assert answer.operation_id == "original"
    method, path, kwargs = captured[0]
    assert method == "POST" and path.endswith("/research/execution-operations")
    assert kwargs["json_body"]["write"]["payload"]["outcome"] == outcome
    assert kwargs["json_body"]["write"]["payload"]["measurements"][0]["value"] == value
    assert kwargs["json_body"]["write"]["payload"]["evaluator"] == scorer.model_dump(mode="json")
    assert "producer" not in kwargs["json_body"]["write"]
    assert kwargs["operation_id"] == "original"


def test_bundle_typed_identity_and_history_raw_do_not_duplicate_mcp__SYN3994():
    wire = {
        "schema_version": "smr_experiment_bundle.v1",
        "experiment_id": "e",
        "project_id": PROJECT,
        "evaluations": [
            {
                "result_id": "r",
                "experiment_run_id": "trial",
                "scorer_id": "scorer",
                "scorer_version": "v1",
                "scorer_config_digest": "a" * 64,
                "value": None,
                "outcome": "null",
            }
        ],
        "trials": [
            {
                "experiment_run_id": "trial",
                "experiment_id": "e",
                "experiment_revision": 1,
                "run_id": "run",
                "role": "primary",
                "metadata": {},
                "created_at": "2026-10-06T00:00:00Z",
            }
        ],
        "artifact_index": [{"artifact_id": "artifact"}],
        "extension": {"retained": True},
    }
    bundle = ExperimentBundle.from_wire(wire)
    assert bundle.evaluations[0].experiment_run_id == bundle.trials[0].experiment_run_id
    assert bundle.evaluations[0].scorer_id == "scorer"
    history_wire = {
        "schema_version": "smr_experiment_history.v1",
        "project_id": PROJECT,
        "bundles": [wire],
    }
    history = ExperimentHistory.from_wire(history_wire)
    assert history.raw == history_wire
    public = _mcp_jsonable(history)
    assert "raw" not in public and "raw" not in public["bundles"][0]
    assert public["bundles"][0]["extension"] == {"retained": True}
    assert public["bundles"][0]["evaluations"][0]["value"] is None
    json.dumps(public)
