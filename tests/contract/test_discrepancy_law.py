"""Correct contracts for the 2026-10-06 audit; expected failures carry finding IDs."""

import dataclasses
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from synth_ai.sdk.research.contracts.factory_operations import (
    ExperimentBundle,
    ExperimentComparison,
    ExperimentHistory,
)
from synth_ai.sdk.research.contracts.run_control import ManagedResearchRunControlError
from synth_ai.sdk.research.contracts.types import RunResourceBindings
from synth_ai.sdk.research.session.client import ResearchSession

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).with_name("fixtures")
SPEC = json.loads((FIXTURES / "research_openapi.generated.json").read_text())
FULL = json.loads((FIXTURES / "backend_full_openapi.generated.json").read_text())


def normalized(path):
    return re.sub(r"\{[^}]+\}", "{}", path)


@pytest.mark.parametrize(
    "method,path,finding",
    [
        ("GET", "/smr/runs/{run_id}/participants", "MX-01"),
        ("GET", "/smr/runs/{run_id}/artifact-progress", "MX-01"),
        ("GET", "/smr/runs/{run_id}/actor-logs", "MX-01"),
        ("GET", "/smr/projects/{project_id}/runs/{run_id}/participants", "MX-01"),
        ("GET", "/smr/projects/{project_id}/runs/{run_id}/artifact-progress", "MX-01"),
        ("GET", "/smr/projects/{project_id}/runs/{run_id}/actor-logs", "MX-01"),
        ("GET", "/api/tag/v1/scopes/{scope_id}/factory-context", "MX-02"),
        ("GET", "/api/tag/v1/sessions/{session_id}/factory-context", "MX-02"),
    ],
)
def test_sdk_call_resolves_to_backend__MX01_MX02(method, path, finding, tmp_path):
    out = tmp_path / "calls.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("extract_sdk_calls.py")),
            str(ROOT),
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    calls = json.loads(out.read_text())
    active = any(c["method"] == method and normalized(c["path"]) == normalized(path) for c in calls)
    routes = json.loads((FIXTURES / "backend_all_routes.json").read_text())
    exists = any(
        method in r["methods"] and normalized(r["path"]) == normalized(path) for r in routes
    )
    assert not active or exists, f"{finding}: SDK calls nonexistent backend route {method} {path}"


@pytest.mark.parametrize(
    "schema",
    [
        "AsyncRuntimeEnsureRequest",
        "SmrRunResourceBindingsRequest",
        "SmrRunnableProjectCreateRequest",
        "SmrAgentModel",
        "SmrProjectTriggerRequest",
    ],
)
def test_vendored_schema_matches_generated__MX04(schema):
    vendored = json.loads((ROOT / "openapi/research-v1.json").read_text())
    assert vendored["components"]["schemas"][schema] == SPEC["components"]["schemas"][schema], (
        f"MX-04: stale vendored {schema}"
    )


def test_explicit_empty_resource_inventories_round_trip__PR02():
    names = {f.name for f in dataclasses.fields(RunResourceBindings)}
    assert "model_file_ids" in names, "PR-02: typed bindings omit model_file_ids"
    bindings = RunResourceBindings(
        model_file_ids=[], external_repository_ids=[], credential_ref_ids=[]
    )
    assert bindings.to_wire() == {
        "model_file_ids": [],
        "external_repository_ids": [],
        "credential_ref_ids": [],
    }, "PR-02: explicit empty inventories must survive serialization"


def session(monkeypatch, response):
    client = ResearchSession(api_key="offline-dummy", backend_base="http://offline.invalid")
    sent = []

    def request(method, path, **kwargs):
        sent.append((method, path, kwargs))
        return response

    monkeypatch.setattr(client, "_request_json", request)
    return client, sent


def test_file_page_parses__PR01_MX11(monkeypatch):
    client, _ = session(monkeypatch, {"files": [], "next_cursor": None})
    try:
        files = client.list_project_files("p")
    except Exception as error:
        pytest.fail(f"PR-01/MX-11: backend file page rejected: {type(error).__name__}: {error}")
    assert files == [], "PR-01: empty backend page must decode"


def test_file_pagination_cursor_exposed__PR01():
    assert "cursor" in inspect.signature(ResearchSession.list_project_files).parameters, (
        "PR-01: no cursor input to follow next_cursor"
    )


@pytest.mark.parametrize(
    "method,finding,schema,kwargs",
    [
        (
            "workspace_confirm_push",
            "PR-18/MX-07b",
            "WorkspaceConfirmPushRequest",
            {"commit_sha": "a" * 40, "archive_key": "archive"},
        ),
        ("patch_factory_status_compat", "MX-06", "SmrFactoryPatchRequest", {"status": "active"}),
    ],
)
def test_request_body_satisfies_backend__PR18_MX06(monkeypatch, method, finding, schema, kwargs):
    client, sent = session(monkeypatch, {})
    getattr(client, method)("p", **kwargs)
    body = sent[-1][2]["json_body"]
    contract = FULL["components"]["schemas"][schema]
    missing = set(contract.get("required", [])) - body.keys()
    forbidden = (
        body.keys() - contract.get("properties", {}).keys()
        if contract.get("additionalProperties") is False
        else set()
    )
    assert not missing and not forbidden, (
        f"{finding}: missing={sorted(missing)}, forbidden={sorted(forbidden)}"
    )


def test_comparison_retains_integrity_status_and_findings__RR02():
    wire = {
        "schema_version": "smr_experiment_comparison.v1",
        "project_id": "p",
        "experiment_ids": ["e1", "e2"],
        "comparable": False,
        "status": "integrity_failed",
        "findings": [{"code": "scorer_mismatch"}],
    }
    comparison = ExperimentComparison.from_wire(wire)
    assert getattr(comparison, "status", None) == wire["status"], (
        "RR-02: integrity_failed status lost"
    )
    assert list(comparison.findings) == wire["findings"], "RR-02: findings lost"


@pytest.mark.parametrize(
    "model,field,finding",
    [
        (ExperimentBundle, "trials", "RR-03"),
        (ExperimentBundle, "artifact_index", "RR-03"),
        (ExperimentHistory, "raw", "RR-03"),
    ],
)
def test_research_evidence_has_typed_fields__RR03(model, field, finding):
    assert field in {f.name for f in dataclasses.fields(model)}, (
        f"{finding}: {model.__name__}.{field} missing"
    )


@pytest.mark.parametrize(
    "code,retryable",
    [("already_terminal", False), ("cleanup_in_progress", True), ("run_finalizing", True)],
)
def test_run_control_refusal_is_typed__RW02(code, retryable):
    payload = {
        "detail": {
            "error_code": code,
            "message": "refused",
            "retryable": retryable,
            "current_state": "stopped",
            "run_id": "r",
        }
    }
    try:
        error = ManagedResearchRunControlError.from_response(
            payload=payload, status_code=409, response_text=json.dumps(payload)
        )
    except ValueError as failure:
        pytest.fail(f"RW-02: backend refusal crashes parser: {failure}")
    assert error.error_code.value == code and error.retryable is retryable, (
        "RW-02: refusal semantics lost"
    )


@pytest.mark.parametrize(
    "path",
    [
        "/smr/projects/{project_id}/research/records/{record_id}",
        "/smr/projects/{project_id}/research/records/{record_id}/citations",
        "/smr/projects/{project_id}/experiments/{experiment_id}/bundle",
        "/smr/projects/{project_id}/experiment-bundles",
    ],
)
def test_scientific_read_has_bounded_response_contract__RR06_RR07(path):
    assert path in SPEC["paths"], (
        f"RR-06/RR-07: scientific read missing from public contract: {path}"
    )
    schema = SPEC["paths"][path]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema, f"RR-07: untyped response for {path}"


@pytest.mark.parametrize(
    "code,expected_retryable",
    [
        ("scientific_writer_transferred", False),
        ("transfer_fence_active", True),
        ("provenance_unbound", False),
        ("run_deployment_pins_missing", False),
    ],
)
def test_authority_errors_are_first_class__RW08(code, expected_retryable):
    import httpx
    from synth_ai.sdk.research.errors import ResearchApiError, ResearchStructuredDenialError
    from synth_ai.sdk.research.transport.http import _raise_for_error_response

    response = httpx.Response(
        409 if code in {"scientific_writer_transferred", "transfer_fence_active"} else 422,
        request=httpx.Request("POST", "http://offline.invalid/smr/projects/p/trigger"),
        json={
            "detail": {
                "error_code": code,
                "message": "launch refused",
                "operation_id": "intent",
                "retry_after": "fence_release" if expected_retryable else None,
            }
        },
    )
    with pytest.raises(ResearchApiError) as refused:
        _raise_for_error_response(response, operation_id="intent")
    assert type(refused.value) not in {ResearchStructuredDenialError, ResearchApiError}, (
        f"RW-08: {code} degrades to generic denial"
    )
    assert refused.value.retryable is expected_retryable, f"RW-08: {code} retryability lost"


def test_uncertain_write_preserves_operation_identity__RR14():
    import httpx
    from synth_ai.sdk.research.errors import ResearchApiError, ResearchStructuredDenialError
    from synth_ai.sdk.research.transport.http import _raise_for_error_response

    response = httpx.Response(
        503,
        request=httpx.Request("POST", "http://offline.invalid/smr/projects/p/research/operations"),
        json={
            "detail": {
                "error_code": "outcome_uncertain",
                "message": "receipt lookup required",
                "operation_id": "original",
            }
        },
    )
    with pytest.raises(ResearchApiError) as refused:
        _raise_for_error_response(response, operation_id="original")
    assert type(refused.value) not in {ResearchStructuredDenialError, ResearchApiError}, (
        "RR-14: uncertain committed effect indistinguishable from deterministic denial"
    )
    assert getattr(refused.value, "operation_id", None) == "original", (
        "RR-14: uncertain outcome drops original operation id"
    )


def test_visual_backend_logical_response_parses__MX08():
    from synth_ai.sdk.research.contracts.visuals import Visual

    wire = {
        "visual_id": "v",
        "project_id": "p",
        "org_id": "o",
        "title": "visual",
        "lifecycle": "draft",
        "current_revision": None,
        "revisions": [],
        "releases": [],
        "inaccessible_sources": [],
        "created_at": "2026-10-06T00:00:00Z",
        "updated_at": "2026-10-06T00:00:00Z",
    }
    # The contract's logical response has no artifact fields; no fictitious fields may be added.
    try:
        visual = Visual.from_wire(wire)
    except (ValueError, TypeError) as failure:
        pytest.fail(f"MX-08: backend logical visual response rejected: {failure}")
    assert str(visual.visual_id) == "v", "MX-08: visual identity changed"


def test_citations_client_available__RR06():
    from synth_ai.sdk.research.scientific_records import ScientificRecordsAPI

    assert callable(getattr(ScientificRecordsAPI, "citations", None)), (
        "RR-06: no SDK method for citation verification"
    )


def test_execution_operations_client_available__RR05():
    from synth_ai.sdk.research.scientific_records import ScientificRecordsAPI

    assert callable(getattr(ScientificRecordsAPI, "execution_write", None)), (
        "RR-05: admitted actor writes have no SDK execution-operations client"
    )


@pytest.mark.parametrize("failure_kind", ["timeout", "connection"])
def test_transport_failure_keeps_original_intent_and_cause__RR14(failure_kind):
    import httpx
    from synth_ai.sdk.research.errors import ResearchApiError
    from synth_ai.sdk.research.transport.http import _raise_for_transport_exception

    cause = (
        httpx.ReadTimeout("no response")
        if failure_kind == "timeout"
        else httpx.ConnectError("connection lost")
    )
    with pytest.raises(ResearchApiError) as refused:
        _raise_for_transport_exception(
            "POST", "/smr/projects/p/research/operations", cause, operation_id="original"
        )
    assert refused.value.__cause__ is cause, "RR-14: transport cause chain lost"
    assert getattr(refused.value, "operation_id", None) == "original", (
        "RR-14: uncertain write drops original operation identity"
    )


@pytest.mark.parametrize("field", ["resource_readiness", "runtime_readiness", "checks"])
def test_preflight_keeps_typed_readiness__PR08(field):
    from synth_ai.sdk.research.contracts.types import SmrLaunchPreflight

    assert field in {member.name for member in dataclasses.fields(SmrLaunchPreflight)}, (
        f"PR-08: typed preflight loses {field}"
    )


def test_native_result_identity_is_typed__RR04():
    from typing import get_type_hints

    annotation = get_type_hints(ExperimentBundle)["evaluations"]
    assert "dict[str, object]" not in str(annotation), (
        "RR-04: experiment_run_id and scorer identity live only in untyped evaluation dictionaries"
    )


def test_all_raw_sdk_calls_resolve_to_backend__MX09(tmp_path):
    from synth_ai.sdk.research.operations import RESEARCH_OPERATIONS

    out = tmp_path / "all_calls.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("extract_sdk_calls.py")),
            str(ROOT),
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    calls = json.loads(out.read_text())
    routes = json.loads((FIXTURES / "backend_all_routes.json").read_text())

    def pattern(path):
        return re.compile("^" + re.sub(r"\\\{[^}]*\\\}", "[^/]+", re.escape(path)) + "$")

    unresolved = []
    for call in calls:
        if call["file"].endswith("operations.py"):
            continue
        method = call["method"]
        if method is None:
            operation = RESEARCH_OPERATIONS.get(call["operation_id"])
            if operation is not None:
                method = operation.method.value
        if method is None:
            continue  # constructors and helpers are not identifiable HTTP calls
        path = call["path"]
        candidates = [
            route
            for route in routes
            if normalized(route["path"]) == normalized(path)
            or pattern(route["path"]).match(path)
            or pattern(path).match(route["path"])
        ]
        if not any(method in route["methods"] for route in candidates):
            unresolved.append(f"{method} {path} ({call['file']}:{call['line']})")
    assert not unresolved, (
        "MX-09/MX-01/MX-02: SDK routes outside bounded registry must resolve:\n"
        + "\n".join(sorted(set(unresolved)))
    )


@pytest.mark.parametrize(
    "name",
    [
        "ManagedResearchRunState",
        "ManagedResearchRunTerminalOutcome",
        "ManagedResearchRunLivenessPhase",
    ],
)
def test_sdk_public_state_vocabulary_matches_backend__RW17(name):
    from synth_ai.sdk.research.contracts import run_state

    expected = json.loads((FIXTURES / "public_enums.generated.json").read_text())[name]
    actual = sorted(item.value for item in getattr(run_state, name))
    assert actual == expected, f"RW-17: SDK advertises {name} values backend does not emit"


def test_full_vendored_snapshot_matches_backend__MX04():
    def without_generator_noise(specification):
        specification = json.loads(json.dumps(specification))
        schemas = specification["components"]["schemas"]
        for name in list(schemas):
            if name.startswith("Body_publish_") and name.endswith("_visual"):
                del schemas[name]
        for name in ("ValidationError",):
            schema = schemas.get(name, {})
            for field in ("ctx", "input"):
                schema.get("properties", {}).pop(field, None)
        return specification

    vendored = json.loads((ROOT / "openapi/research-v1.json").read_text())
    assert without_generator_noise(vendored) == without_generator_noise(SPEC), (
        "MX-04: SDK snapshot differs from authoritative generated backend surface"
    )
