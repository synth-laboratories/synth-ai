"""Promoted RW-02 / MX-01 / MX-02 / MX-09 laws, with original assertions intact."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from synth_ai.sdk.research.contracts.run_control import ManagedResearchRunControlError

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).with_name("fixtures")


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
    routes = json.loads(
        (
            Path(os.environ.get("DISCREPANCY_BACKEND_ROUTES", FIXTURES / "backend_all_routes.json"))
        ).read_text()
    )
    exists = any(
        method in r["methods"] and normalized(r["path"]) == normalized(path) for r in routes
    )
    assert not active or exists, f"{finding}: SDK calls nonexistent backend route {method} {path}"


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
    routes = json.loads(
        (
            Path(os.environ.get("DISCREPANCY_BACKEND_ROUTES", FIXTURES / "backend_all_routes.json"))
        ).read_text()
    )

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
    "path,operation_id",
    [
        ("/smr/runs/{run_id}/tasks", "list_run_tasks"),
        ("/smr/projects/{project_id}/runs/{run_id}/task-events", "list_project_run_task_events"),
    ],
)
def test_task_reads_in_producer_public_contract(path, operation_id):
    from synth_ai.sdk.research.operations import RESEARCH_OPERATIONS

    fixture = Path(
        os.environ.get("DISCREPANCY_BACKEND_OPENAPI", FIXTURES / "research_openapi.generated.json")
    )
    spec = json.loads(fixture.read_text())
    assert spec["paths"][path]["get"]["operationId"] == operation_id
    assert RESEARCH_OPERATIONS[operation_id].path_template == path
