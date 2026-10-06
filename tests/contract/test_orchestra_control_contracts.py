"""RW-02: real lifecycle transport failures retain backend refusal semantics."""

import ast
import json
import os
from pathlib import Path

import httpx
import pytest
from synth_ai.sdk.research.contracts.run_control import (
    ManagedResearchRunControlError,
    RunLifecycleControlErrorCode,
)
from synth_ai.sdk.research.session.client import ResearchSession


@pytest.mark.parametrize("action", ["stop", "pause", "resume"])
@pytest.mark.parametrize("project_id", [None, "project"])
@pytest.mark.parametrize(
    "code,retryable",
    [
        ("already_terminal", False),
        ("cleanup_in_progress", True),
        ("run_finalizing", True),
    ],
)
def test_control_refusal_through_transport(action, project_id, code, retryable):
    detail = {
        "error_code": code,
        "message": "refused",
        "retryable": retryable,
        "current_state": "stopped",
        "run_id": "run",
    }
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(409, json={"detail": detail})

    with ResearchSession(api_key="offline-dummy", backend_base="http://offline.invalid") as client:
        client._transport.client.close()
        client._transport.client = httpx.Client(
            base_url="http://offline.invalid", transport=httpx.MockTransport(respond)
        )
        with pytest.raises(ManagedResearchRunControlError) as refusal:
            getattr(client, f"{action}_run")("run", project_id=project_id)
    error = refusal.value
    assert error.error_code.value == code
    assert error.retryable is retryable
    assert error.status_code == 409
    assert error.current_state == "stopped" and error.run_id == "run"
    assert error.detail == detail and error.__cause__ is not None
    prefix = f"/smr/projects/{project_id}" if project_id else "/smr"
    assert len(requests) == 1  # retry-later is surfaced, never replayed automatically
    assert requests[0].method == "POST"
    assert requests[0].url.path == f"{prefix}/runs/run/{action}"


def test_refusal_enum_matches_backend_source():
    # Read the producer without booting FastAPI or importing its persistence graph.
    if not os.environ.get("DISCREPANCY_BACKEND_ROOT"):
        fixture = json.loads(
            (Path(__file__).with_name("fixtures") / "run_control_enum.generated.json").read_text()
        )
        assert {code.value for code in RunLifecycleControlErrorCode} == set(fixture["values"])
        return
    root = Path(os.environ["DISCREPANCY_BACKEND_ROOT"])
    source = root / "packages/smr/control/public_api/run_controls.py"
    tree = ast.parse(source.read_text())
    enum = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RunLifecycleControlErrorCode"
    )
    expected = {ast.literal_eval(node.value) for node in enum.body if isinstance(node, ast.Assign)}
    assert {code.value for code in RunLifecycleControlErrorCode} == expected
