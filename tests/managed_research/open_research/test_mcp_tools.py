"""MCP-level tests for the ``open_research_*`` tools."""

from __future__ import annotations

import pytest
from synth_ai.managed_research.mcp.server import ManagedResearchMcpServer, RpcError
from synth_ai.managed_research.open_research.client import OpenResearchClient

from .conftest import (
    RecordedRequest,
    ScriptedResponse,
    patch_client_transport,
)

EXPECTED_TOOLS = {
    "open_research_list_projects",
    "open_research_get_project",
    "open_research_list_queues",
    "open_research_submit_question",
    "open_research_get_submission",
    "open_research_list_experiments",
    "open_research_get_experiment",
    "open_research_get_receipt",
    "open_research_download_bundle",
}


def test_all_nine_open_research_tools_registered() -> None:
    server = ManagedResearchMcpServer()
    names = set(server.available_tool_names())
    assert EXPECTED_TOOLS.issubset(names)


def test_tool_schemas_are_public_safe_and_descriptions_present() -> None:
    server = ManagedResearchMcpServer()
    payload = {tool["name"]: tool for tool in server.list_tool_payload()}
    for name in EXPECTED_TOOLS:
        entry = payload[name]
        description = entry["description"]
        # No leak of internal scoring or profanity-list semantics.
        assert "profanity" not in description.lower()
        assert "scoring weight" not in description.lower()
        assert len(description) > 10


def test_submit_question_tool_requires_rubric_ack() -> None:
    server = ManagedResearchMcpServer()
    with pytest.raises(Exception) as info:
        server.call_tool(
            "open_research_submit_question",
            {
                "project_slug": "craftax",
                "queue_id": "q_oed_1h_craftax",
                "prompt": "prompt",
                "metric_target": {
                    "name": "craftax.reward.mean",
                    "operator": ">=",
                    "value": 0.5,
                },
                "deo_kind": "open_ended_discovery",
                "rubric_acknowledged": False,
                "submitter_handle": "anon-abc12",
                "submitter_fingerprint": "fp_test",
            },
        )
    assert "rubric_acknowledged" in str(info.value)


def _patched_server(monkeypatch, responses, recorder):
    """Replace OpenResearchClient.__post_init__ to mount a mock transport."""

    real_post_init = OpenResearchClient.__post_init__

    def patched(self) -> None:  # type: ignore[no-untyped-def]
        real_post_init(self)
        patch_client_transport(self, responses, recorder)

    monkeypatch.setattr(OpenResearchClient, "__post_init__", patched)
    return ManagedResearchMcpServer()


def test_list_projects_handler_returns_dict(monkeypatch) -> None:
    recorder: list[RecordedRequest] = []
    server = _patched_server(
        monkeypatch,
        [
            ScriptedResponse(
                json_body={
                    "projects": [
                        {
                            "slug": "craftax",
                            "name": "Craftax",
                            "tagline": "tag",
                            "challenge_url": "/x",
                            "default_queue_id": "q1",
                            "supported_queue_ids": ["q1"],
                        }
                    ]
                }
            )
        ],
        recorder,
    )
    out = server.call_tool("open_research_list_projects", {})
    assert out["projects"][0]["slug"] == "craftax"
    assert recorder[0].method == "GET"


def test_submit_question_uses_fingerprint_when_no_api_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MANAGED_RESEARCH_OR_FINGERPRINT_PATH", str(tmp_path / "fp"))
    recorder: list[RecordedRequest] = []
    server = _patched_server(
        monkeypatch,
        [
            ScriptedResponse(
                status_code=202,
                json_body={
                    "submission_id": "sub_1",
                    "status": "review_pending",
                    "review_verdict": None,
                    "duplicate": False,
                    "idempotency_key": "idem_1",
                },
            )
        ],
        recorder,
    )
    out = server.call_tool(
        "open_research_submit_question",
        {
            "project_slug": "craftax",
            "queue_id": "q_oed_1h_craftax",
            "prompt": "Sketch a wood plan",
            "metric_target": {
                "name": "craftax.reward.mean",
                "operator": ">=",
                "value": 0.6,
            },
            "deo_kind": "open_ended_discovery",
            "rubric_acknowledged": True,
            "submitter_handle": "anon-abc12",
        },
    )
    assert out["submission_id"] == "sub_1"
    rec = recorder[0]
    # Anonymous path: no Authorization, X-OR-Fingerprint set.
    assert "authorization" not in rec.headers
    assert rec.headers["x-or-fingerprint"]
    # Body submitter.fingerprint mirrors the header.
    assert rec.json_body["submitter"]["fingerprint"] == rec.headers["x-or-fingerprint"]


def test_submit_question_uses_authorization_when_api_key_present(monkeypatch) -> None:
    recorder: list[RecordedRequest] = []
    server = _patched_server(
        monkeypatch,
        [
            ScriptedResponse(
                status_code=202,
                json_body={
                    "submission_id": "sub_2",
                    "status": "review_pending",
                    "review_verdict": None,
                    "duplicate": False,
                    "idempotency_key": "idem_2",
                },
            )
        ],
        recorder,
    )
    server.call_tool(
        "open_research_submit_question",
        {
            "api_key": "syk_test_key",
            "project_slug": "craftax",
            "queue_id": "q_oed_1h_craftax",
            "prompt": "Sketch a wood plan",
            "metric_target": {
                "name": "craftax.reward.mean",
                "operator": ">=",
                "value": 0.6,
            },
            "deo_kind": "open_ended_discovery",
            "rubric_acknowledged": True,
            "submitter_handle": "user",
        },
    )
    rec = recorder[0]
    assert rec.headers["authorization"] == "Bearer syk_test_key"


def test_typed_error_envelope_handled_in_tool_path(monkeypatch) -> None:
    """Server handler should map ``OpenResearchError`` onto ``RpcError``."""
    recorder: list[RecordedRequest] = []
    server = _patched_server(
        monkeypatch,
        [
            ScriptedResponse(
                status_code=403,
                json_body={
                    "error": {
                        "class": "theme_fit",
                        "code": "submission_off_theme",
                        "message": "Off-theme for craftax.",
                        "actionable": "Refocus on a Craftax code-policy.",
                        "retry_after_seconds": None,
                        "request_id": "req_x",
                    }
                },
            )
        ],
        recorder,
    )

    # Invoke the handler directly (bypass JSON-RPC wrapping) so we can
    # assert on the typed RpcError that bubbles up.
    handler = server.get_tool_definition("open_research_submit_question").handler
    with pytest.raises(RpcError) as info:
        handler(
            {
                "project_slug": "craftax",
                "queue_id": "q_oed_1h_craftax",
                "prompt": "knitting tutorial",
                "metric_target": {
                    "name": "craftax.reward.mean",
                    "operator": ">=",
                    "value": 0.5,
                },
                "deo_kind": "open_ended_discovery",
                "rubric_acknowledged": True,
                "submitter_handle": "anon",
                "submitter_fingerprint": "fp_x",
            }
        )
    err = info.value
    data = err.data
    assert isinstance(data, dict)
    assert data["error"]["class"] == "theme_fit"
    assert data["error"]["code"] == "submission_off_theme"
    assert data["error"]["actionable"] == "Refocus on a Craftax code-policy."
    assert data["http_status"] == 403


def test_list_experiments_passes_filter_query(monkeypatch) -> None:
    recorder: list[RecordedRequest] = []
    server = _patched_server(
        monkeypatch,
        [ScriptedResponse(json_body={"experiments": [], "next_cursor": None})],
        recorder,
    )
    server.call_tool(
        "open_research_list_experiments",
        {"project_slug": "craftax", "status": "running", "limit": 10},
    )
    assert recorder[0].params == {
        "project_slug": "craftax",
        "status": "running",
        "limit": "10",
    }


def test_download_bundle_writes_file_and_returns_sha(monkeypatch, tmp_path) -> None:
    import gzip
    import hashlib

    blob = gzip.compress(b"contents")
    sha = hashlib.sha256(blob).hexdigest()
    recorder: list[RecordedRequest] = []
    server = _patched_server(
        monkeypatch,
        [
            ScriptedResponse(
                status_code=200,
                raw_body=blob,
                content_type="application/gzip",
            )
        ],
        recorder,
    )
    dest = tmp_path / "exp.tar.gz"
    out = server.call_tool(
        "open_research_download_bundle",
        {"experiment_id": "exp_1", "dest_path": str(dest)},
    )
    assert out["sha256"] == sha
    assert out["bytes_written"] == len(blob)
    assert dest.read_bytes() == blob
