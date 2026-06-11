"""Unit tests for ``OpenResearchClient`` HTTP contract behavior."""

from __future__ import annotations

import gzip
import hashlib

import pytest
from synth_ai.managed_research.open_research import (
    OpenResearchClient,
    OpenResearchError,
    SubmitQuestionArgs,
)
from synth_ai.managed_research.open_research.client import OPEN_RESEARCH_BASE
from synth_ai.managed_research.open_research.models import MetricTarget
from tests.open_research.conftest import (
    RecordedRequest,
    ScriptedResponse,
    patch_client_transport,
)


def _new_client(*, api_key=None, fingerprint=None):
    client = OpenResearchClient(
        api_key=api_key,
        fingerprint=fingerprint,
        backend_base="http://test.invalid",
    )
    return client


def test_list_projects_hits_locked_path_and_returns_typed_payload() -> None:
    client = _new_client()
    recorder: list[RecordedRequest] = []
    patch_client_transport(
        client,
        [
            ScriptedResponse(
                status_code=200,
                json_body={
                    "projects": [
                        {
                            "slug": "craftax",
                            "name": "Craftax",
                            "tagline": "Open-ended Craftax",
                            "challenge_url": "/open-research/craftax",
                            "baseline_score": 0.42,
                            "current_best_score": 0.71,
                            "best_experiment_id": "exp_1",
                            "default_queue_id": "q_oed_1h_craftax",
                            "supported_queue_ids": ["q_oed_1h_craftax"],
                        }
                    ]
                },
            )
        ],
        recorder,
    )

    result = client.list_projects()

    assert recorder[0].method == "GET"
    assert recorder[0].path == f"{OPEN_RESEARCH_BASE}/projects"
    assert len(result.projects) == 1
    assert result.projects[0].slug == "craftax"
    assert result.projects[0].default_queue_id == "q_oed_1h_craftax"


def test_signed_in_auth_header_is_bearer_token() -> None:
    client = _new_client(api_key="syk_secret")
    recorder: list[RecordedRequest] = []
    patch_client_transport(
        client,
        [ScriptedResponse(json_body={"projects": []})],
        recorder,
    )
    client.list_projects()
    assert recorder[0].headers["authorization"] == "Bearer syk_secret"
    assert "x-or-fingerprint" not in recorder[0].headers


def test_anonymous_caller_sets_fingerprint_header_and_omits_authorization() -> None:
    client = _new_client(fingerprint="fp_abc")
    recorder: list[RecordedRequest] = []
    patch_client_transport(
        client,
        [ScriptedResponse(json_body={"projects": []})],
        recorder,
    )
    client.list_projects()
    assert recorder[0].headers["x-or-fingerprint"] == "fp_abc"
    assert "authorization" not in recorder[0].headers


def test_list_queues_passes_project_slug_query_param() -> None:
    client = _new_client()
    recorder: list[RecordedRequest] = []
    patch_client_transport(
        client,
        [ScriptedResponse(json_body={"queues": []})],
        recorder,
    )
    client.list_queues(project_slug="craftax")
    assert recorder[0].path == f"{OPEN_RESEARCH_BASE}/queues"
    assert recorder[0].params == {"project_slug": "craftax"}


def test_submit_question_posts_contract_body_with_submitter() -> None:
    client = _new_client(fingerprint="fp_seed")
    recorder: list[RecordedRequest] = []
    patch_client_transport(
        client,
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
    args = SubmitQuestionArgs(
        project_slug="craftax",
        queue_id="q_oed_1h_craftax",
        prompt="Sketch a wood-collecting plan",
        hypothesis="Mining-first wastes early budget",
        metric_target=MetricTarget(name="craftax.reward.mean", operator=">=", value=0.6),
        deo_kind="open_ended_discovery",
        rubric_acknowledged=True,
        submitter_handle="anon-abc12",
        submitter_fingerprint="fp_explicit",
    )
    response = client.submit_question(args)

    rec = recorder[0]
    assert rec.method == "POST"
    assert rec.path == f"{OPEN_RESEARCH_BASE}/submissions"
    body = rec.json_body or {}
    assert body["project_slug"] == "craftax"
    assert body["queue_id"] == "q_oed_1h_craftax"
    assert body["rubric_acknowledged"] is True
    assert body["metric_target"] == {
        "name": "craftax.reward.mean",
        "operator": ">=",
        "value": 0.6,
    }
    assert body["submitter"] == {"handle": "anon-abc12", "fingerprint": "fp_explicit"}
    assert response.submission_id == "sub_1"
    assert response.status == "review_pending"


def test_submit_question_rubric_must_be_true() -> None:
    with pytest.raises(ValueError):
        SubmitQuestionArgs(
            project_slug="craftax",
            queue_id="q_oed_1h_craftax",
            prompt="prompt",
            metric_target=MetricTarget(name="m", operator=">=", value=0.1),
            deo_kind="open_ended_discovery",
            rubric_acknowledged=False,
            submitter_handle="anon",
        )


def test_typed_error_envelope_lifts_class_and_actionable() -> None:
    client = _new_client(fingerprint="fp_abc")
    recorder: list[RecordedRequest] = []
    patch_client_transport(
        client,
        [
            ScriptedResponse(
                status_code=403,
                json_body={
                    "error": {
                        "class": "safety",
                        "code": "submission_disallowed_content",
                        "message": "We can't run this prompt.",
                        "actionable": "Rephrase without disallowed content and resubmit.",
                        "retry_after_seconds": None,
                        "request_id": "req_42",
                    }
                },
            )
        ],
        recorder,
    )
    args = SubmitQuestionArgs(
        project_slug="craftax",
        queue_id="q_oed_1h_craftax",
        prompt="prompt",
        metric_target=MetricTarget(name="m", operator=">=", value=0.1),
        deo_kind="open_ended_discovery",
        rubric_acknowledged=True,
        submitter_handle="anon",
    )
    with pytest.raises(OpenResearchError) as info:
        client.submit_question(args)
    err = info.value
    assert err.error_class == "safety"
    assert err.code == "submission_disallowed_content"
    assert err.actionable == "Rephrase without disallowed content and resubmit."
    assert err.status_code == 403
    assert err.request_id == "req_42"
    payload = err.to_mcp_payload()
    assert payload["error"]["class"] == "safety"
    assert payload["error"]["actionable"].startswith("Rephrase")
    assert payload["http_status"] == 403


def test_list_experiments_drops_unset_query_params() -> None:
    client = _new_client()
    recorder: list[RecordedRequest] = []
    patch_client_transport(
        client,
        [ScriptedResponse(json_body={"experiments": [], "next_cursor": None})],
        recorder,
    )
    client.list_experiments(project_slug="craftax", limit=25)
    assert recorder[0].params == {"project_slug": "craftax", "limit": "25"}


def test_get_submission_path(tmp_path) -> None:
    client = _new_client()
    recorder: list[RecordedRequest] = []
    patch_client_transport(
        client,
        [
            ScriptedResponse(
                json_body={
                    "submission_id": "sub_1",
                    "project_slug": "craftax",
                    "queue_id": "q_oed_1h_craftax",
                    "status": "approved",
                    "review_verdict": None,
                    "experiment_id": "exp_1",
                    "objective_id": "obj_1",
                    "submitted_at": "2026-05-13T22:13:00Z",
                    "launched_at": "2026-05-13T22:14:09Z",
                }
            )
        ],
        recorder,
    )
    result = client.get_submission("sub_1")
    assert recorder[0].path == f"{OPEN_RESEARCH_BASE}/submissions/sub_1"
    assert result.status == "approved"
    assert result.experiment_id == "exp_1"


def test_download_bundle_streams_to_disk_and_reports_sha256(tmp_path) -> None:
    client = _new_client()
    recorder: list[RecordedRequest] = []
    payload = gzip.compress(b"hello bundle")
    expected_sha = hashlib.sha256(payload).hexdigest()
    patch_client_transport(
        client,
        [
            ScriptedResponse(
                status_code=200,
                raw_body=payload,
                content_type="application/gzip",
            )
        ],
        recorder,
    )
    dest = tmp_path / "bundle.tar.gz"
    result = client.download_bundle("exp_1", dest)
    assert recorder[0].path == f"{OPEN_RESEARCH_BASE}/experiments/exp_1/bundle"
    assert result.bytes_written == len(payload)
    assert result.sha256 == expected_sha
    assert dest.read_bytes() == payload
    assert result.content_type == "application/gzip"


def test_download_bundle_typed_error_for_pending_assembly(tmp_path) -> None:
    client = _new_client()
    recorder: list[RecordedRequest] = []
    patch_client_transport(
        client,
        [
            ScriptedResponse(
                status_code=409,
                json_body={
                    "error": {
                        "class": "transient",
                        "code": "bundle_assembly_pending",
                        "message": "Bundle is still being assembled.",
                        "actionable": "Retry in a few seconds.",
                        "retry_after_seconds": 5,
                        "request_id": "req_99",
                    }
                },
            )
        ],
        recorder,
    )
    with pytest.raises(OpenResearchError) as info:
        client.download_bundle("exp_1", tmp_path / "bundle.tar.gz")
    assert info.value.error_class == "transient"
    assert info.value.retry_after_seconds == 5
