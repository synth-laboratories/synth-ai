"""Unit tests for managed_research SDK."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

import httpx


def test_resolve_api_key_explicit() -> None:
    from synth_ai.sdk.managed_research import _resolve_api_key

    assert _resolve_api_key("sk_explicit") == "sk_explicit"


def test_resolve_api_key_env_fallback() -> None:
    from synth_ai.sdk.managed_research import _resolve_api_key

    old_value = os.environ.get("SYNTH_API_KEY")
    try:
        os.environ["SYNTH_API_KEY"] = "sk_env"
        with patch("synth_ai.sdk.managed_research.get_api_key", side_effect=Exception("missing")):
            assert _resolve_api_key(None) == "sk_env"
    finally:
        if old_value is None:
            os.environ.pop("SYNTH_API_KEY", None)
        else:
            os.environ["SYNTH_API_KEY"] = old_value


def test_list_runs_falls_back_to_canonical_route() -> None:
    from synth_ai.sdk.managed_research import SmrControlClient

    calls: list[tuple[str, str, dict | None]] = []

    def fake_request(
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json: dict | None = None,
    ) -> httpx.Response:
        del json
        calls.append((method, path, params))
        request = httpx.Request(method, f"https://api.example.com{path}")
        if path == "/smr/projects/project-1/runs":
            return httpx.Response(404, request=request, json={"detail": "not found"})
        if path == "/smr/runs":
            return httpx.Response(200, request=request, json=[{"run_id": "run-1"}])
        raise AssertionError(f"Unexpected path: {path}")

    with SmrControlClient(api_key="sk_test", backend_base="https://api.example.com") as client:
        client._client.request = fake_request  # type: ignore[assignment]
        runs = client.list_runs("project-1")

    assert runs == [{"run_id": "run-1"}]
    assert calls == [
        ("GET", "/smr/projects/project-1/runs", None),
        ("GET", "/smr/runs", {"project_id": "project-1"}),
    ]


def test_set_provider_key_retries_with_plaintext_on_422() -> None:
    from synth_ai.sdk.managed_research import SmrControlClient

    payloads: list[dict] = []

    def fake_post(path: str, *, json: dict | None = None) -> httpx.Response:
        payloads.append(dict(json or {}))
        request = httpx.Request("POST", f"https://api.example.com{path}")
        if len(payloads) == 1:
            return httpx.Response(422, request=request, json={"detail": "schema mismatch"})
        return httpx.Response(200, request=request, json={"ok": True})

    with SmrControlClient(api_key="sk_test", backend_base="https://api.example.com") as client:
        client._client.post = fake_post  # type: ignore[assignment]
        out = client.set_provider_key(
            "project-1",
            provider="openai",
            funding_source="customer",
            api_key="plain-key",
            encrypted_key_b64="encrypted-key",
        )

    assert out == {"ok": True}
    assert payloads[0]["encrypted_key_b64"] == "encrypted-key"
    assert payloads[1]["api_key"] == "plain-key"


def test_list_runs_returns_empty_when_both_routes_404() -> None:
    from synth_ai.sdk.managed_research import SmrControlClient

    calls: list[tuple[str, str, dict | None]] = []

    def fake_request(
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json: dict | None = None,
    ) -> httpx.Response:
        del json
        calls.append((method, path, params))
        request = httpx.Request(method, f"https://api.example.com{path}")
        if path in {"/smr/projects/project-404/runs", "/smr/runs"}:
            return httpx.Response(404, request=request, json={"detail": "not found"})
        raise AssertionError(f"Unexpected path: {path}")

    with SmrControlClient(api_key="sk_test", backend_base="https://api.example.com") as client:
        client._client.request = fake_request  # type: ignore[assignment]
        runs = client.list_runs("project-404")

    assert runs == []
    assert calls == [
        ("GET", "/smr/projects/project-404/runs", None),
        ("GET", "/smr/runs", {"project_id": "project-404"}),
    ]


def test_first_id_returns_first_non_empty_value() -> None:
    from synth_ai.sdk.managed_research import first_id

    rows = [
        {"run_id": ""},
        {"run_id": "  "},
        {"run_id": "run-2"},
        {"run_id": "run-3"},
    ]
    assert first_id(rows, "run_id") == "run-2"


def test_get_starting_data_upload_urls_posts_expected_payload() -> None:
    from synth_ai.sdk.managed_research import SmrControlClient

    calls: list[tuple[str, str, dict | None]] = []

    def fake_request(
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json: dict | None = None,
    ) -> httpx.Response:
        del params
        calls.append((method, path, json))
        request = httpx.Request(method, f"https://api.example.com{path}")
        return httpx.Response(
            200,
            request=request,
            json={
                "dataset_ref": "starting-data/banking77",
                "s3_path": "s3://bucket/prefix",
                "uploads": [{"path": "banking77/input_spec.json", "upload_url": "https://upload.example/1"}],
            },
        )

    with SmrControlClient(api_key="sk_test", backend_base="https://api.example.com") as client:
        client._client.request = fake_request  # type: ignore[assignment]
        response = client.get_starting_data_upload_urls(
            "project-1",
            dataset_ref="starting-data/banking77",
            files=[
                {"path": "banking77/input_spec.json", "content_type": "application/json"},
                {"path": "banking77/README.md", "content_type": "text/markdown"},
            ],
        )

    assert response["dataset_ref"] == "starting-data/banking77"
    assert calls == [
        (
            "POST",
            "/smr/projects/project-1/starting-data/upload-urls",
            {
                "dataset_ref": "starting-data/banking77",
                "files": [
                    {"path": "banking77/input_spec.json", "content_type": "application/json"},
                    {"path": "banking77/README.md", "content_type": "text/markdown"},
                ],
            },
        )
    ]


def test_upload_starting_data_files_uploads_presigned_urls() -> None:
    from synth_ai.sdk.managed_research import SmrControlClient

    upload_client = MagicMock()
    upload_client.__enter__.return_value = upload_client
    upload_client.__exit__.return_value = None
    upload_client.put.side_effect = [
        httpx.Response(200, request=httpx.Request("PUT", "https://upload.example/input-spec")),
        httpx.Response(200, request=httpx.Request("PUT", "https://upload.example/readme")),
    ]
    with (
        SmrControlClient(api_key="sk_test", backend_base="https://api.example.com") as client,
        patch.object(
            client,
            "get_starting_data_upload_urls",
            return_value={
                "dataset_ref": "starting-data/banking77",
                "s3_path": "s3://bucket/prefix",
                "uploads": [
                    {
                        "path": "banking77/input_spec.json",
                        "upload_url": "https://upload.example/input-spec",
                    },
                    {
                        "path": "banking77/README.md",
                        "upload_url": "https://upload.example/readme",
                    },
                ],
            },
        ) as mock_get_urls,
        patch("synth_ai.sdk.managed_research.httpx.Client", return_value=upload_client),
    ):
        response = client.upload_starting_data_files(
            "project-1",
            dataset_ref="starting-data/banking77",
            files=[
                {
                    "path": "banking77/input_spec.json",
                    "content": '{"kind":"eval_job"}',
                    "content_type": "application/json",
                },
                {
                    "path": "banking77/README.md",
                    "content": "# Banking77",
                    "content_type": "text/markdown",
                },
            ],
        )

    assert response["dataset_ref"] == "starting-data/banking77"
    mock_get_urls.assert_called_once()
    assert upload_client.put.call_count == 2
    first_call = upload_client.put.call_args_list[0]
    assert first_call.args[0] == "https://upload.example/input-spec"
    assert first_call.kwargs["headers"] == {"Content-Type": "application/json"}


def test_upload_starting_data_directory_collects_local_files(tmp_path) -> None:
    from synth_ai.sdk.managed_research import SmrControlClient

    data_dir = tmp_path / "starting_data"
    data_dir.mkdir()
    (data_dir / "banking77").mkdir()
    (data_dir / "banking77" / "input_spec.json").write_text('{"kind":"eval_job"}', encoding="utf-8")
    (data_dir / "banking77" / "README.md").write_text("# Banking77", encoding="utf-8")

    with (
        SmrControlClient(api_key="sk_test", backend_base="https://api.example.com") as client,
        patch.object(client, "upload_starting_data_files", return_value={"ok": True}) as mock_upload,
    ):
        out = client.upload_starting_data_directory(
            "project-1",
            data_dir,
            dataset_ref="starting-data/banking77",
        )

    assert out == {"ok": True}
    call = mock_upload.call_args
    assert call.kwargs["dataset_ref"] == "starting-data/banking77"
    uploaded_paths = [item["path"] for item in call.kwargs["files"]]
    assert uploaded_paths == ["banking77/README.md", "banking77/input_spec.json"]


@pytest.mark.skip(reason="get_run_spend_entries not yet implemented")
def test_get_run_spend_entries_uses_admin_route() -> None:
    from synth_ai.sdk.managed_research import SmrControlClient

    calls: list[tuple[str, str, dict | None]] = []

    def fake_request(
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json: dict | None = None,
    ) -> httpx.Response:
        del json
        calls.append((method, path, params))
        request = httpx.Request(method, f"https://api.example.com{path}")
        return httpx.Response(200, request=request, json={"run_id": "run-1", "entries": []})

    with SmrControlClient(api_key="sk_test", backend_base="https://api.example.com") as client:
        client._client.request = fake_request  # type: ignore[assignment]
        out = client.get_run_spend_entries("run-1")

    assert out["run_id"] == "run-1"
    assert calls == [("GET", "/smr/admin/runs/run-1/spend", None)]


@pytest.mark.skip(reason="get_run_usage_by_actor not yet implemented")
def test_get_run_usage_by_actor_groups_worker_and_orchestrator_usage() -> None:
    from synth_ai.sdk.managed_research import SmrControlClient

    spend_payload = {
        "run_id": "run-1",
        "entries": [
            {
                "ledger_id": "l-orch-1",
                "project_id": "project-1",
                "episode_id": "orchestrator.first_turn",
                "provider": "openai",
                "model": "gpt-5.2",
                "meter_kind": "token_input",
                "quantity": 1000,
                "cost_cents": 10,
                "funding_source": "synth",
                "metadata": {
                    "usage_category": "orchestrator",
                    "worker_id": "orch-1",
                    "sandbox_id": "sb-orch-1",
                    "agent_model": "gpt-5.2",
                },
                "created_at": "2026-02-20T00:00:00Z",
            },
            {
                "ledger_id": "l-worker-1",
                "project_id": "project-1",
                "episode_id": "task-1",
                "provider": "openai",
                "model": "gpt-5.2",
                "meter_kind": "token_output",
                "quantity": 500,
                "cost_cents": 15,
                "funding_source": "synth",
                "metadata": {
                    "usage_category": "worker",
                    "sandbox_id": "sb-worker-1",
                    "agent_model": "gpt-5.2",
                },
                "created_at": "2026-02-20T00:01:00Z",
            },
            {
                "ledger_id": "l-worker-2",
                "project_id": "project-1",
                "episode_id": "task-1",
                "provider": "daytona",
                "model": None,
                "meter_kind": "sandbox_seconds",
                "quantity": 240,
                "cost_cents": 4,
                "funding_source": "synth",
                "metadata": {
                    "usage_category": "worker",
                    "sandbox_id": "sb-worker-1",
                },
                "created_at": "2026-02-20T00:01:10Z",
            },
            {
                "ledger_id": "l-worker-3",
                "project_id": "project-1",
                "episode_id": "task-2",
                "provider": "openai",
                "model": "gpt-5.2",
                "meter_kind": "token_input",
                "quantity": 200,
                "cost_cents": 3,
                "funding_source": "synth",
                "metadata": {
                    "usage_category": "worker",
                    "sandbox_id": "sb-worker-2",
                },
                "created_at": "2026-02-20T00:02:00Z",
            },
        ],
    }
    ops_status_payload = {
        "orchestrator": {"status": "running", "claimed_by": "orch-1"},
        "workers": {
            "run_id": "run-1",
            "active_workers": ["worker-a", "worker-b"],
            "tasks": [
                {
                    "task_id": "task-1",
                    "task_key": "policy_opt:gepa",
                    "kind": "policy_optimization",
                    "state": "executing",
                    "claimed_by": "worker-a",
                },
                {
                    "task_id": "task-2",
                    "task_key": "synthesize_results",
                    "kind": "synthesis",
                    "state": "queued",
                    "claimed_by": "worker-b",
                },
            ],
        },
    }

    with (
        SmrControlClient(api_key="sk_test", backend_base="https://api.example.com") as client,
        patch.object(client, "get_run_spend_entries", return_value=spend_payload),
        patch.object(client, "get_ops_status", return_value=ops_status_payload),
    ):
        out = client.get_run_usage_by_actor("run-1")

    summary = out["summary"]
    assert summary["total_cost_cents"] == 32
    assert summary["total_cost_usd"] == 0.32
    assert summary["orchestrator_total_cost_cents"] == 10
    assert summary["orchestrator_total_cost_usd"] == 0.1
    assert summary["worker_total_cost_cents"] == 22
    assert summary["worker_total_cost_usd"] == 0.22
    assert summary["orchestrator_count"] == 1
    assert summary["worker_count"] == 2
    assert summary["cost_data_available"] is True
    assert summary["meter_quantities"]["token_input"] == 1200.0
    assert summary["meter_quantities"]["token_output"] == 500.0
    assert summary["meter_quantities"]["sandbox_seconds"] == 240.0
    assert summary["meter_cost_cents"]["token_input"] == 13
    assert summary["meter_cost_cents"]["token_output"] == 15
    assert summary["meter_cost_cents"]["sandbox_seconds"] == 4
    assert summary["token_usage"]["input_tokens"] == 1200
    assert summary["token_usage"]["output_tokens"] == 500
    assert summary["token_usage"]["total_tokens"] == 1700
    assert summary["token_cost_cents"]["input_cost_cents"] == 13
    assert summary["token_cost_cents"]["output_cost_cents"] == 15
    assert summary["token_cost_cents"]["total_cost_cents"] == 28

    orchestrators = {entry["actor_id"]: entry for entry in out["orchestrators"]}
    workers = {entry["actor_id"]: entry for entry in out["workers"]}

    assert set(orchestrators.keys()) == {"orch-1"}
    assert set(workers.keys()) == {"worker-a", "worker-b"}

    assert orchestrators["orch-1"]["total_cost_cents"] == 10
    assert orchestrators["orch-1"]["total_cost_usd"] == 0.1
    assert orchestrators["orch-1"]["token_usage"]["input_tokens"] == 1000
    assert orchestrators["orch-1"]["token_usage"]["output_tokens"] == 0
    assert orchestrators["orch-1"]["token_usage"]["total_tokens"] == 1000
    assert orchestrators["orch-1"]["token_cost_cents"]["total_cost_cents"] == 10
    orch_models = {
        (row["provider"], row["model"]): row["total_cost_cents"]
        for row in orchestrators["orch-1"]["models"]
    }
    assert orch_models[("openai", "gpt-5.2")] == 10

    assert workers["worker-a"]["total_cost_cents"] == 19
    assert workers["worker-a"]["total_cost_usd"] == 0.19
    assert workers["worker-a"]["active"] is True
    assert workers["worker-a"]["meter_quantities"]["token_output"] == 500.0
    assert workers["worker-a"]["meter_quantities"]["sandbox_seconds"] == 240.0
    assert workers["worker-a"]["token_usage"]["input_tokens"] == 0
    assert workers["worker-a"]["token_usage"]["output_tokens"] == 500
    assert workers["worker-a"]["token_usage"]["total_tokens"] == 500
    assert workers["worker-a"]["token_cost_cents"]["total_cost_cents"] == 15
    worker_a_models = {
        (row["provider"], row["model"]): row["total_cost_cents"]
        for row in workers["worker-a"]["models"]
    }
    assert worker_a_models[("openai", "gpt-5.2")] == 15
    assert worker_a_models[("daytona", None)] == 4
    assert workers["worker-a"]["tasks"] == [
        {
            "task_id": "task-1",
            "task_key": "policy_opt:gepa",
            "kind": "policy_optimization",
            "state": "executing",
            "claimed_by": "worker-a",
        }
    ]

    assert workers["worker-b"]["total_cost_cents"] == 3
    assert workers["worker-b"]["total_cost_usd"] == 0.03
    assert workers["worker-b"]["active"] is True
    assert workers["worker-b"]["token_usage"]["input_tokens"] == 200
    assert workers["worker-b"]["token_usage"]["output_tokens"] == 0
    assert workers["worker-b"]["token_usage"]["total_tokens"] == 200
    assert workers["worker-b"]["token_cost_cents"]["total_cost_cents"] == 3


@pytest.mark.skip(reason="get_run_usage_by_actor not yet implemented")
def test_get_run_usage_by_actor_falls_back_to_logs_when_admin_spend_unavailable() -> None:
    from synth_ai.sdk.managed_research import SmrApiError, SmrControlClient

    token_status_orchestrator = (
        '[{"type":"status","label":"thread.token_usage.updated","detail":"'
        '{\\"threadId\\":\\"t-1\\",\\"tokenUsage\\":{\\"total\\":{'
        '\\"inputTokens\\":120,\\"cachedInputTokens\\":20,\\"outputTokens\\":40,'
        '\\"reasoningOutputTokens\\":5,\\"totalTokens\\":160}}}"}]'
    )
    token_status_worker = (
        '[{"type":"status","label":"thread.token_usage.updated","detail":"'
        '{\\"threadId\\":\\"t-2\\",\\"tokenUsage\\":{\\"total\\":{'
        '\\"inputTokens\\":300,\\"cachedInputTokens\\":100,\\"outputTokens\\":80,'
        '\\"reasoningOutputTokens\\":10,\\"totalTokens\\":380}}}"}]'
    )
    logs_payload = {
        "ok": True,
        "records": [
            {
                "_time": "2026-02-20T01:00:00Z",
                "component": "orchestrator",
                "agent_id": "orch-1",
                "session_id": "sess-orch-1",
                "raw.data.metadata.model": "gpt-5.2",
                "raw.data.item.content": token_status_orchestrator,
            },
            {
                "_time": "2026-02-20T01:01:00Z",
                "component": "worker",
                "agent_id": "worker-a",
                "task_key": "policy_opt:gepa",
                "session_id": "sess-worker-1",
                "raw.data.metadata.model": "gpt-5.2",
                "raw.data.item.content": token_status_worker,
            },
        ],
    }
    ops_status_payload = {
        "orchestrator": {"status": "running", "claimed_by": "orch-1"},
        "workers": {
            "run_id": "run-1",
            "active_workers": ["worker-a"],
            "tasks": [
                {
                    "task_id": "task-1",
                    "task_key": "policy_opt:gepa",
                    "kind": "policy_optimization",
                    "state": "executing",
                    "claimed_by": "worker-a",
                }
            ],
        },
    }

    with (
        SmrControlClient(api_key="sk_test", backend_base="https://api.example.com") as client,
        patch.object(
            client,
            "get_run_spend_entries",
            side_effect=SmrApiError("GET /smr/admin/runs/run-1/spend failed (503): admin auth not configured"),
        ),
        patch.object(client, "get_ops_status", return_value=ops_status_payload),
        patch.object(client, "search_victoria_logs", return_value=logs_payload),
        patch.object(
            client,
            "get_usage",
            return_value={
                "project_id": "project-1",
                "per_run": [{"run_id": "run-1", "total_cost_cents": 54}],
            },
        ),
        patch.object(client, "get_run", return_value={"run_id": "run-1", "project_id": "project-1"}),
    ):
        out = client.get_run_usage_by_actor("run-1", project_id="project-1")

    assert out["usage_mode"] == "logs_thread_totals"
    assert "admin auth not configured" in str(out["spend_error"])
    assert out["summary"]["cost_data_available"] is False
    assert out["summary"]["total_cost_usd"] is None
    assert out["summary"]["token_cost_cents"] is None
    assert out["summary"]["estimated_total_cost_cents"] == 54
    assert out["summary"]["estimated_total_cost_usd"] == 0.54
    assert out["summary"]["estimated_orchestrator_total_cost_cents"] == 16
    assert out["summary"]["estimated_worker_total_cost_cents"] == 38
    assert out["summary"]["estimated_cost_data_available"] is True
    assert out["summary"]["estimated_cost_source"] == "project_usage_per_run_token_share"
    assert out["summary"]["orchestrator_count"] == 1
    assert out["summary"]["worker_count"] == 1
    assert out["summary"]["token_usage"]["total_tokens"] == 540
    assert out["summary"]["meter_quantities"]["token_input"] == 300.0
    assert out["summary"]["meter_quantities"]["token_cached_input"] == 120.0
    assert out["summary"]["meter_quantities"]["token_output"] == 105.0
    assert out["summary"]["meter_quantities"]["token_reasoning"] == 15.0
    assert out["summary"]["orchestrator_token_usage"]["total_tokens"] == 160
    assert out["summary"]["worker_token_usage"]["total_tokens"] == 380

    orchestrator = out["orchestrators"][0]
    worker = out["workers"][0]

    assert orchestrator["actor_id"] == "orch-1"
    assert orchestrator["active"] is True
    assert orchestrator["total_cost_usd"] is None
    assert orchestrator["token_cost_cents"] is None
    assert orchestrator["estimated_total_cost_cents"] == 16
    assert orchestrator["estimated_total_cost_usd"] == 0.16
    assert orchestrator["meter_quantities"]["token_input"] == 100.0
    assert orchestrator["meter_quantities"]["token_cached_input"] == 20.0
    assert orchestrator["meter_quantities"]["token_output"] == 35.0
    assert orchestrator["meter_quantities"]["token_reasoning"] == 5.0
    assert orchestrator["models"][0]["model"] == "gpt-5.2"
    assert orchestrator["models"][0]["token_usage"]["total_tokens"] == 160

    assert worker["actor_id"] == "worker-a"
    assert worker["active"] is True
    assert worker["total_cost_usd"] is None
    assert worker["token_cost_cents"] is None
    assert worker["estimated_total_cost_cents"] == 38
    assert worker["estimated_total_cost_usd"] == 0.38
    assert worker["meter_quantities"]["token_input"] == 200.0
    assert worker["meter_quantities"]["token_cached_input"] == 100.0
    assert worker["meter_quantities"]["token_output"] == 70.0
    assert worker["meter_quantities"]["token_reasoning"] == 10.0
    assert worker["models"][0]["model"] == "gpt-5.2"
    assert worker["models"][0]["token_usage"]["total_tokens"] == 380
    assert worker["tasks"] == [
        {
            "task_id": "task-1",
            "task_key": "policy_opt:gepa",
            "kind": "policy_optimization",
            "state": "executing",
            "claimed_by": "worker-a",
        }
    ]
