"""Unit tests for managed_research SDK."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

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

    with SmrControlClient(api_key="sk_test", backend_base="https://api.example.com") as client:
        with patch.object(
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
        ) as mock_get_urls:
            upload_client = MagicMock()
            upload_client.__enter__.return_value = upload_client
            upload_client.__exit__.return_value = None
            upload_client.put.side_effect = [
                httpx.Response(200, request=httpx.Request("PUT", "https://upload.example/input-spec")),
                httpx.Response(200, request=httpx.Request("PUT", "https://upload.example/readme")),
            ]

            with patch("synth_ai.sdk.managed_research.httpx.Client", return_value=upload_client):
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

    with SmrControlClient(api_key="sk_test", backend_base="https://api.example.com") as client:
        with patch.object(client, "upload_starting_data_files", return_value={"ok": True}) as mock_upload:
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
