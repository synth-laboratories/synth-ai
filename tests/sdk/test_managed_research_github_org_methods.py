from __future__ import annotations

from typing import Any

import pytest

from synth_ai.sdk import managed_research
from synth_ai.sdk.managed_research import SmrControlClient


def test_github_org_methods_call_expected_routes() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def fake_request_json(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> dict[str, Any]:
        calls.append((method, path, json_body))
        return {"ok": True}

    client._request_json = fake_request_json  # type: ignore[method-assign]

    assert client.github_org_status() == {"ok": True}
    assert client.github_org_oauth_start(redirect_uri="http://localhost:9876/callback") == {"ok": True}
    assert (
        client.github_org_oauth_callback(
            code="oauth-code",
            state="oauth-state",
            redirect_uri="http://localhost:9876/callback",
        )
        == {"ok": True}
    )
    assert client.github_org_disconnect() == {"ok": True}
    assert client.github_org_pat_connect(pat="ghp_org_pat_123") == {"ok": True}
    assert (
        client.github_pat_connect(
            project_id="proj_789",
            pat="ghp_project_pat_123",
            repo="owner/repo",
            pr_write_enabled=True,
        )
        == {"ok": True}
    )
    assert client.github_link_org("proj_123") == {"ok": True}
    assert client.link_org_github("proj_456") == {"ok": True}

    assert calls == [
        ("GET", "/smr/integrations/github/org/status", None),
        (
            "POST",
            "/smr/integrations/github/org/oauth/start",
            {"redirect_uri": "http://localhost:9876/callback"},
        ),
        (
            "POST",
            "/smr/integrations/github/org/oauth/callback",
            {
                "code": "oauth-code",
                "state": "oauth-state",
                "redirect_uri": "http://localhost:9876/callback",
            },
        ),
        ("DELETE", "/smr/integrations/github/org/disconnect", None),
        ("POST", "/smr/integrations/github/org/pat/connect", {"pat": "ghp_org_pat_123"}),
        (
            "POST",
            "/smr/integrations/github/pat/connect",
            {
                "project_id": "proj_789",
                "pat": "ghp_project_pat_123",
                "repo": "owner/repo",
                "pr_write_enabled": True,
            },
        ),
        ("POST", "/smr/projects/proj_123/integrations/github/link-org", None),
        ("POST", "/smr/projects/proj_456/integrations/github/link-org", None),
    ]
    client.close()


def test_linear_integration_methods_call_expected_routes() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def fake_request_json(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> dict[str, Any]:
        calls.append((method, path, json_body))
        return {"ok": True}

    client._request_json = fake_request_json  # type: ignore[method-assign]

    assert (
        client.linear_oauth_start(project_id="proj_1", redirect_uri="http://localhost:9876/callback")
        == {"ok": True}
    )
    assert (
        client.linear_oauth_callback(
            project_id="proj_1",
            code="oauth-code",
            state="oauth-state",
            redirect_uri="http://localhost:9876/callback",
        )
        == {"ok": True}
    )
    assert client.linear_status("proj_1") == {"ok": True}
    assert client.linear_disconnect("proj_1") == {"ok": True}
    assert client.linear_list_teams("proj_1") == {"ok": True}
    assert client.linear_provision("proj_1", team_id="team_1", project_name="SMR Work") == {"ok": True}

    assert calls == [
        (
            "POST",
            "/smr/integrations/linear/oauth/start",
            {"project_id": "proj_1", "redirect_uri": "http://localhost:9876/callback"},
        ),
        (
            "POST",
            "/smr/integrations/linear/oauth/callback",
            {
                "project_id": "proj_1",
                "code": "oauth-code",
                "state": "oauth-state",
                "redirect_uri": "http://localhost:9876/callback",
            },
        ),
        ("GET", "/smr/projects/proj_1/integrations/linear/status", None),
        ("DELETE", "/smr/projects/proj_1/integrations/linear/disconnect", None),
        ("GET", "/smr/projects/proj_1/integrations/linear/teams", None),
        (
            "POST",
            "/smr/projects/proj_1/integrations/linear/provision",
            {"team_id": "team_1", "project_name": "SMR Work"},
        ),
    ]
    client.close()


def test_normalize_provider_funding_source_uses_canonical_smr_values() -> None:
    assert managed_research._normalize_provider_funding_source("synth") == "synth_managed"
    assert managed_research._normalize_provider_funding_source("byok") == "customer_byok"
    assert managed_research._normalize_provider_funding_source("chatgpt") == "user_connected"
    assert managed_research._normalize_provider_funding_source("customer_byok") == "customer_byok"
    with pytest.raises(ValueError):
        managed_research._normalize_provider_funding_source("invalid-source")


def test_list_jobs_calls_jobs_route_with_filters() -> None:
    client = SmrControlClient(api_key="test-key", backend_base="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_request_json(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> list[dict[str, Any]]:
        captured["method"] = method
        captured["path"] = path
        captured["params"] = params
        return [{"run_id": "run_1", "project_id": "proj_1", "state": "queued"}]

    client._request_json = fake_request_json  # type: ignore[method-assign]

    rows = client.list_jobs(
        project_id="proj_1",
        state="queued,executing",
        active_only=True,
        limit=25,
    )
    assert rows[0]["run_id"] == "run_1"
    assert captured == {
        "method": "GET",
        "path": "/smr/jobs",
        "params": {
            "limit": 25,
            "project_id": "proj_1",
            "state": "queued,executing",
            "active_only": True,
        },
    }
    client.close()
