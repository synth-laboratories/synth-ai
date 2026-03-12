from __future__ import annotations

from typing import Any

from synth_ai.mcp.managed_research_server import ManagedResearchMcpServer


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def github_org_status(self) -> dict[str, Any]:
        self.calls.append(("github_org_status", {}))
        return {"ok": True}

    def github_org_oauth_start(self, *, redirect_uri: str | None = None) -> dict[str, Any]:
        self.calls.append(("github_org_oauth_start", {"redirect_uri": redirect_uri}))
        return {"ok": True}

    def github_org_oauth_callback(
        self,
        *,
        code: str,
        state: str | None = None,
        redirect_uri: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "github_org_oauth_callback",
                {"code": code, "state": state, "redirect_uri": redirect_uri},
            )
        )
        return {"ok": True}

    def github_org_disconnect(self) -> dict[str, Any]:
        self.calls.append(("github_org_disconnect", {}))
        return {"ok": True}

    def linear_status(self, project_id: str) -> dict[str, Any]:
        self.calls.append(("linear_status", {"project_id": project_id}))
        return {"ok": True}

    def linear_oauth_start(
        self,
        *,
        project_id: str,
        redirect_uri: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "linear_oauth_start",
                {"project_id": project_id, "redirect_uri": redirect_uri},
            )
        )
        return {"ok": True}

    def linear_oauth_callback(
        self,
        *,
        project_id: str,
        code: str,
        state: str | None = None,
        redirect_uri: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "linear_oauth_callback",
                {
                    "project_id": project_id,
                    "code": code,
                    "state": state,
                    "redirect_uri": redirect_uri,
                },
            )
        )
        return {"ok": True}

    def linear_disconnect(self, project_id: str) -> dict[str, Any]:
        self.calls.append(("linear_disconnect", {"project_id": project_id}))
        return {"ok": True}

    def linear_list_teams(self, project_id: str) -> dict[str, Any]:
        self.calls.append(("linear_list_teams", {"project_id": project_id}))
        return {"ok": True}

    def get_run_usage(
        self,
        run_id: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(("get_run_usage", {"run_id": run_id, "project_id": project_id}))
        return {"ok": True, "run_id": run_id}


def test_mcp_registers_integration_lifecycle_tools() -> None:
    server = ManagedResearchMcpServer()
    names = set(server.available_tool_names())
    assert {
        "smr_github_org_status",
        "smr_github_org_oauth_start",
        "smr_github_org_oauth_callback",
        "smr_github_org_disconnect",
        "smr_linear_status",
        "smr_linear_oauth_start",
        "smr_linear_oauth_callback",
        "smr_linear_disconnect",
        "smr_linear_list_teams",
        "smr_get_run_usage",
    }.issubset(names)


def test_mcp_integration_handlers_delegate_to_sdk_methods() -> None:
    server = ManagedResearchMcpServer()
    fake = _FakeClient()
    server._client_from_args = lambda args: fake  # type: ignore[method-assign]

    assert server._tool_github_org_status({}) == {"ok": True}
    assert (
        server._tool_github_org_oauth_start({"redirect_uri": "http://localhost:8765/callback"})
        == {"ok": True}
    )
    assert (
        server._tool_github_org_oauth_callback(
            {
                "code": "oauth-code",
                "state": "oauth-state",
                "redirect_uri": "http://localhost:8765/callback",
            }
        )
        == {"ok": True}
    )
    assert server._tool_github_org_disconnect({}) == {"ok": True}
    assert server._tool_linear_status({"project_id": "proj_123"}) == {"ok": True}
    assert (
        server._tool_linear_oauth_start(
            {"project_id": "proj_123", "redirect_uri": "http://localhost:8765/callback"}
        )
        == {"ok": True}
    )
    assert (
        server._tool_linear_oauth_callback(
            {
                "project_id": "proj_123",
                "code": "oauth-code",
                "state": "oauth-state",
                "redirect_uri": "http://localhost:8765/callback",
            }
        )
        == {"ok": True}
    )
    assert server._tool_linear_disconnect({"project_id": "proj_123"}) == {"ok": True}
    assert server._tool_linear_list_teams({"project_id": "proj_123"}) == {"ok": True}
    assert (
        server._tool_get_run_usage({"run_id": "run_123", "project_id": "proj_123"})
        == {"ok": True, "run_id": "run_123"}
    )

    assert fake.calls == [
        ("github_org_status", {}),
        ("github_org_oauth_start", {"redirect_uri": "http://localhost:8765/callback"}),
        (
            "github_org_oauth_callback",
            {
                "code": "oauth-code",
                "state": "oauth-state",
                "redirect_uri": "http://localhost:8765/callback",
            },
        ),
        ("github_org_disconnect", {}),
        ("linear_status", {"project_id": "proj_123"}),
        (
            "linear_oauth_start",
            {"project_id": "proj_123", "redirect_uri": "http://localhost:8765/callback"},
        ),
        (
            "linear_oauth_callback",
            {
                "project_id": "proj_123",
                "code": "oauth-code",
                "state": "oauth-state",
                "redirect_uri": "http://localhost:8765/callback",
            },
        ),
        ("linear_disconnect", {"project_id": "proj_123"}),
        ("linear_list_teams", {"project_id": "proj_123"}),
        ("get_run_usage", {"run_id": "run_123", "project_id": "proj_123"}),
    ]
