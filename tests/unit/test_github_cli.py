"""Unit tests for `synth-ai gh` CLI commands."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from synth_ai.cli.main import cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client(**method_returns):
    """Build a MagicMock SmrControlClient that returns preset data."""
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    for name, value in method_returns.items():
        getattr(client, name).return_value = value
    return client


CLIENT_PATH = "synth_ai.cli.commands.managed_research.SmrControlClient"


# ---------------------------------------------------------------------------
# `synth-ai gh --help`
# ---------------------------------------------------------------------------


def test_gh_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["gh", "--help"])
    assert result.exit_code == 0
    assert "Manage org-level GitHub connection" in result.output


# ---------------------------------------------------------------------------
# `synth-ai gh status`
# ---------------------------------------------------------------------------


@patch(CLIENT_PATH)
def test_gh_status_not_connected(mock_cls):
    mock_cls.return_value = _mock_client(
        github_org_status={"connected": False},
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["gh", "status", "--api-key", "sk_test", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["connected"] is False


@patch(CLIENT_PATH)
def test_gh_status_connected(mock_cls):
    mock_cls.return_value = _mock_client(
        github_org_status={
            "connected": True,
            "mode": "oauth",
            "github_user_login": "testuser",
        },
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["gh", "status", "--api-key", "sk_test", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["connected"] is True
    assert payload["github_user_login"] == "testuser"


# ---------------------------------------------------------------------------
# `synth-ai gh disconnect`
# ---------------------------------------------------------------------------


@patch(CLIENT_PATH)
def test_gh_disconnect(mock_cls):
    client = _mock_client(github_org_disconnect={"ok": True})
    mock_cls.return_value = client
    runner = CliRunner()
    result = runner.invoke(cli, ["gh", "disconnect", "--api-key", "sk_test", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    client.github_org_disconnect.assert_called_once()


# ---------------------------------------------------------------------------
# `synth-ai gh link <project_id>`
# ---------------------------------------------------------------------------


@patch(CLIENT_PATH)
def test_gh_link(mock_cls):
    client = _mock_client(
        github_link_org={"ok": True, "project_id": "proj-123", "github_user": "testuser"},
    )
    mock_cls.return_value = client
    runner = CliRunner()
    result = runner.invoke(cli, ["gh", "link", "proj-123", "--api-key", "sk_test", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["project_id"] == "proj-123"
    client.github_link_org.assert_called_once_with("proj-123")


# ---------------------------------------------------------------------------
# `synth-ai gh connect` (OAuth flow, mocked browser + HTTP server)
# ---------------------------------------------------------------------------


@patch(CLIENT_PATH)
def test_gh_connect_success(mock_cls):
    """Simulate successful OAuth: mock backend start, fake browser callback, mock exchange."""
    client = _mock_client(
        github_org_oauth_start={
            "authorize_url": "https://github.com/login/oauth/authorize?client_id=test&state=teststate",
        },
        github_org_oauth_callback={"ok": True, "github_user": "testuser"},
    )
    mock_cls.return_value = client

    # We need to intercept the HTTP server and browser open:
    # 1. Patch webbrowser.open to not actually open a browser
    # 2. Patch HTTPServer.handle_request to simulate receiving a callback
    import http.server

    original_init = http.server.HTTPServer.__init__

    def fake_handle_request(self):
        """Simulate GitHub calling back with code=goodcode&state=goodstate."""
        # Build a fake request to invoke the handler
        import io
        import socket

        # We'll directly set the captured dict by calling the handler
        class FakeRequest:
            def makefile(self, mode, bufsize=None):
                request_line = b"GET /callback?code=goodcode&state=goodstate HTTP/1.1\r\nHost: localhost\r\n\r\n"
                return io.BytesIO(request_line)

            def sendall(self, data):
                pass

            def getpeername(self):
                return ("127.0.0.1", 12345)

        # Invoke the handler directly
        self.RequestHandlerClass(FakeRequest(), ("127.0.0.1", 12345), self)

    with patch("webbrowser.open"), \
         patch.object(http.server.HTTPServer, "handle_request", fake_handle_request), \
         patch.object(http.server.HTTPServer, "server_close"):
        runner = CliRunner()
        result = runner.invoke(cli, ["gh", "connect", "--api-key", "sk_test", "--json"])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    client.github_org_oauth_start.assert_called_once()
    client.github_org_oauth_callback.assert_called_once()
    # Verify code and state were passed through
    call_kwargs = client.github_org_oauth_callback.call_args
    assert call_kwargs.kwargs["code"] == "goodcode"
    assert call_kwargs.kwargs["state"] == "goodstate"


@patch(CLIENT_PATH)
def test_gh_connect_no_authorize_url(mock_cls):
    """Backend returns no authorize_url → error."""
    client = _mock_client(
        github_org_oauth_start={"error": "missing client_id"},
    )
    mock_cls.return_value = client

    runner = CliRunner()
    result = runner.invoke(cli, ["gh", "connect", "--api-key", "sk_test"])
    assert result.exit_code != 0
    assert "authorize_url" in result.output.lower() or "error" in result.output.lower()


# ---------------------------------------------------------------------------
# `synth-ai managed-research github` (long form) also works
# ---------------------------------------------------------------------------


@patch(CLIENT_PATH)
def test_managed_research_github_status(mock_cls):
    """Verify the full path `managed-research github status` also works."""
    mock_cls.return_value = _mock_client(
        github_org_status={"connected": False},
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["managed-research", "github", "status", "--api-key", "sk_test", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["connected"] is False


# ---------------------------------------------------------------------------
# `synth-ai gh status` with --backend-url
# ---------------------------------------------------------------------------


@patch(CLIENT_PATH)
def test_gh_status_custom_backend_url(mock_cls):
    mock_cls.return_value = _mock_client(
        github_org_status={"connected": True, "mode": "oauth"},
    )
    runner = CliRunner()
    result = runner.invoke(cli, [
        "gh", "status",
        "--api-key", "sk_test",
        "--backend-url", "http://localhost:9999",
        "--json",
    ])
    assert result.exit_code == 0
    # Verify the client was constructed with the custom backend URL
    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args
    backend_base = call_kwargs.kwargs.get("backend_base", "")
    assert "localhost:9999" in backend_base, f"Expected localhost:9999 in backend_base, got: {call_kwargs}"


# ---------------------------------------------------------------------------
# Pretty output (non-JSON) mode
# ---------------------------------------------------------------------------


@patch(CLIENT_PATH)
def test_gh_status_pretty_output(mock_cls):
    """Without --json, output should be indented JSON."""
    mock_cls.return_value = _mock_client(
        github_org_status={"connected": True, "mode": "oauth", "github_user_login": "dev"},
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["gh", "status", "--api-key", "sk_test"])
    assert result.exit_code == 0
    # Pretty output has newlines and indentation
    assert "\n" in result.output
    payload = json.loads(result.output)
    assert payload["connected"] is True
