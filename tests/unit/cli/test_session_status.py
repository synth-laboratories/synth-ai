"""Tests for session status command."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest import mock
from uuid import UUID, uuid4

import pytest
from click.testing import CliRunner

from synth_ai.cli.commands.status.subcommands.session import session_status_cmd
from synth_ai.session.models import AgentSession, AgentSessionLimit, AgentSessionUsage


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def mock_session():
    """Create a mock session with limits."""
    org_id = uuid4()
    session_id = "sess_test123"
    
    limits = [
        AgentSessionLimit(
            limit_id=uuid4(),
            limit_type="hard",
            metric_type="cost_usd",
            limit_value=Decimal("20.0"),
            current_usage=Decimal("5.50"),
            warning_threshold=None,
            window_seconds=None,
        ),
        AgentSessionLimit(
            limit_id=uuid4(),
            limit_type="hard",
            metric_type="tokens",
            limit_value=Decimal("100000"),
            current_usage=Decimal("25000"),
            warning_threshold=None,
            window_seconds=None,
        ),
    ]
    
    usage = AgentSessionUsage(
        tokens=Decimal("25000"),
        cost_usd=Decimal("5.50"),
        gpu_hours=Decimal("0.5"),
        api_calls=10,
    )
    
    return AgentSession(
        session_id=session_id,
        org_id=org_id,
        user_id=None,
        api_key_id=None,
        created_at=datetime.utcnow(),
        expires_at=None,
        ended_at=None,
        status="active",
        limit_exceeded_reason=None,
        usage=usage,
        limits=limits,
        tracing_session_id=None,
        session_type="codex_agent",
        metadata={},
    )


def test_session_status_with_session_id(runner: CliRunner, mock_session):
    """Test session status command with explicit session ID."""
    with mock.patch("synth_ai.cli.commands.status.subcommands.session.resolve_backend_config") as mock_config, \
         mock.patch("synth_ai.cli.commands.status.subcommands.session.AgentSessionClient") as mock_client_class:
        
        mock_cfg = mock.Mock()
        mock_cfg.base_url = "https://api.example.com"
        mock_cfg.api_key = "test-key"
        mock_config.return_value = mock_cfg
        
        mock_client = mock.Mock()
        mock_client.get = mock.AsyncMock(return_value=mock_session)
        mock_client._http = mock.Mock()
        mock_client_class.return_value = mock_client
        
        result = runner.invoke(session_status_cmd, ["--session-id", "sess_test123"])
    
    assert result.exit_code == 0
    assert "sess_test123" in result.output
    assert "active" in result.output
    assert "$5.50" in result.output
    assert "25,000" in result.output
    assert "$20.00" in result.output
    assert "✓ OK" in result.output


def test_session_status_from_env(runner: CliRunner, mock_session, monkeypatch):
    """Test session status command using SYNTH_SESSION_ID from environment."""
    monkeypatch.setenv("SYNTH_SESSION_ID", "sess_env123")
    
    with mock.patch("synth_ai.cli.commands.status.subcommands.session.resolve_backend_config") as mock_config, \
         mock.patch("synth_ai.cli.commands.status.subcommands.session.AgentSessionClient") as mock_client_class:
        
        mock_cfg = mock.Mock()
        mock_cfg.base_url = "https://api.example.com"
        mock_cfg.api_key = "test-key"
        mock_config.return_value = mock_cfg
        
        mock_client = mock.Mock()
        mock_client.get = mock.AsyncMock(return_value=mock_session)
        mock_client._http = mock.Mock()
        mock_client.list = mock.AsyncMock(return_value=[])
        mock_client_class.return_value = mock_client
        
        result = runner.invoke(session_status_cmd)
    
    assert result.exit_code == 0
    assert "sess_test123" in result.output


def test_session_status_with_exceeded_limit(runner: CliRunner):
    """Test session status command shows warning when limit exceeded."""
    org_id = uuid4()
    session_id = "sess_exceeded"
    
    limits = [
        AgentSessionLimit(
            limit_id=uuid4(),
            limit_type="hard",
            metric_type="cost_usd",
            limit_value=Decimal("20.0"),
            current_usage=Decimal("20.0"),  # Exceeded!
            warning_threshold=None,
            window_seconds=None,
        ),
    ]
    
    usage = AgentSessionUsage(
        tokens=Decimal("0"),
        cost_usd=Decimal("20.0"),
        gpu_hours=Decimal("0"),
        api_calls=0,
    )
    
    session = AgentSession(
        session_id=session_id,
        org_id=org_id,
        user_id=None,
        api_key_id=None,
        created_at=datetime.utcnow(),
        expires_at=None,
        ended_at=None,
        status="limit_exceeded",
        limit_exceeded_reason="Cost limit exceeded",
        usage=usage,
        limits=limits,
        tracing_session_id=None,
        session_type="codex_agent",
        metadata={},
    )
    
    with mock.patch("synth_ai.cli.commands.status.subcommands.session.resolve_backend_config") as mock_config, \
         mock.patch("synth_ai.cli.commands.status.subcommands.session.AgentSessionClient") as mock_client_class:
        
        mock_cfg = mock.Mock()
        mock_cfg.base_url = "https://api.example.com"
        mock_cfg.api_key = "test-key"
        mock_config.return_value = mock_cfg
        
        mock_client = mock.Mock()
        mock_client.get = mock.AsyncMock(return_value=session)
        mock_client._http = mock.Mock()
        mock_client_class.return_value = mock_client
        
        result = runner.invoke(session_status_cmd, ["--session-id", session_id])
    
    assert result.exit_code == 0
    assert "✗ EXCEEDED" in result.output
    assert "WARNING: Session limits exceeded!" in result.output


def test_session_status_no_active_session(runner: CliRunner):
    """Test session status command when no active session exists."""
    with mock.patch("synth_ai.cli.commands.status.subcommands.session.resolve_backend_config") as mock_config, \
         mock.patch("synth_ai.cli.commands.status.subcommands.session.AgentSessionClient") as mock_client_class:
        
        mock_cfg = mock.Mock()
        mock_cfg.base_url = "https://api.example.com"
        mock_cfg.api_key = "test-key"
        mock_config.return_value = mock_cfg
        
        mock_client = mock.Mock()
        mock_http_context = mock.AsyncMock()
        mock_http_context.__aenter__ = mock.AsyncMock(return_value=mock_http_context)
        mock_http_context.__aexit__ = mock.AsyncMock(return_value=None)
        mock_client._http = mock_http_context
        mock_client.list = mock.AsyncMock(return_value=[])  # No active sessions
        mock_client_class.return_value = mock_client
        
        result = runner.invoke(session_status_cmd)
    
    assert result.exit_code == 0
    assert "No active session found" in result.output


def test_session_status_session_not_found(runner: CliRunner):
    """Test session status command when session doesn't exist."""
    from synth_ai.session.exceptions import SessionNotFoundError
    
    with mock.patch("synth_ai.cli.commands.status.subcommands.session.resolve_backend_config") as mock_config, \
         mock.patch("synth_ai.cli.commands.status.subcommands.session.AgentSessionClient") as mock_client_class:
        
        mock_cfg = mock.Mock()
        mock_cfg.base_url = "https://api.example.com"
        mock_cfg.api_key = "test-key"
        mock_config.return_value = mock_cfg
        
        mock_client = mock.Mock()
        mock_client.get = mock.AsyncMock(side_effect=SessionNotFoundError("sess_notfound"))
        mock_client._http = mock.Mock()
        mock_client_class.return_value = mock_client
        
        result = runner.invoke(session_status_cmd, ["--session-id", "sess_notfound"])
    
    assert result.exit_code == 0
    assert "Session not found" in result.output

