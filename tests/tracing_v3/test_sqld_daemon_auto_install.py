#!/usr/bin/env python3
"""
Integration tests for sqld daemon auto-install functionality.

Tests the daemon's ability to:
- Find existing sqld binaries in PATH
- Find sqld in common install locations
- Auto-install sqld when missing (interactive mode)
- Respect SYNTH_AI_AUTO_INSTALL_SQLD environment variable
- Provide quality error messages when installation fails
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synth_ai.tracing_v3.turso.daemon import SqldDaemon, start_sqld


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide a temporary database path for tests."""
    return tmp_path / "test_sqld.db"


@pytest.fixture
def mock_sqld_binary(tmp_path):
    """Create a fake sqld binary for testing."""
    fake_binary = tmp_path / "fake_sqld"
    fake_binary.write_text("#!/bin/sh\necho 'fake sqld'")
    fake_binary.chmod(0o755)
    return str(fake_binary)


class TestSqldDaemonBinaryDetection:
    """Tests for sqld binary detection logic."""

    def test_finds_sqld_in_path(self, temp_db_path, mock_sqld_binary, monkeypatch):
        """Test that daemon finds sqld when it's in PATH."""
        # Mock shutil.which to return our fake binary
        import synth_ai.tracing_v3.turso.daemon as daemon_module
        
        def mock_which(cmd):
            if cmd == "sqld":
                return mock_sqld_binary
            return None
        
        monkeypatch.setattr(daemon_module.shutil, "which", mock_which)
        
        # Create daemon (should find binary without installing)
        hrana_port = 18084
        http_port = 18085
        daemon = SqldDaemon(db_path=str(temp_db_path), hrana_port=hrana_port, http_port=http_port)
        
        assert daemon.binary_path == mock_sqld_binary

    def test_finds_libsql_server_in_path(self, temp_db_path, mock_sqld_binary, monkeypatch):
        """Test that daemon finds libsql-server alternative name."""
        import synth_ai.tracing_v3.turso.daemon as daemon_module
        
        def mock_which(cmd):
            if cmd == "sqld":
                return None
            if cmd == "libsql-server":
                return mock_sqld_binary
            return None
        
        monkeypatch.setattr(daemon_module.shutil, "which", mock_which)
        
        hrana_port = 18086
        http_port = 18087
        daemon = SqldDaemon(db_path=str(temp_db_path), hrana_port=hrana_port, http_port=http_port)
        
        assert daemon.binary_path == mock_sqld_binary

    def test_finds_sqld_in_common_location(self, temp_db_path, mock_sqld_binary, monkeypatch):
        """Test that daemon finds sqld in common install locations."""
        import synth_ai.tracing_v3.turso.daemon as daemon_module
        
        # Mock PATH lookup to fail
        monkeypatch.setattr(daemon_module.shutil, "which", lambda cmd: None)
        
        # Mock common location lookup to succeed
        def mock_find_sqld_binary():
            return mock_sqld_binary
        
        hrana_port = 18088
        http_port = 18089
        with patch("synth_ai.utils.sqld.find_sqld_binary", side_effect=mock_find_sqld_binary):
            daemon = SqldDaemon(db_path=str(temp_db_path), hrana_port=hrana_port, http_port=http_port)
            
            assert daemon.binary_path == mock_sqld_binary


class TestSqldDaemonAutoInstall:
    """Tests for auto-install functionality."""

    def test_auto_install_in_interactive_mode(self, temp_db_path, mock_sqld_binary, monkeypatch):
        """Test that daemon auto-installs sqld in interactive mode."""
        import synth_ai.tracing_v3.turso.daemon as daemon_module
        
        # Mock all detection to fail
        monkeypatch.setattr(daemon_module.shutil, "which", lambda cmd: None)
        
        # Mock interactive terminal
        monkeypatch.setattr(daemon_module.sys.stdin, "isatty", lambda: True)
        
        # Set auto-install enabled
        monkeypatch.setenv("SYNTH_AI_AUTO_INSTALL_SQLD", "true")
        
        install_called = {"called": False}
        
        def mock_find_sqld_binary():
            return None  # Not found in common locations
        
        def mock_install_sqld():
            install_called["called"] = True
            return mock_sqld_binary
        
        def mock_confirm(msg, default=True):
            return True  # User accepts
        
        hrana_port = 18090
        http_port = 18091
        with patch("synth_ai.utils.sqld.find_sqld_binary", side_effect=mock_find_sqld_binary):
            with patch("synth_ai.utils.sqld.install_sqld", side_effect=mock_install_sqld):
                with patch("click.confirm", side_effect=mock_confirm):
                    daemon = SqldDaemon(db_path=str(temp_db_path), hrana_port=hrana_port, http_port=http_port)
                    
                    assert install_called["called"]
                    assert daemon.binary_path == mock_sqld_binary

    def test_auto_install_respects_user_decline(self, temp_db_path, monkeypatch):
        """Test that daemon respects when user declines auto-install."""
        import synth_ai.tracing_v3.turso.daemon as daemon_module
        
        # Mock all detection to fail
        monkeypatch.setattr(daemon_module.shutil, "which", lambda cmd: None)
        monkeypatch.setattr(daemon_module.sys.stdin, "isatty", lambda: True)
        monkeypatch.setenv("SYNTH_AI_AUTO_INSTALL_SQLD", "true")
        
        def mock_find_sqld_binary():
            return None
        
        def mock_confirm(msg, default=True):
            return False  # User declines
        
        hrana_port = 18092
        http_port = 18093
        with patch("synth_ai.utils.sqld.find_sqld_binary", side_effect=mock_find_sqld_binary):
            with patch("click.confirm", side_effect=mock_confirm):
                with pytest.raises(RuntimeError, match="sqld binary not found"):
                    SqldDaemon(db_path=str(temp_db_path), hrana_port=hrana_port, http_port=http_port)

    def test_auto_install_disabled_via_env_var(self, temp_db_path, monkeypatch):
        """Test that auto-install can be disabled via environment variable."""
        import synth_ai.tracing_v3.turso.daemon as daemon_module
        
        # Mock all detection to fail
        monkeypatch.setattr(daemon_module.shutil, "which", lambda cmd: None)
        
        # Disable auto-install
        monkeypatch.setenv("SYNTH_AI_AUTO_INSTALL_SQLD", "false")
        
        def mock_find_sqld_binary():
            return None
        
        hrana_port = 18094
        http_port = 18095
        with patch("synth_ai.utils.sqld.find_sqld_binary", side_effect=mock_find_sqld_binary):
            with pytest.raises(RuntimeError, match="sqld binary not found"):
                SqldDaemon(db_path=str(temp_db_path), hrana_port=hrana_port, http_port=http_port)

    def test_auto_install_skipped_in_non_interactive(self, temp_db_path, monkeypatch):
        """Test that auto-install is skipped in non-interactive environments (CI/CD)."""
        import synth_ai.tracing_v3.turso.daemon as daemon_module
        
        # Mock all detection to fail
        monkeypatch.setattr(daemon_module.shutil, "which", lambda cmd: None)
        
        # Simulate non-interactive terminal
        monkeypatch.setattr(daemon_module.sys.stdin, "isatty", lambda: False)
        
        # Enable auto-install (but should be skipped due to non-interactive)
        monkeypatch.setenv("SYNTH_AI_AUTO_INSTALL_SQLD", "true")
        
        def mock_find_sqld_binary():
            return None
        
        hrana_port = 18096
        http_port = 18097
        with patch("synth_ai.utils.sqld.find_sqld_binary", side_effect=mock_find_sqld_binary):
            with pytest.raises(RuntimeError, match="sqld binary not found"):
                SqldDaemon(db_path=str(temp_db_path), hrana_port=hrana_port, http_port=http_port)


class TestSqldDaemonErrorMessages:
    """Tests for error message quality."""

    def test_error_message_mentions_install_options(self, temp_db_path, monkeypatch):
        """Test that error message provides multiple install options."""
        import synth_ai.tracing_v3.turso.daemon as daemon_module
        
        monkeypatch.setattr(daemon_module.shutil, "which", lambda cmd: None)
        monkeypatch.setenv("SYNTH_AI_AUTO_INSTALL_SQLD", "false")
        
        def mock_find_sqld_binary():
            return None
        
        hrana_port = 18098
        http_port = 18099
        with patch("synth_ai.utils.sqld.find_sqld_binary", side_effect=mock_find_sqld_binary):
            with pytest.raises(RuntimeError) as exc_info:
                SqldDaemon(db_path=str(temp_db_path), hrana_port=hrana_port, http_port=http_port)
            
            error_msg = str(exc_info.value)
            
            # Check that error message includes all expected content
            assert "synth-ai turso" in error_msg
            assert "brew install" in error_msg
            assert "CI/CD" in error_msg
            assert "SYNTH_AI_AUTO_INSTALL_SQLD" in error_msg

    def test_error_message_is_multiline(self, temp_db_path, monkeypatch):
        """Test that error message is formatted with multiple lines for readability."""
        import synth_ai.tracing_v3.turso.daemon as daemon_module
        
        monkeypatch.setattr(daemon_module.shutil, "which", lambda cmd: None)
        monkeypatch.setenv("SYNTH_AI_AUTO_INSTALL_SQLD", "false")
        
        def mock_find_sqld_binary():
            return None
        
        hrana_port = 18100
        http_port = 18101
        with patch("synth_ai.utils.sqld.find_sqld_binary", side_effect=mock_find_sqld_binary):
            with pytest.raises(RuntimeError) as exc_info:
                SqldDaemon(db_path=str(temp_db_path), hrana_port=hrana_port, http_port=http_port)
            
            error_msg = str(exc_info.value)
            
            # Error message should have multiple lines
            assert error_msg.count("\n") >= 5


@pytest.mark.integration
class TestSqldDaemonIntegration:
    """Integration tests for daemon lifecycle with auto-install."""

    def test_start_sqld_helper_with_auto_install(self, temp_db_path, mock_sqld_binary, monkeypatch):
        """Test that start_sqld helper works with auto-install."""
        import synth_ai.tracing_v3.turso.daemon as daemon_module
        
        # Mock to return our fake binary
        monkeypatch.setattr(daemon_module.shutil, "which", lambda cmd: mock_sqld_binary if cmd == "sqld" else None)
        
        # Create daemon via helper with explicit ports
        hrana_port = 18080
        http_port = 18081
        daemon = start_sqld(db_path=str(temp_db_path), hrana_port=hrana_port, http_port=http_port)
        
        assert daemon.binary_path == mock_sqld_binary
        assert daemon.get_hrana_port() == hrana_port
        assert daemon.get_http_port() == http_port

    def test_daemon_with_explicit_binary_path(self, temp_db_path, mock_sqld_binary):
        """Test that daemon accepts explicit binary path, bypassing auto-detection."""
        hrana_port = 18082
        http_port = 18083
        daemon = SqldDaemon(
            db_path=str(temp_db_path),
            hrana_port=hrana_port,
            http_port=http_port,
            binary_path=mock_sqld_binary
        )
        
        assert daemon.binary_path == mock_sqld_binary
        assert daemon.get_hrana_port() == hrana_port
        assert daemon.get_http_port() == http_port

    def test_daemon_port_configuration(self, temp_db_path, mock_sqld_binary, monkeypatch):
        """Test that daemon correctly configures hrana and http ports."""
        import synth_ai.tracing_v3.turso.daemon as daemon_module
        
        monkeypatch.setattr(daemon_module.shutil, "which", lambda cmd: mock_sqld_binary if cmd == "sqld" else None)
        
        hrana_port = 18080
        http_port = 18081
        
        daemon = SqldDaemon(
            db_path=str(temp_db_path),
            hrana_port=hrana_port,
            http_port=http_port
        )
        
        assert daemon.get_hrana_port() == hrana_port
        assert daemon.get_http_port() == http_port


@pytest.mark.integration
@pytest.mark.slow
class TestSqldDaemonRealBinary:
    """Integration tests using real sqld binary (if installed)."""

    @pytest.mark.skipif(
        not os.getenv("SYNTH_AI_TEST_REAL_SQLD"),
        reason="Set SYNTH_AI_TEST_REAL_SQLD=1 to run tests with real sqld binary"
    )
    def test_real_sqld_detection(self, temp_db_path):
        """Test detection with real sqld binary."""
        import shutil
        
        real_sqld = shutil.which("sqld") or shutil.which("libsql-server")
        if not real_sqld:
            pytest.skip("sqld not installed on system")
        
        hrana_port = 28080
        http_port = 28081
        daemon = SqldDaemon(db_path=str(temp_db_path), hrana_port=hrana_port, http_port=http_port)
        
        assert daemon.binary_path == real_sqld

    @pytest.mark.skipif(
        not os.getenv("SYNTH_AI_TEST_REAL_SQLD"),
        reason="Set SYNTH_AI_TEST_REAL_SQLD=1 to run tests with real sqld binary"
    )
    def test_real_sqld_start_and_stop(self, temp_db_path):
        """Test starting and stopping real sqld daemon."""
        import shutil
        import time
        
        real_sqld = shutil.which("sqld") or shutil.which("libsql-server")
        if not real_sqld:
            pytest.skip("sqld not installed on system")
        
        # Use high port numbers to avoid conflicts
        hrana_port = 28080
        http_port = 28081
        
        daemon = start_sqld(db_path=str(temp_db_path), hrana_port=hrana_port, http_port=http_port)
        
        try:
            # Give it a moment to start
            time.sleep(0.5)
            
            assert daemon.is_running()
            assert daemon.get_hrana_port() == hrana_port
            assert daemon.get_http_port() == http_port
        finally:
            daemon.stop()
            time.sleep(0.1)
            assert not daemon.is_running()


if __name__ == "__main__":
    # Allow running directly with pytest
    pytest.main([__file__, "-v"])

