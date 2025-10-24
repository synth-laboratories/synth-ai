#!/usr/bin/env python3
"""
Tests for Turso embedded replica synchronization.
"""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

from ..config import CONFIG
from ..replica_sync import (
    ReplicaSync,
    get_replica_sync,
    start_replica_sync,
    stop_replica_sync,
)


@pytest.mark.asyncio
class TestReplicaSync:
    """Test embedded replica synchronization functionality."""

    @pytest_asyncio.fixture
    async def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest_asyncio.fixture
    async def mock_libsql_connection(self):
        """Mock libsql connection."""
        mock_conn = Mock()
        mock_conn.sync = Mock()
        mock_conn.close = Mock()
        return mock_conn

    async def test_replica_sync_init(self, temp_db):
        """Test ReplicaSync initialization."""
        sync = ReplicaSync(
            db_path=temp_db,
            sync_url="libsql://test.turso.io",
            auth_token="test-token",
            sync_interval=5,
        )

        assert sync.db_path == temp_db
        assert sync.sync_url == "libsql://test.turso.io"
        assert sync.auth_token == "test-token"
        assert sync.sync_interval == 5
        assert sync._sync_task is None
        assert sync._conn is None

    async def test_replica_sync_init_from_config(self, temp_db):
        """Test ReplicaSync initialization from CONFIG."""
        with patch.object(CONFIG, "sync_url", "libsql://config.turso.io"):
            with patch.object(CONFIG, "auth_token", "config-token"):
                with patch.object(CONFIG, "sync_interval", 10):
                    sync = ReplicaSync(db_path=temp_db)

                    assert sync.sync_url == "libsql://config.turso.io"
                    assert sync.auth_token == "config-token"
                    assert sync.sync_interval == 10

    @pytest.mark.fast
    async def test_sync_once_success(self, temp_db, mock_libsql_connection):
        """Test successful single sync operation."""
        with patch("libsql.connect", return_value=mock_libsql_connection):
            sync = ReplicaSync(
                db_path=temp_db, sync_url="libsql://test.turso.io", auth_token="test-token"
            )

            # Mock asyncio.to_thread
            with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
                result = await sync.sync_once()

                assert result is True
                mock_to_thread.assert_called_once_with(mock_libsql_connection.sync)

    async def test_sync_once_failure(self, temp_db):
        """Test failed sync operation."""
        sync = ReplicaSync(
            db_path=temp_db, sync_url="libsql://test.turso.io", auth_token="test-token"
        )

        # Mock connection to raise error
        with patch("libsql.connect", side_effect=Exception("Connection failed")):
            result = await sync.sync_once()

            assert result is False

    async def test_sync_once_no_url_error(self, temp_db):
        """Test sync_once with no sync_url configured."""
        sync = ReplicaSync(db_path=temp_db, sync_url="")

        # Should return False when sync_url is empty (ValueError is caught internally)
        result = await sync.sync_once()
        assert result is False

    @pytest.mark.fast
    async def test_start_background_sync(self, temp_db, mock_libsql_connection):
        """Test starting background sync task."""
        with patch("libsql.connect", return_value=mock_libsql_connection):
            sync = ReplicaSync(
                db_path=temp_db,
                sync_url="libsql://test.turso.io",
                auth_token="test-token",
                sync_interval=1,  # Short interval for testing
            )

            # Start background sync
            task = sync.start_background_sync()

            assert task is not None
            assert not task.done()
            assert sync._sync_task == task

            # Let it run for a bit
            await asyncio.sleep(0.2)

            # Stop it
            await sync.stop()

            assert task.done()

    async def test_start_background_sync_already_running(self, temp_db):
        """Test starting background sync when already running."""
        sync = ReplicaSync(db_path=temp_db, sync_url="libsql://test.turso.io", sync_interval=1)

        # Create a mock task that's not done
        mock_task = Mock()
        mock_task.done.return_value = False
        sync._sync_task = mock_task

        # Try to start again
        result = sync.start_background_sync()

        assert result == mock_task  # Should return existing task

    async def test_stop_sync(self, temp_db, mock_libsql_connection):
        """Test stopping sync task and closing connection."""
        with patch("libsql.connect", return_value=mock_libsql_connection):
            sync = ReplicaSync(
                db_path=temp_db, sync_url="libsql://test.turso.io", sync_interval=1
            )

            # Start and establish connection
            await sync.sync_once()
            task = sync.start_background_sync()

            # Stop
            await sync.stop()

            assert task.done()
            mock_libsql_connection.close.assert_called_once()
            assert sync._conn is None

    async def test_global_replica_sync(self, temp_db):
        """Test global replica sync management."""
        # Ensure we start clean
        await stop_replica_sync()

        with patch("libsql.connect") as mock_connect:
            mock_conn = Mock()
            mock_conn.sync = Mock()
            mock_conn.close = Mock()
            mock_connect.return_value = mock_conn

            # Start global sync
            with patch("asyncio.to_thread", new_callable=AsyncMock):
                sync = await start_replica_sync(
                    db_path=temp_db,
                    sync_url="libsql://test.turso.io",
                    auth_token="test-token",
                    sync_interval=1,
                )

                assert sync is not None
                assert get_replica_sync() == sync

                # Try to start again - should return existing
                sync2 = await start_replica_sync(db_path=temp_db)
                assert sync2 == sync

                # Stop global sync
                await stop_replica_sync()
                assert get_replica_sync() is None

    @pytest.mark.fast
    async def test_keep_fresh_error_handling(self, temp_db, mock_libsql_connection):
        """Test error handling in keep_fresh loop."""
        call_count = 0

        def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                # Raise error on second call
                raise Exception("Sync error")

        mock_libsql_connection.sync.side_effect = side_effect

        with patch("libsql.connect", return_value=mock_libsql_connection):
            sync = ReplicaSync(
                db_path=temp_db, sync_url="libsql://test.turso.io", sync_interval=0.05
            )

            # Start background sync
            task = sync.start_background_sync()

            # Let it run through a few iterations
            await asyncio.sleep(0.2)

            # Stop it
            await sync.stop()

            # Should have called sync multiple times despite error
            assert call_count >= 3
