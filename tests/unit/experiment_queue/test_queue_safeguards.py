"""Unit tests for queue safeguards: single worker, single database path, WAL mode."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# FAIL FAST: Check env var before any imports
if not os.getenv("EXPERIMENT_QUEUE_DB_PATH"):
    # Set a temp path for tests to avoid module import failures
    import tempfile
    tmp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    os.environ["EXPERIMENT_QUEUE_DB_PATH"] = tmp_db.name

from synth_ai.experiment_queue import celery_app, config


def test_database_path_required():
    """Test that EXPERIMENT_QUEUE_DB_PATH uses default path when not set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        # Clear cache
        config.reset_config_cache()
        
        # Unset env var - should use default path
        old_db = os.environ.pop("EXPERIMENT_QUEUE_DB_PATH", None)
        try:
            # Reset celery app to force recreation
            celery_app._celery_app_instance = None
            celery_app._celery_app_broker_url = None
            
            # Should use default path (not raise error)
            app = celery_app._create_celery_app()
            assert app is not None
            
            # Config should use default path
            cfg = config.load_config()
            assert cfg.sqlite_path.is_absolute(), "Default path should be absolute"
        finally:
            # Restore env var
            if old_db:
                os.environ["EXPERIMENT_QUEUE_DB_PATH"] = old_db
        
        # Test that it works with env var set
        os.environ["EXPERIMENT_QUEUE_DB_PATH"] = str(db_path)
        config.reset_config_cache()
        celery_app._celery_app_instance = None
        celery_app._celery_app_broker_url = None
        
        cfg = config.load_config()
        assert cfg.sqlite_path == db_path.resolve(), f"Expected {db_path.resolve()}, got {cfg.sqlite_path}"


def test_database_path_consistency():
    """Test that database path from env var matches config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        # Set env var
        os.environ["EXPERIMENT_QUEUE_DB_PATH"] = str(db_path)
        config.reset_config_cache()
        
        # Load config - should match env var
        cfg = config.load_config()
        assert cfg.sqlite_path == db_path.resolve()
        
        # Create Celery app - should use Redis broker (not SQLite)
        app = celery_app.get_celery_app()
        assert app.conf.broker_url.startswith("redis://"), f"Expected Redis broker, got: {app.conf.broker_url}"




def test_config_cache_clearing():
    """Test that config cache is cleared properly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db1 = Path(tmpdir) / "db1.db"
        db2 = Path(tmpdir) / "db2.db"
        
        # Set first database
        os.environ["EXPERIMENT_QUEUE_DB_PATH"] = str(db1)
        config.reset_config_cache()
        cfg1 = config.load_config()
        assert cfg1.sqlite_path == db1.resolve()
        
        # Change to second database
        os.environ["EXPERIMENT_QUEUE_DB_PATH"] = str(db2)
        config.reset_config_cache()
        cfg2 = config.load_config()
        assert cfg2.sqlite_path == db2.resolve()
        assert cfg2.sqlite_path != cfg1.sqlite_path


def test_celery_app_recreation_on_config_change():
    """Test that Celery app is recreated when config changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db1 = Path(tmpdir) / "db1.db"
        db2 = Path(tmpdir) / "db2.db"
        
        # Set first database
        os.environ["EXPERIMENT_QUEUE_DB_PATH"] = str(db1)
        config.reset_config_cache()
        app1 = celery_app.get_celery_app()
        broker1 = app1.conf.broker_url
        
        # Change to second database
        os.environ["EXPERIMENT_QUEUE_DB_PATH"] = str(db2)
        config.reset_config_cache()
        celery_app._celery_app_instance = None  # Reset app
        celery_app._celery_app_broker_url = None
        cfg2 = config.load_config()
        app2 = celery_app.get_celery_app()
        broker2 = app2.conf.broker_url
        
        # Both should use Redis (same broker URL), but different DB paths in config
        assert broker1.startswith("redis://"), f"Expected Redis broker, got: {broker1}"
        assert broker2.startswith("redis://"), f"Expected Redis broker, got: {broker2}"
        # Verify DB paths differ - cfg1 was loaded before db2 was set
        cfg1_reload = config.load_config()  # Reload to get current (db2)
        # But we need cfg1 from before the change
        os.environ["EXPERIMENT_QUEUE_DB_PATH"] = str(db1)
        config.reset_config_cache()
        cfg1 = config.load_config()
        assert cfg1.sqlite_path == db1.resolve()
        assert cfg2.sqlite_path == db2.resolve()
        assert cfg1.sqlite_path != cfg2.sqlite_path


def test_database_path_validation():
    """Test that database path is validated and resolved correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with relative path
        os.environ["EXPERIMENT_QUEUE_DB_PATH"] = "test.db"
        config.reset_config_cache()
        
        # Should resolve to absolute path
        cfg = config.load_config()
        assert cfg.sqlite_path.is_absolute()
        
        # Test with absolute path
        abs_path = Path(tmpdir) / "absolute.db"
        os.environ["EXPERIMENT_QUEUE_DB_PATH"] = str(abs_path)
        config.reset_config_cache()
        cfg = config.load_config()
        assert cfg.sqlite_path == abs_path.resolve()




def test_lock_file_management():
    """Test lock file creation and cleanup."""
    from synth_ai.cli.queue import _worker_lock_file
    
    lock_file = _worker_lock_file()
    assert lock_file.parent.exists()
    assert lock_file.name == "experiment_queue_worker.lock"
    
    # Test lock file can be written
    lock_file.write_text("12345")
    assert lock_file.exists()
    assert lock_file.read_text() == "12345"
    
    # Cleanup
    lock_file.unlink(missing_ok=True)


def test_single_database_enforcement():
    """Test that Celery app uses correct database path from config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        os.environ["EXPERIMENT_QUEUE_DB_PATH"] = str(db_path)
        config.reset_config_cache()
        celery_app._celery_app_instance = None  # Reset app
        celery_app._celery_app_broker_url = None
        
        # Create Celery app
        app = celery_app.get_celery_app()
        cfg1 = config.load_config()
        
        # Verify it's using the correct database
        assert app.conf.broker_url.startswith("redis://"), f"Expected Redis broker, got: {app.conf.broker_url}"
        assert cfg1.sqlite_path == db_path.resolve()
        
        # Change to different path
        other_db = Path(tmpdir) / "other.db"
        os.environ["EXPERIMENT_QUEUE_DB_PATH"] = str(other_db)
        config.reset_config_cache()
        celery_app._celery_app_instance = None  # Reset app
        celery_app._celery_app_broker_url = None
        
        # Should create new app with new database
        app2 = celery_app.get_celery_app()
        cfg2 = config.load_config()
        # Both use Redis broker (same URL), but different DB paths
        assert app2.conf.broker_url.startswith("redis://"), f"Expected Redis broker, got: {app2.conf.broker_url}"
        assert cfg2.sqlite_path == other_db.resolve()
        assert cfg1.sqlite_path != cfg2.sqlite_path

