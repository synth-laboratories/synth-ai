"""Tests that SHOULD FAIL to verify safeguards are working correctly."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# FAIL FAST: Set temp DB path before imports to avoid module-level failures
if not os.getenv("EXPERIMENT_QUEUE_DB_PATH"):
    tmp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    os.environ["EXPERIMENT_QUEUE_DB_PATH"] = tmp_db.name

from synth_ai.cli.local.experiment_queue import celery_app, config


@pytest.mark.skip(reason="EXPERIMENT_QUEUE_DB_PATH now uses default path instead of raising error")
def test_should_fail_without_db_path():
    """This test SHOULD FAIL - app creation should require EXPERIMENT_QUEUE_DB_PATH."""
    # FAIL FAST: Save and remove env var immediately
    old_db = os.environ.pop("EXPERIMENT_QUEUE_DB_PATH", None)
    try:
        # Clear cache and reset app
        config.reset_config_cache()
        celery_app._celery_app_instance = None
        celery_app._celery_app_broker_url = None
        
        # FAIL FAST: Should raise error immediately, not create app
        with pytest.raises(RuntimeError, match="EXPERIMENT_QUEUE_DB_PATH") as exc_info:
            celery_app._create_celery_app()
        
        # Verify error message
        assert "EXPERIMENT_QUEUE_DB_PATH" in str(exc_info.value) or "REQUIRED" in str(exc_info.value)
    finally:
        # Restore env var
        if old_db:
            os.environ["EXPERIMENT_QUEUE_DB_PATH"] = old_db






def test_should_require_db_path_in_cli():
    """This test SHOULD FAIL - CLI should require EXPERIMENT_QUEUE_DB_PATH."""
    from synth_ai.cli.queue import start_cmd
    from click.testing import CliRunner
    
    # Unset env var
    if "EXPERIMENT_QUEUE_DB_PATH" in os.environ:
        del os.environ["EXPERIMENT_QUEUE_DB_PATH"]
    
    # Use CliRunner to properly invoke the command
    runner = CliRunner()
    result = runner.invoke(start_cmd, [
        "--concurrency", "1",
        "--loglevel", "info",
        "--pool", "prefork",
        "--no-beat",
        "--foreground",
    ])
    
    # This SHOULD raise ClickException
    assert result.exit_code != 0
    assert "EXPERIMENT_QUEUE_DB_PATH" in result.output or "must be set" in result.output


def test_celery_app_module_level_creation_bypasses_safeguards():
    """This test SHOULD FAIL - module-level app creation bypasses safeguards."""
    # The issue: celery_app is created at module import time
    # This means safeguards might not run if the module was already imported
    
    # Clear env var
    if "EXPERIMENT_QUEUE_DB_PATH" in os.environ:
        del os.environ["EXPERIMENT_QUEUE_DB_PATH"]
    
    # Force reload the module
    import importlib
    import synth_ai.cli.local.experiment_queue.celery_app as celery_module
    
    # Reset the module-level app
    celery_module._celery_app_instance = None
    celery_module._celery_app_broker_url = None
    
    # Reload module - this should trigger safeguards
    importlib.reload(celery_module)
    
    # The module-level celery_app should be a placeholder that raises on access
    # Try to access it - should raise RuntimeError
    with pytest.raises(RuntimeError, match="EXPERIMENT_QUEUE_DB_PATH"):
        _ = celery_module.celery_app.conf

