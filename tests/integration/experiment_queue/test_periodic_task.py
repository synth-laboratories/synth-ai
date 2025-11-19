"""Fast tests for periodic task - validate Beat scheduling separately."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

from synth_ai.experiment_queue import celery_app, config, database, models, service
from synth_ai.experiment_queue.schemas import ExperimentSubmitRequest

# Path to test config file
BANKING77_CONFIG = Path(__file__).parent.parent.parent.parent / "examples" / "blog_posts" / "langprobe" / "task_specific" / "banking77" / "banking77_gepa.toml"


@pytest.fixture(autouse=True)
def queue_env(tmp_path, monkeypatch):
    """Set up test environment."""
    db_root = tmp_path / "queue_db"
    monkeypatch.setenv("EXPERIMENT_QUEUE_DB_PATH", str(db_root))
    monkeypatch.setenv("EXPERIMENT_QUEUE_BROKER_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("EXPERIMENT_QUEUE_RESULT_BACKEND_URL", "redis://localhost:6379/1")
    config.reset_config_cache()
    
    # Reload celery_app first (it's imported by tasks)
    import synth_ai.experiment_queue.celery_app as celery_app_module
    importlib.reload(celery_app_module)
    
    # Then reload tasks module after env is set
    import synth_ai.experiment_queue.tasks as tasks_module
    importlib.reload(tasks_module)
    
    database.init_db()
    
    yield
    config.reset_config_cache()


@pytest.fixture
def queue_tasks():
    """Get tasks module (already loaded with correct env)."""
    from synth_ai.experiment_queue import tasks
    return tasks


def test_periodic_task_is_registered():
    """FAIL FAST: Test periodic task is registered in Beat schedule."""
    app = celery_app.get_celery_app()
    schedule = app.conf.beat_schedule
    
    assert "process-experiment-queue" in schedule
    task_config = schedule["process-experiment-queue"]
    assert task_config["task"] == "synth_ai.experiment_queue.process_experiment_queue"
    assert task_config["schedule"] == 5.0


def test_periodic_task_can_be_called(queue_tasks):
    """FAIL FAST: Test periodic task function can be called directly."""
    # Call it directly (without Celery)
    result = queue_tasks.process_experiment_queue()
    
    assert isinstance(result, dict)
    assert "dispatched" in result
    assert "experiments_checked" in result


def test_periodic_task_finds_queued_jobs(queue_tasks):
    """FAIL FAST: Test periodic task finds and processes queued jobs."""
    # Create experiment with queued job
    request = ExperimentSubmitRequest.model_validate({
        "name": "Test",
        "parallelism": 1,
        "jobs": [{
            "job_type": "gepa",
            "config_path": str(BANKING77_CONFIG),
        }],
    })
    
    experiment = service.create_experiment(request)
    
    # Call periodic task directly
    result = queue_tasks.process_experiment_queue()
    
    assert result["experiments_checked"] >= 1
    # Should have attempted to dispatch (may be 0 if parallelism limit reached)


def test_periodic_task_handles_no_jobs(queue_tasks):
    """FAIL FAST: Test periodic task handles empty queue."""
    result = queue_tasks.process_experiment_queue()
    
    assert result["dispatched"] == 0
    assert result["experiments_checked"] >= 0

